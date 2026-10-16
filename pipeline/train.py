"""
Phase 3 (in practice): LoRA fine-tuning on a laptop CPU.

Trains a small LoRA adapter on top of Eason918/malicious-url-detector-v2
using:
  1. A real, publicly labeled URL dataset (same one used by
     pipeline/label_validation.py), with each example stripped down to a
     bare hostname -- matching the shape of real DNS queries, instead of
     full URLs-with-paths the base model originally trained on. This
     directly fixes the root cause behind the "google.com -> phishing"
     bug found in pipeline/inference.py (the trailing-slash normalization
     there is a stopgap; proper fine-tuning on hostname-shaped data is the
     real fix).
  2. A small set of hand-picked examples -- including the DNS record this
     pipeline previously (and incorrectly) flagged as phishing with 99%+
     confidence: "_9750._https.rsa.myroad-service.com". That's a
     DANE/HTTPS-service-binding style DNS record name, not a URL a user
     would browse to, and it's benign. We label it (and a few structurally
     similar synthetic variants) as benign so the adapter learns this
     DNS-specific naming pattern rather than just memorizing one string.

Runtime is kept laptop-CPU friendly: ~600 training examples, short sequence
length (hostnames are short), LoRA (only a few hundred thousand trainable
params instead of 82M), 2 epochs. Expect a few minutes on a modern laptop
CPU, no GPU required.
"""

from __future__ import annotations

import csv
import io
import random
import urllib.request
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from pipeline.inference import LABEL_MAPPING
from pipeline.logging_config import get_logger, setup_logging
from pipeline.model import LoRAConfig, apply_lora_adapter, load_foundation_model

logger = get_logger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/ommohan/malware-phishing-detector/"
    "main/data/malicious_phish.csv"
)

# Single source of truth for label<->id, derived from the mapping already
# confirmed empirically in pipeline/label_validation.py (LABEL_0=benign,
# LABEL_1=defacement, LABEL_2=malware, LABEL_3=phishing).
ID2LABEL: dict[int, str] = {int(k.split("_")[1]): v for k, v in LABEL_MAPPING.items()}
LABEL2ID: dict[str, int] = {v: k for k, v in ID2LABEL.items()}

ADAPTER_OUTPUT_DIR = "lora-adapter"

# The domain we picked from pipeline/observability.py's flagged output, plus
# a handful of synthetic look-alikes (same "_label._proto.host" DNS-record
# shape) so the adapter generalizes to the naming convention instead of
# memorizing one string. All are legitimate DNS service-binding/record
# patterns, not browsable phishing URLs -- hence "benign".
CUSTOM_EXAMPLES: dict[str, str] = {
    "_9750._https.rsa.myroad-service.com": "benign",
    "_443._tcp.rsa.example-service.com": "benign",
    "_dmarc.corp-mail-service.com": "benign",
    "_acme-challenge.cdn-edge-service.net": "benign",
    "_sip._tls.voip-provider.net": "benign",
}
CUSTOM_EXAMPLE_REPEATS = 8  # up-weight these few hand-picked examples


def extract_hostname(url: str) -> str:
    """Strip a dataset URL down to a bare hostname (no scheme/path/query).

    Mirrors exactly what pipeline/ingestion.py gives us from a DNS capture:
    just the hostname, nothing else.
    """
    url = url.strip()
    if "://" in url:
        url = url.split("://", 1)[1]
    hostname = url.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    return hostname.lower()


def load_dataset_split(
    dataset_url: str = DATASET_URL,
    train_per_class: int = 150,
    val_per_class: int = 30,
    seed: int = 7,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Stream the CSV once, reservoir-sample train+val per class, return
    (train_examples, val_examples) as (hostname, label) pairs."""
    rng = random.Random(seed)
    logger.info("Downloading training dataset from %s", dataset_url)
    with urllib.request.urlopen(dataset_url) as response:
        text = response.read().decode("utf-8", errors="ignore")

    want_per_class = train_per_class + val_per_class
    reservoirs: dict[str, list[str]] = defaultdict(list)
    seen_counts: dict[str, int] = defaultdict(int)

    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        url = (row.get("url") or "").strip()
        category = (row.get("type") or "").strip().lower()
        if not url or category not in LABEL2ID:
            continue

        hostname = extract_hostname(url)
        if not hostname:
            continue

        seen_counts[category] += 1
        bucket = reservoirs[category]
        if len(bucket) < want_per_class:
            bucket.append(hostname)
        else:
            j = rng.randint(0, seen_counts[category] - 1)
            if j < want_per_class:
                bucket[j] = hostname

    train_examples: list[tuple[str, str]] = []
    val_examples: list[tuple[str, str]] = []
    for category, hostnames in reservoirs.items():
        rng.shuffle(hostnames)
        train_examples += [(h, category) for h in hostnames[:train_per_class]]
        val_examples += [(h, category) for h in hostnames[train_per_class:want_per_class]]
        logger.info(
            "Category '%s': %d train, %d val (of %d seen in dataset)",
            category, min(train_per_class, len(hostnames)),
            max(0, len(hostnames) - train_per_class), seen_counts[category],
        )

    # Inject the hand-picked examples into training only (not validation --
    # we want val accuracy to reflect general performance, not memorization).
    for hostname, label in CUSTOM_EXAMPLES.items():
        train_examples += [(hostname, label)] * CUSTOM_EXAMPLE_REPEATS
    logger.info(
        "Added %d custom examples x%d repeats = %d rows to training set",
        len(CUSTOM_EXAMPLES), CUSTOM_EXAMPLE_REPEATS, len(CUSTOM_EXAMPLES) * CUSTOM_EXAMPLE_REPEATS,
    )

    rng.shuffle(train_examples)
    return train_examples, val_examples


class HostnameDataset(Dataset):
    """Tokenized (hostname, label) pairs for the HF Trainer."""

    def __init__(self, examples: list[tuple[str, str]], tokenizer, max_length: int = 32):
        hostnames = [h for h, _ in examples]
        self.labels = [LABEL2ID[label] for _, label in examples]
        self.encodings = tokenizer(
            hostnames, truncation=True, padding="max_length", max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    return {"accuracy": accuracy}


def train(
    train_per_class: int = 150,
    val_per_class: int = 30,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
) -> None:
    train_examples, val_examples = load_dataset_split(
        train_per_class=train_per_class, val_per_class=val_per_class,
    )
    logger.info("Total: %d train examples, %d val examples", len(train_examples), len(val_examples))

    tokenizer, base_model = load_foundation_model()
    model = apply_lora_adapter(base_model, LoRAConfig())
    model.to("cpu")  # keep this laptop-friendly and deterministic; no GPU/MPS required
    model.print_trainable_parameters()

    train_ds = HostnameDataset(train_examples, tokenizer)
    val_ds = HostnameDataset(val_examples, tokenizer)

    args = TrainingArguments(
        output_dir="lora-training-checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        use_cpu=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting LoRA fine-tuning (%d epochs, batch_size=%d)...", epochs, batch_size)
    trainer.train()

    metrics = trainer.evaluate()
    logger.info("Final validation metrics: %s", metrics)

    model.save_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
    logger.info("Saved LoRA adapter to ./%s", ADAPTER_OUTPUT_DIR)

    # Sanity-check the exact domain we picked, directly, post-training.
    model.eval()
    for hostname in CUSTOM_EXAMPLES:
        inputs = tokenizer(hostname, return_tensors="pt", truncation=True, max_length=32)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_id = int(logits.argmax(dim=-1))
        print(f"{hostname:<45} -> {ID2LABEL[pred_id]}")


if __name__ == "__main__":
    setup_logging()
    train()

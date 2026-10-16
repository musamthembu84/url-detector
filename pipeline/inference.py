"""
Phase 4: Low-Latency Deployment & Scale.

Standing in for the real TensorRT/Triton stack, this module batches
classifier calls (mirroring dynamic hardware batching) and reports p95
latency per string so you can see, locally, whether the plain HF pipeline
would meet the sub-15ms target called out in the lifecycle doc (it won't on
CPU -- that gap is exactly why the real system compiles to TensorRT and
serves on Triton/EKS instead of raw PyTorch).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from transformers import pipeline

from pipeline.logging_config import get_logger
from pipeline.model import MODEL_NAME

logger = get_logger(__name__)

# Confirmed empirically in pipeline/label_validation.py by running 100 real,
# labeled URLs (25/class) from the public "malicious_phish" dataset through
# this model and majority-voting the true category per raw LABEL_n output:
#   LABEL_0 -> benign     (92% purity)
#   LABEL_1 -> defacement (86% purity)
#   LABEL_2 -> malware    (100% purity)
#   LABEL_3 -> phishing   (76% purity)
# Re-run label_validation.py if the model is updated/swapped, since this
# mapping is tied to this specific model revision, not guaranteed elsewhere.
LABEL_MAPPING: dict[str, str] = {
    "LABEL_0": "benign",
    "LABEL_1": "defacement",
    "LABEL_2": "malware",
    "LABEL_3": "phishing",
}


@dataclass(frozen=True)
class ClassificationResult:
    domain: str
    label: str
    raw_label: str
    confidence: float
    latency_ms: float


class BatchClassifier:
    """Thin batching wrapper around the HF text-classification pipeline."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = 16,
        adapter_dir: str | None = None,
    ):
        """If adapter_dir is set, load the base model + a LoRA adapter saved
        by pipeline/train.py (e.g. "lora-adapter") instead of the stock
        model. The fine-tuned adapter is trained on bare hostnames, so it
        doesn't need the trailing-slash normalization below -- but we keep
        that normalization active regardless, since it's harmless for an
        adapter that already expects hostname-shaped input.
        """
        self.batch_size = batch_size
        logger.info("Loading model '%s' (batch_size=%d, adapter=%s)...", model_name, batch_size, adapter_dir)
        t0 = time.perf_counter()

        if adapter_dir:
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
            base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model = PeftModel.from_pretrained(base_model, adapter_dir)
            model.eval()
            self.classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                batch_size=batch_size,
            )
        else:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                truncation=True,
                batch_size=batch_size,
            )
        logger.info("Model loaded in %.2fs", time.perf_counter() - t0)

    def classify(self, domains: list[str]) -> list[ClassificationResult]:
        """Classify a list of hostnames/URLs.

        The model was fine-tuned on full URLs that almost always include a
        path (e.g. "amazon.com/dp/1569"), not bare hostnames. A bare domain
        with no path (exactly what raw DNS queries give us) structurally
        resembles the model's phishing training examples and gets
        misclassified regardless of actual reputation -- verified
        empirically: "google.com" -> phishing 99.9%, "google.com/" -> benign
        99.99%. We normalize by appending "/" to any path-less input so the
        shape matches what the model expects.
        """
        normalized = [d if "/" in d.split("://")[-1] else f"{d}/" for d in domains]
        logger.info("Classifying %d domains in batches of %d", len(domains), self.batch_size)
        results: list[ClassificationResult] = []
        num_batches = (len(domains) + self.batch_size - 1) // self.batch_size

        for batch_idx, start in enumerate(range(0, len(domains), self.batch_size), start=1):
            batch = domains[start : start + self.batch_size]
            norm_batch = normalized[start : start + self.batch_size]

            t0 = time.perf_counter()
            outputs = self.classifier(norm_batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_item_ms = elapsed_ms / len(batch)
            logger.debug(
                "Batch %d/%d: %d items in %.2fms (%.2fms/item)",
                batch_idx, num_batches, len(batch), elapsed_ms, per_item_ms,
            )

            for domain, output in zip(batch, outputs):
                raw_label = output["label"]
                mapped_label = LABEL_MAPPING.get(raw_label, raw_label)
                if raw_label not in LABEL_MAPPING:
                    logger.warning("Unmapped raw label '%s' for domain '%s' -- passing through as-is", raw_label, domain)
                logger.debug(
                    "%s -> raw=%s mapped=%s confidence=%.2f%%",
                    domain, raw_label, mapped_label, output["score"] * 100,
                )
                results.append(
                    ClassificationResult(
                        domain=domain,
                        label=mapped_label,
                        raw_label=raw_label,
                        confidence=output["score"],
                        latency_ms=per_item_ms,
                    )
                )

        logger.info("Classified %d domains in %d batches", len(results), num_batches)
        return results


def p95_latency_ms(results: list[ClassificationResult]) -> float:
    if not results:
        return 0.0
    latencies = sorted(r.latency_ms for r in results)
    index = min(int(len(latencies) * 0.95), len(latencies) - 1)
    return latencies[index]


if __name__ == "__main__":
    from pipeline.ingestion import sample_unique_queries
    from pipeline.logging_config import setup_logging

    setup_logging()
    sample = sample_unique_queries("sample_3.pcap", sample_size=32)
    batcher = BatchClassifier(batch_size=8)
    outcomes = batcher.classify(sample)

    for r in outcomes:
        print(f"{r.domain:<45} {r.label:<10} {r.confidence:.2%}  ({r.latency_ms:.2f} ms/item)")

    print(f"\np95 latency: {p95_latency_ms(outcomes):.2f} ms/item "
          f"(SLA target: <15 ms on Triton/TensorRT, not plain CPU pipeline)")
    print(f"mean: {statistics.mean(r.latency_ms for r in outcomes):.2f} ms/item")

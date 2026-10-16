"""
Label-validation script: empirically determine what LABEL_0..LABEL_3 mean
for Eason918/malicious-url-detector-v2 by running a real, publicly
labeled URL dataset through it and seeing which LABEL_n each true category
lands on most often.

Dataset: the well-known "malicious_phish.csv" set (url, type columns; types
are benign/defacement/phishing/malware) -- the same 4 classes this model's
config.json defines (LABEL_0..LABEL_3), mirrored across many public GitHub
repos derived from the original Kaggle release. We pull a small CSV mirror
and stratified-sample a bounded number of rows per class so this stays fast
and memory-light, instead of hand-crafted guesses at what phishing/malware
URLs "look like."
"""

from __future__ import annotations

import csv
import io
import random
import urllib.request
from collections import Counter, defaultdict

from pipeline.inference import BatchClassifier

DATASET_URL = (
    "https://raw.githubusercontent.com/ommohan/malware-phishing-detector/"
    "main/data/malicious_phish.csv"
)


def load_labeled_urls(
    dataset_url: str = DATASET_URL,
    samples_per_class: int = 25,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Download the CSV and return a stratified sample: {true_category: [urls]}.

    Reservoir-samples per class while streaming the CSV rows so we never
    need to hold the full dataset (or even a full per-class list beyond
    samples_per_class) in memory longer than necessary.
    """
    rng = random.Random(seed)
    with urllib.request.urlopen(dataset_url) as response:
        text = response.read().decode("utf-8", errors="ignore")

    reservoirs: dict[str, list[str]] = defaultdict(list)
    seen_counts: Counter = Counter()

    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        url = (row.get("url") or "").strip()
        category = (row.get("type") or "").strip().lower()
        if not url or not category:
            continue

        seen_counts[category] += 1
        bucket = reservoirs[category]
        if len(bucket) < samples_per_class:
            bucket.append(url)
        else:
            j = rng.randint(0, seen_counts[category] - 1)
            if j < samples_per_class:
                bucket[j] = url

    return dict(reservoirs)


def run_validation(
    category_samples: dict[str, list[str]],
    batch_size: int = 8,
) -> dict[str, Counter]:
    """Classify every sample and tally raw LABEL_n counts per true category."""
    classifier = BatchClassifier(batch_size=batch_size)
    tallies: dict[str, Counter] = defaultdict(Counter)

    for true_category, urls in category_samples.items():
        results = classifier.classify(urls)
        for url, result in zip(urls, results):
            tallies[true_category][result.label] += 1
            print(f"[{true_category:<10}] {url[:60]:<60} -> {result.label} ({result.confidence:.2%})")

    return tallies


def infer_label_mapping(tallies: dict[str, Counter]) -> dict[str, str]:
    """Majority-vote: map each LABEL_n to whichever true category produced it most."""
    label_votes: dict[str, Counter] = defaultdict(Counter)
    for true_category, label_counts in tallies.items():
        for label, count in label_counts.items():
            label_votes[label][true_category] += count

    mapping = {}
    for label, votes in label_votes.items():
        best_category, best_count = votes.most_common(1)[0]
        total = sum(votes.values())
        purity = best_count / total
        mapping[label] = best_category
        print(f"{label}: inferred = '{best_category}' "
              f"(purity {purity:.0%}, votes={dict(votes)})")

    return mapping


if __name__ == "__main__":
    print(f"Downloading and sampling labeled dataset from {DATASET_URL} ...")
    samples = load_labeled_urls(samples_per_class=25)
    for category, urls in samples.items():
        print(f"  {category:<10}: {len(urls)} sampled")

    tallies = run_validation(samples)
    print("\n--- Per-category label distribution ---")
    for category, counts in tallies.items():
        print(f"{category:<10}: {dict(counts)}")

    print("\n--- Inferred LABEL_n -> category mapping ---")
    mapping = infer_label_mapping(tallies)
    print(f"\nid2label candidate: {mapping}")
    print("Treat this as a hypothesis, not ground truth -- confirm with a "
          "larger, real labeled dataset before using it in production.")

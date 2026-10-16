"""
End-to-end demo: Phase 2 (ingestion) -> Phase 4 (inference) -> Phase 5
(observability), run against a bounded sample of sample_3.pcap so the full
256k+-packet capture is never loaded into memory at once.

Phase 3 (model.py) is used for architecture/LoRA inspection separately,
since this demo only needs inference, not retraining.
"""

from __future__ import annotations

import argparse
import logging
import os

from pipeline.ingestion import sample_unique_queries
from pipeline.inference import BatchClassifier, p95_latency_ms
from pipeline.logging_config import get_logger, setup_logging
from pipeline.observability import route_results, summarize

logger = get_logger(__name__)

# Default to the fine-tuned adapter from pipeline/train.py if it's been
# trained and saved; falls back to the stock model otherwise.
DEFAULT_ADAPTER_DIR = "lora-adapter" if os.path.isdir("lora-adapter") else None


def main() -> None:
    parser = argparse.ArgumentParser(description="DNS pcap -> classifier demo pipeline")
    parser.add_argument("--pcap", default="sample_3.pcap")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--adapter", default=DEFAULT_ADAPTER_DIR,
        help="Path to a LoRA adapter dir from pipeline/train.py (default: "
             f"{DEFAULT_ADAPTER_DIR!r} if present). Use --no-adapter to force the stock model.",
    )
    parser.add_argument(
        "--no-adapter", action="store_true",
        help="Ignore any default/configured adapter and use the stock base model.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-item DEBUG logs (raw label, per-batch timing, etc.)",
    )
    args = parser.parse_args()
    adapter_dir = None if args.no_adapter else args.adapter

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("=== Phase 2: Ingestion ===")
    domains = sample_unique_queries(args.pcap, sample_size=args.sample_size)
    print(f"[Phase 2] Got {len(domains)} domains, e.g. {domains[:3]}\n")

    logger.info("=== Phase 4: Inference ===")
    classifier = BatchClassifier(batch_size=args.batch_size, adapter_dir=adapter_dir)
    results = classifier.classify(domains)
    print(f"[Phase 4] p95 latency: {p95_latency_ms(results):.2f} ms/item\n")

    logger.info("=== Phase 5: Observability ===")
    routed = route_results(results)
    summarize(routed)


if __name__ == "__main__":
    main()

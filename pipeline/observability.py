"""
Phase 5: Autonomous Production Support, Observability & Feedback Loops.

Applies the soft-max confidence thresholds described in the lifecycle doc:
high-confidence predictions are auto-actioned, borderline ones are routed to
a human-review queue, and basic label-distribution stats are emitted as a
stand-in for the CloudWatch drift telemetry.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pipeline.inference import ClassificationResult
from pipeline.logging_config import get_logger

logger = get_logger(__name__)

AUTO_ACTION_THRESHOLD = 0.90
BENIGN_LABEL = "benign"


@dataclass(frozen=True)
class RoutedResult:
    result: ClassificationResult
    route: str  # "auto" or "human_review"


def route_results(
    results: list[ClassificationResult],
    threshold: float = AUTO_ACTION_THRESHOLD,
) -> list[RoutedResult]:
    """Split results into auto-actioned vs. human-review queues by confidence."""
    logger.info("Routing %d results with auto-action threshold=%.0f%%", len(results), threshold * 100)
    routed = []
    for r in results:
        route = "auto" if r.confidence >= threshold else "human_review"
        if r.label != BENIGN_LABEL:
            logger.info(
                "FLAGGED: %s predicted=%s confidence=%.2f%% route=%s",
                r.domain, r.label, r.confidence * 100, route,
            )
        if route == "human_review":
            logger.info(
                "-> human_review: %s predicted=%s confidence=%.2f%% (below %.0f%% threshold)",
                r.domain, r.label, r.confidence * 100, threshold * 100,
            )
        else:
            logger.debug("-> auto: %s predicted=%s confidence=%.2f%%", r.domain, r.label, r.confidence * 100)
        routed.append(RoutedResult(result=r, route=route))

    auto_count = sum(1 for x in routed if x.route == "auto")
    logger.info("Routing complete: %d auto-actioned, %d human_review", auto_count, len(routed) - auto_count)
    return routed


def label_distribution(results: list[ClassificationResult]) -> Counter:
    """Snapshot of predicted-label counts, the basic signal drift monitoring
    watches over time (a sudden shift here is what triggers retraining)."""
    dist = Counter(r.label for r in results)
    logger.info("Label distribution snapshot: %s", dict(dist))
    return dist


def summarize(routed: list[RoutedResult]) -> None:
    auto = [r for r in routed if r.route == "auto"]
    human = [r for r in routed if r.route == "human_review"]
    flagged = [r for r in routed if r.result.label != BENIGN_LABEL]

    print(f"Auto-actioned: {len(auto)}/{len(routed)}")
    for r in human:
        print(f"  -> human review: {r.result.domain} "
              f"({r.result.label}, {r.result.confidence:.2%})")

    print(f"\nFlagged (non-benign) domains: {len(flagged)}/{len(routed)}")
    for r in sorted(flagged, key=lambda r: -r.result.confidence):
        print(f"  -> {r.result.domain:<45} {r.result.label:<10} "
              f"{r.result.confidence:.2%}  [{r.route}]")

    dist = label_distribution([r.result for r in routed])
    print(f"\nLabel distribution (drift snapshot): {dict(dist)}")


if __name__ == "__main__":
    from pipeline.ingestion import sample_unique_queries
    from pipeline.inference import BatchClassifier
    from pipeline.logging_config import setup_logging

    setup_logging()
    sample = sample_unique_queries("sample_3.pcap", sample_size=32)
    batcher = BatchClassifier(batch_size=8)
    outcomes = batcher.classify(sample)

    routed = route_results(outcomes)
    summarize(routed)

"""
Phase 2: Ingestion & Data Pipeline Architecture.

Streams DNS query names out of a raw .pcap file one packet at a time so the
whole capture never has to sit in memory at once (important for large,
production-scale captures pulled from the data lake). Also applies the basic
curation step called out in the lifecycle doc: de-duplication and
normalization to cut down the extreme class imbalance between benign and
malicious query volume before anything reaches the model.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterator

import dpkt

from pipeline.logging_config import get_logger

logger = get_logger(__name__)

PROGRESS_LOG_INTERVAL = 25_000  # log a progress line every N packets scanned


@dataclass(frozen=True)
class DNSQuery:
    """A single normalized DNS question extracted from the capture."""

    name: str
    qtype: int
    timestamp: float


def iter_dns_queries(pcap_path: str) -> Iterator[DNSQuery]:
    """Lazily yield one DNSQuery per DNS question found in the pcap.

    Uses dpkt.pcap.Reader, which reads the file sequentially (a generator
    over the on-disk frames) instead of scapy's rdpcap(), which loads every
    packet into a list up front. This keeps memory flat regardless of
    capture size.
    """
    logger.info("Opening pcap file: %s", pcap_path)
    packets_scanned = 0
    dns_packets = 0
    questions_yielded = 0
    skipped_arpa = 0
    parse_errors = 0

    with open(pcap_path, "rb") as fh:
        reader = dpkt.pcap.Reader(fh)
        for timestamp, buf in reader:
            packets_scanned += 1
            if packets_scanned % PROGRESS_LOG_INTERVAL == 0:
                logger.info(
                    "...scanned %d packets (dns=%d, yielded=%d, arpa_skipped=%d, parse_errors=%d)",
                    packets_scanned, dns_packets, questions_yielded, skipped_arpa, parse_errors,
                )

            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                parse_errors += 1
                continue

            ip = eth.data
            if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                continue

            udp = ip.data
            if not isinstance(udp, dpkt.udp.UDP):
                continue
            if udp.sport != 53 and udp.dport != 53:
                continue

            try:
                dns = dpkt.dns.DNS(udp.data)
            except Exception:
                parse_errors += 1
                continue

            dns_packets += 1
            for question in dns.qd:
                name = question.name.rstrip(".").lower()
                # Skip reverse-DNS (PTR) lookups -- not URL-shaped strings,
                # so they're out of scope for the malicious-URL classifier.
                if name and not name.endswith((".in-addr.arpa", ".ip6.arpa")):
                    questions_yielded += 1
                    yield DNSQuery(name=name, qtype=question.type, timestamp=timestamp)
                elif name:
                    skipped_arpa += 1

    logger.info(
        "Finished scanning %s: %d packets total, %d DNS packets, %d questions yielded, "
        "%d arpa skipped, %d parse errors",
        pcap_path, packets_scanned, dns_packets, questions_yielded, skipped_arpa, parse_errors,
    )


def sample_unique_queries(
    pcap_path: str,
    sample_size: int = 500,
    seed: int = 42,
) -> list[str]:
    """Build a deduped, class-imbalance-aware sample of query names.

    Implements reservoir sampling over the *unique* hostname set so we read
    the pcap exactly once, never materialize the full 256k+ query list, and
    still get a uniform random sample suitable for feeding the classifier
    in Phase 3/4.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    reservoir: list[str] = []
    unique_count = 0
    swaps = 0

    logger.info("Sampling up to %d unique queries from %s (seed=%d)", sample_size, pcap_path, seed)
    t0 = time.perf_counter()

    for query in iter_dns_queries(pcap_path):
        if query.name in seen:
            continue
        seen.add(query.name)
        unique_count += 1

        if len(reservoir) < sample_size:
            reservoir.append(query.name)
        else:
            j = rng.randint(0, unique_count - 1)
            if j < sample_size:
                reservoir[j] = query.name
                swaps += 1

    elapsed = time.perf_counter() - t0
    logger.info(
        "Sampling complete in %.2fs: %d unique domains seen, %d kept in reservoir, %d swaps",
        elapsed, unique_count, len(reservoir), swaps,
    )
    return reservoir


if __name__ == "__main__":
    import sys

    from pipeline.logging_config import setup_logging

    setup_logging()
    pcap_file = sys.argv[1] if len(sys.argv) > 1 else "sample_3.pcap"
    sample = sample_unique_queries(pcap_file, sample_size=20)
    for domain in sample:
        print(domain)

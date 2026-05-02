from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MASK64 = (1 << 64) - 1


@dataclass
class StreamingStats:
    mode: str
    requested: int
    sieve_limit: int
    segment_size: int
    primes_analyzed: int = 0
    largest_prime: int = 0
    comparisons: int = 0
    increases: int = 0
    decreases: int = 0
    equals: int = 0
    segments_processed: int = 0
    prime_checksum: int = 0
    gap_checksum: int = 0
    started_at_utc: str = ""
    elapsed_seconds: float = 0.0
    python: str = ""
    numpy: str = ""
    platform: str = ""

    @property
    def strict_total(self) -> int:
        return self.increases + self.decreases

    @property
    def raw_imbalance(self) -> float:
        if self.comparisons == 0:
            return 0.0
        return abs(self.increases - self.decreases) / self.comparisons

    @property
    def strict_relative_imbalance(self) -> float:
        if self.strict_total == 0:
            return 0.0
        return abs(self.increases - self.decreases) / self.strict_total

    def to_json_dict(self) -> dict[str, int | float | str]:
        payload = asdict(self)
        payload["strict_total"] = self.strict_total
        payload["raw_imbalance"] = self.raw_imbalance
        payload["strict_relative_imbalance"] = self.strict_relative_imbalance
        return payload


def nth_prime_upper_bound(count: int) -> int:
    if count < 1:
        raise ValueError("count must be positive")
    small_bounds = [0, 2, 3, 5, 7, 11]
    if count < len(small_bounds):
        return small_bounds[count]

    n = float(count)
    return int(n * (math.log(n) + math.log(math.log(n)))) + 64


def small_primes_up_to(limit: int) -> np.ndarray:
    if limit < 2:
        return np.array([], dtype=np.int64)

    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[:2] = False
    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    return np.flatnonzero(is_prime).astype(np.int64)


def odd_segment_primes(low: int, high: int, base_primes: np.ndarray) -> np.ndarray:
    if high < 3 or high < low:
        return np.array([], dtype=np.int64)

    odd_low = max(3, low)
    if odd_low % 2 == 0:
        odd_low += 1

    odd_high = high if high % 2 else high - 1
    if odd_high < odd_low:
        return np.array([], dtype=np.int64)

    length = ((odd_high - odd_low) // 2) + 1
    is_prime = np.ones(length, dtype=np.bool_)

    for raw_prime in base_primes:
        p = int(raw_prime)
        if p == 2:
            continue
        p_squared = p * p
        if p_squared > odd_high:
            break

        start = max(p_squared, ((odd_low + p - 1) // p) * p)
        if start % 2 == 0:
            start += p

        index = (start - odd_low) // 2
        if index < length:
            is_prime[index::p] = False

    candidates = np.arange(odd_low, odd_high + 1, 2, dtype=np.int64)
    return candidates[is_prime]


def pct(value: int, total: int) -> str:
    if total == 0:
        return "0.00000%"
    return f"{100 * value / total:.5f}%"


def update_checksums(stats: StreamingStats, primes: np.ndarray, gaps: np.ndarray) -> None:
    for prime in primes:
        stats.prime_checksum = ((stats.prime_checksum * 1_315_423_911) + int(prime)) & MASK64
    for gap in gaps:
        stats.gap_checksum = ((stats.gap_checksum * 2_654_435_761) + int(gap)) & MASK64


def process_prime_batch(
    stats: StreamingStats,
    primes: np.ndarray,
    previous_prime: int | None,
    previous_gap: int | None,
) -> tuple[int | None, int | None]:
    if primes.size == 0:
        return previous_prime, previous_gap

    if previous_prime is None:
        joined = primes
    else:
        joined = np.concatenate((np.array([previous_prime], dtype=np.int64), primes))

    gaps = np.diff(joined)
    if previous_gap is None:
        comparable_gaps = gaps
    else:
        comparable_gaps = np.concatenate((np.array([previous_gap], dtype=np.int64), gaps))

    if comparable_gaps.size >= 2:
        changes = np.diff(comparable_gaps)
        stats.increases += int(np.count_nonzero(changes > 0))
        stats.decreases += int(np.count_nonzero(changes < 0))
        stats.equals += int(np.count_nonzero(changes == 0))
        stats.comparisons = stats.increases + stats.decreases + stats.equals

    update_checksums(stats, primes, gaps)
    stats.primes_analyzed += int(primes.size)
    stats.largest_prime = int(primes[-1])

    next_previous_prime = int(primes[-1])
    next_previous_gap = int(gaps[-1]) if gaps.size else previous_gap
    return next_previous_prime, next_previous_gap


def write_jsonl(path: Path, payload: dict[str, int | float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def maybe_checkpoint(
    stats: StreamingStats,
    checkpoint_file: Path | None,
    next_checkpoint: int,
    checkpoint_every: int,
    quiet: bool,
) -> int:
    if checkpoint_every <= 0:
        return next_checkpoint
    if stats.primes_analyzed < next_checkpoint:
        return next_checkpoint

    payload = stats.to_json_dict()
    payload["checkpoint_utc"] = datetime.now(timezone.utc).isoformat()
    if checkpoint_file is not None:
        write_jsonl(checkpoint_file, payload)
    if not quiet:
        print(
            f"checkpoint primes={stats.primes_analyzed:,} "
            f"largest={stats.largest_prime:,} "
            f"comparisons={stats.comparisons:,} "
            f"raw_imbalance={100 * stats.raw_imbalance:.5f}%"
        )

    while stats.primes_analyzed >= next_checkpoint:
        next_checkpoint += checkpoint_every
    return next_checkpoint


def run_streaming(
    *,
    mode: str,
    requested: int,
    sieve_limit: int,
    segment_size: int,
    checkpoint_every: int = 0,
    checkpoint_file: Path | None = None,
    quiet: bool = False,
) -> StreamingStats:
    started_at = datetime.now(timezone.utc)
    start_time = time.time()

    stats = StreamingStats(
        mode=mode,
        requested=requested,
        sieve_limit=sieve_limit,
        segment_size=segment_size,
        started_at_utc=started_at.isoformat(),
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
    )

    base_primes = small_primes_up_to(math.isqrt(sieve_limit) + 1)
    previous_prime: int | None = None
    previous_gap: int | None = None

    if sieve_limit >= 2:
        initial = np.array([2], dtype=np.int64)
        if mode == "count" and requested == 1:
            initial = initial[:1]
        previous_prime, previous_gap = process_prime_batch(stats, initial, previous_prime, previous_gap)

    next_checkpoint = checkpoint_every if checkpoint_every > 0 else 0
    next_checkpoint = maybe_checkpoint(stats, checkpoint_file, next_checkpoint, checkpoint_every, quiet)

    low = 3
    while low <= sieve_limit:
        high = min(low + segment_size - 1, sieve_limit)
        primes = odd_segment_primes(low, high, base_primes)
        stats.segments_processed += 1

        if mode == "count":
            remaining = requested - stats.primes_analyzed
            if remaining <= 0:
                break
            if primes.size > remaining:
                primes = primes[:remaining]

        previous_prime, previous_gap = process_prime_batch(stats, primes, previous_prime, previous_gap)
        stats.elapsed_seconds = time.time() - start_time
        next_checkpoint = maybe_checkpoint(stats, checkpoint_file, next_checkpoint, checkpoint_every, quiet)

        if mode == "count" and stats.primes_analyzed >= requested:
            break
        low = high + 1

    stats.elapsed_seconds = time.time() - start_time
    if mode == "count" and stats.primes_analyzed < requested:
        raise RuntimeError(
            f"sieve limit {sieve_limit:,} produced only {stats.primes_analyzed:,} primes; "
            "increase the limit estimator or run with --limit"
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming segmented runner for very large prime-gap comparisons."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--count", type=int, help="Analyze the first COUNT primes.")
    mode.add_argument("--limit", type=int, help="Analyze all primes up to LIMIT.")
    parser.add_argument(
        "--segment-size",
        type=int,
        default=50_000_000,
        help="Integer width of each sieve segment.",
    )
    parser.add_argument(
        "--checkpoint-every-primes",
        type=int,
        default=0,
        help="Emit a checkpoint every N primes. Disabled by default.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        help="Optional JSONL file for checkpoints.",
    )
    parser.add_argument("--json-summary", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress checkpoints on stdout.")
    args = parser.parse_args()

    if args.count is not None and args.count < 1:
        parser.error("--count must be positive")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    if args.segment_size < 1_000:
        parser.error("--segment-size must be at least 1000")
    if args.checkpoint_every_primes < 0:
        parser.error("--checkpoint-every-primes cannot be negative")

    return args


def print_summary(stats: StreamingStats) -> None:
    print(f"Mode             : {stats.mode}")
    print(f"Requested        : {stats.requested:,}")
    print(f"Sieve limit      : {stats.sieve_limit:,}")
    print(f"Segment size     : {stats.segment_size:,}")
    print(f"Segments         : {stats.segments_processed:,}")
    print(f"Primes analyzed  : {stats.primes_analyzed:,}")
    print(f"Largest prime    : {stats.largest_prime:,}")
    print(f"Comparisons      : {stats.comparisons:,}")
    print()
    print(f"d_(n+1) > d_n    : {stats.increases:,} ({pct(stats.increases, stats.comparisons)})")
    print(f"d_(n+1) < d_n    : {stats.decreases:,} ({pct(stats.decreases, stats.comparisons)})")
    print(f"d_(n+1) = d_n    : {stats.equals:,} ({pct(stats.equals, stats.comparisons)})")
    print()
    print(f"Raw imbalance    : {100 * stats.raw_imbalance:.5f}%")
    print(f"Strict relative  : {stats.strict_relative_imbalance:.8f}")
    print(f"Prime checksum   : {stats.prime_checksum:016x}")
    print(f"Gap checksum     : {stats.gap_checksum:016x}")
    print(f"Elapsed          : {stats.elapsed_seconds:.2f}s")
    print()
    print("Status           : streaming finite computation, not a proof")


def main() -> None:
    args = parse_args()
    if args.count is not None:
        mode = "count"
        requested = args.count
        sieve_limit = nth_prime_upper_bound(args.count)
    else:
        mode = "limit"
        requested = args.limit
        sieve_limit = args.limit

    stats = run_streaming(
        mode=mode,
        requested=requested,
        sieve_limit=sieve_limit,
        segment_size=args.segment_size,
        checkpoint_every=args.checkpoint_every_primes,
        checkpoint_file=args.checkpoint_file,
        quiet=args.quiet,
    )

    print_summary(stats)

    if args.json_summary is not None:
        args.json_summary.parent.mkdir(parents=True, exist_ok=True)
        args.json_summary.write_text(
            json.dumps(stats.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

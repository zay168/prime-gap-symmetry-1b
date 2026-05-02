from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from prime_gap_symmetry import consecutive_gaps, primes_by_count, sieve


@dataclass(frozen=True)
class StrictComparisonStats:
    comparisons: int
    increases: int
    decreases: int
    equals: int

    @property
    def strict_total(self) -> int:
        return self.increases + self.decreases

    @property
    def z_score(self) -> float:
        if self.strict_total == 0:
            return 0.0
        expected = self.strict_total / 2
        std_dev = math.sqrt(self.strict_total / 4)
        return (self.increases - expected) / std_dev


def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.00000%"
    return f"{100 * part / total:.5f}%"


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]

    covariance = sum(x * y for x, y in zip(centered_x, centered_y))
    variance_x = sum(x * x for x in centered_x)
    variance_y = sum(y * y for y in centered_y)

    if variance_x == 0 or variance_y == 0:
        return 0.0
    return covariance / math.sqrt(variance_x * variance_y)


def strict_comparisons(gaps: list[int]) -> StrictComparisonStats:
    increases = decreases = equals = 0
    for current_gap, next_gap in zip(gaps, gaps[1:]):
        if next_gap > current_gap:
            increases += 1
        elif next_gap < current_gap:
            decreases += 1
        else:
            equals += 1

    return StrictComparisonStats(increases + decreases + equals, increases, decreases, equals)


def normalized_strict_comparisons(primes: list[int], gaps: list[int]) -> StrictComparisonStats:
    normalized = [gap / math.log(prime) for gap, prime in zip(gaps, primes[:-1])]
    increases = decreases = equals = 0

    for current_gap, next_gap in zip(normalized, normalized[1:]):
        if next_gap > current_gap:
            increases += 1
        elif next_gap < current_gap:
            decreases += 1
        else:
            equals += 1

    return StrictComparisonStats(increases + decreases + equals, increases, decreases, equals)


def longest_equal_run(gaps: list[int]) -> tuple[int, int, int]:
    best_start = best_length = best_gap = 0
    index = 0

    while index < len(gaps):
        gap = gaps[index]
        length = 1
        while index + length < len(gaps) and gaps[index + length] == gap:
            length += 1

        if length > best_length:
            best_start = index
            best_length = length
            best_gap = gap

        index += length

    return best_start, best_length, best_gap


def conditional_next_gap(gaps: list[int], min_occurrences: int) -> list[dict[str, float | int]]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for current_gap, next_gap in zip(gaps, gaps[1:]):
        buckets[current_gap].append(next_gap)

    rows: list[dict[str, float | int]] = []
    for gap, next_gaps in sorted(buckets.items()):
        if len(next_gaps) < min_occurrences:
            continue

        increases = sum(1 for next_gap in next_gaps if next_gap > gap)
        decreases = sum(1 for next_gap in next_gaps if next_gap < gap)
        equals = len(next_gaps) - increases - decreases
        mean_next = sum(next_gaps) / len(next_gaps)

        rows.append(
            {
                "gap": gap,
                "count": len(next_gaps),
                "mean_next": mean_next,
                "mean_change": mean_next - gap,
                "increases": increases,
                "decreases": decreases,
                "equals": equals,
            }
        )

    return rows


def sliding_window_imbalances(gaps: list[int], window: int) -> list[tuple[int, int, int, int]]:
    if window < 2:
        return []

    rows: list[tuple[int, int, int, int]] = []
    step = max(1, window // 2)
    last_start = len(gaps) - window

    for start in range(0, max(0, last_start), step):
        sample = gaps[start : start + window]
        stats = strict_comparisons(sample)
        raw_imbalance = abs(stats.increases - stats.decreases)
        rows.append((start, stats.comparisons, raw_imbalance, stats.equals))

    rows.sort(key=lambda row: row[2] / row[1] if row[1] else 0, reverse=True)
    return rows


def print_comparison_block(label: str, stats: StrictComparisonStats) -> None:
    print(label)
    print(f"  comparisons : {stats.comparisons:,}")
    print(f"  increases   : {stats.increases:,} ({pct(stats.increases, stats.comparisons)})")
    print(f"  decreases   : {stats.decreases:,} ({pct(stats.decreases, stats.comparisons)})")
    print(f"  equals      : {stats.equals:,} ({pct(stats.equals, stats.comparisons)})")
    print(f"  sign z-score: {stats.z_score:.4f}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Honest diagnostics for consecutive prime-gap experiments."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--limit", type=int, help="Generate all primes up to this integer.")
    mode.add_argument("--count", type=int, help="Generate this many primes.")
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=100,
        help="Minimum current-gap occurrences for conditional tables.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5000,
        help="Sliding-window size for finite imbalance diagnostics.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 5:
        parser.error("--limit must be at least 5")
    if args.count is not None and args.count < 3:
        parser.error("--count must be at least 3")
    if args.min_occurrences < 1:
        parser.error("--min-occurrences must be positive")
    if args.window < 3:
        parser.error("--window must be at least 3")

    return args


def main() -> None:
    args = parse_args()
    primes = sieve(args.limit) if args.limit is not None else primes_by_count(args.count)
    gaps = consecutive_gaps(primes)

    print(f"Primes analyzed : {len(primes):,}")
    print(f"Largest prime   : {primes[-1]:,}")
    print(f"Gaps analyzed   : {len(gaps):,}")
    print()

    print_comparison_block("Raw gap comparisons", strict_comparisons(gaps))
    print_comparison_block("Normalized gap comparisons", normalized_strict_comparisons(primes, gaps))

    lag_correlation = pearson([float(gap) for gap in gaps[:-1]], [float(gap) for gap in gaps[1:]])
    print(f"Lag-1 Pearson correlation of raw gaps: {lag_correlation:.6f}")

    start, length, gap = longest_equal_run(gaps)
    print(f"Longest equal-gap run: start={start:,}, length={length:,}, gap={gap}")
    print()

    rows = conditional_next_gap(gaps, args.min_occurrences)
    print(f"Conditional next-gap table, min occurrences = {args.min_occurrences:,}")
    print("gap  count      mean next  mean change  P(>)       P(<)       P(=)")
    for row in rows[:20]:
        total = int(row["count"])
        print(
            f"{int(row['gap']):>3}  "
            f"{total:>9,}  "
            f"{float(row['mean_next']):>9.3f}  "
            f"{float(row['mean_change']):>11.3f}  "
            f"{pct(int(row['increases']), total):>9}  "
            f"{pct(int(row['decreases']), total):>9}  "
            f"{pct(int(row['equals']), total):>9}"
        )
    print()

    windows = sliding_window_imbalances(gaps, args.window)
    print(f"Largest finite-window imbalances, window = {args.window:,}")
    print("start      comparisons  raw imbalance  equals")
    for start, comparisons, imbalance, equals in windows[:10]:
        print(f"{start:>9,}  {comparisons:>11,}  {pct(imbalance, comparisons):>13}  {equals:>6,}")
    print()
    print("Status           : diagnostics only; finite-window patterns are not proofs")


if __name__ == "__main__":
    main()

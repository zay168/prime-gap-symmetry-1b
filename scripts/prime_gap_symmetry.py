from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GapStats:
    prime_count: int
    largest_prime: int
    comparisons: int
    increases: int
    decreases: int
    equals: int

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


def sieve(limit: int) -> list[int]:
    if limit < 2:
        return []

    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"

    root = math.isqrt(limit)
    for p in range(2, root + 1):
        if is_prime[p]:
            start = p * p
            is_prime[start : limit + 1 : p] = b"\x00" * (((limit - start) // p) + 1)

    return [n for n in range(2, limit + 1) if is_prime[n]]


def primes_by_count(count: int) -> list[int]:
    if count <= 0:
        return []
    if count < 6:
        limit = 15
    else:
        n = float(count)
        limit = int(n * (math.log(n) + math.log(math.log(n)))) + 16

    while True:
        primes = sieve(limit)
        if len(primes) >= count:
            return primes[:count]
        limit *= 2


def consecutive_gaps(primes: list[int]) -> list[int]:
    return [b - a for a, b in zip(primes, primes[1:])]


def compare_from_primes(primes: list[int]) -> GapStats:
    gaps = consecutive_gaps(primes)
    increases = decreases = equals = 0

    for current_gap, next_gap in zip(gaps, gaps[1:]):
        if next_gap > current_gap:
            increases += 1
        elif next_gap < current_gap:
            decreases += 1
        else:
            equals += 1

    return GapStats(
        prime_count=len(primes),
        largest_prime=primes[-1] if primes else 0,
        comparisons=increases + decreases + equals,
        increases=increases,
        decreases=decreases,
        equals=equals,
    )


def residue_breakdown(gaps: list[int]) -> dict[int, GapStats]:
    buckets: dict[int, list[tuple[int, int]]] = {residue: [] for residue in range(6)}
    for current_gap, next_gap in zip(gaps, gaps[1:]):
        residue = current_gap % 6
        buckets[residue].append((current_gap, next_gap))

    result: dict[int, GapStats] = {}
    for residue, pairs in buckets.items():
        increases = sum(1 for current_gap, next_gap in pairs if next_gap > current_gap)
        decreases = sum(1 for current_gap, next_gap in pairs if next_gap < current_gap)
        equals = len(pairs) - increases - decreases
        result[residue] = GapStats(0, 0, len(pairs), increases, decreases, equals)
    return result


def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.00000%"
    return f"{100 * part / total:.5f}%"


def print_stats(stats: GapStats) -> None:
    print(f"Primes analyzed : {stats.prime_count:,}")
    print(f"Largest prime   : {stats.largest_prime:,}")
    print(f"Comparisons     : {stats.comparisons:,}")
    print()
    print(f"d_(n+1) > d_n   : {stats.increases:,} ({pct(stats.increases, stats.comparisons)})")
    print(f"d_(n+1) < d_n   : {stats.decreases:,} ({pct(stats.decreases, stats.comparisons)})")
    print(f"d_(n+1) = d_n   : {stats.equals:,} ({pct(stats.equals, stats.comparisons)})")
    print()
    print(f"Raw imbalance   : {pct(abs(stats.increases - stats.decreases), stats.comparisons)}")
    print(f"Strict relative : {stats.strict_relative_imbalance:.8f}")


def print_residue_breakdown(gaps: list[int]) -> None:
    print()
    print("Residue breakdown by current gap modulo 6")
    print("residue  comparisons  increases   decreases   equals")
    for residue, stats in residue_breakdown(gaps).items():
        print(
            f"{residue:>7}  "
            f"{stats.comparisons:>11,}  "
            f"{stats.increases:>9,}  "
            f"{stats.decreases:>9,}  "
            f"{stats.equals:>7,}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare consecutive prime gaps.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--limit", type=int, help="Generate all primes up to this integer.")
    mode.add_argument("--count", type=int, help="Generate this many primes.")
    parser.add_argument("--no-residues", action="store_true", help="Skip modulo-6 breakdown.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 5:
        parser.error("--limit must be at least 5 to compare consecutive gaps.")
    if args.count is not None and args.count < 3:
        parser.error("--count must be at least 3 to compare consecutive gaps.")

    return args


def main() -> None:
    args = parse_args()

    if args.limit is not None:
        primes = sieve(args.limit)
    else:
        primes = primes_by_count(args.count)

    stats = compare_from_primes(primes)
    print_stats(stats)

    if not args.no_residues:
        print_residue_breakdown(consecutive_gaps(primes))


if __name__ == "__main__":
    main()

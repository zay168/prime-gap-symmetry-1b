from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from prime_gap_symmetry import consecutive_gaps, primes_by_count, sieve

STANDARD_RESIDUES = (0, 2, 4)


def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.00000%"
    return f"{100 * part / total:.5f}%"


def print_gap_residues(gaps: list[int]) -> None:
    ordinary_gaps = [gap for gap in gaps if gap % 2 == 0]
    residues = Counter(gap % 6 for gap in ordinary_gaps)
    total = sum(residues.values())

    print("Gap residues modulo 6")
    print("residue  count       frequency")
    for residue in STANDARD_RESIDUES:
        print(f"{residue:>7}  {residues[residue]:>10,}  {pct(residues[residue], total):>10}")


def print_transition_matrix(gaps: list[int]) -> None:
    transitions: dict[int, Counter[int]] = defaultdict(Counter)
    for current_gap, next_gap in zip(gaps, gaps[1:]):
        if current_gap % 2 or next_gap % 2:
            continue
        transitions[current_gap % 6][next_gap % 6] += 1

    print()
    print("Transition matrix P(next gap residue | current gap residue)")
    print("from\\to        0         2         4")
    for source in STANDARD_RESIDUES:
        total = sum(transitions[source].values())
        row = [pct(transitions[source][target], total).rjust(9) for target in STANDARD_RESIDUES]
        print(f"{source:>7}  {' '.join(row)}")


def print_comparison_by_residue(gaps: list[int]) -> None:
    buckets: dict[int, dict[str, int]] = {
        residue: {"increase": 0, "decrease": 0, "equal": 0} for residue in range(6)
    }

    for current_gap, next_gap in zip(gaps, gaps[1:]):
        if current_gap % 2 or next_gap % 2:
            continue
        bucket = buckets[current_gap % 6]
        if next_gap > current_gap:
            bucket["increase"] += 1
        elif next_gap < current_gap:
            bucket["decrease"] += 1
        else:
            bucket["equal"] += 1

    print()
    print("Strict comparisons grouped by current gap residue")
    print("residue  comparisons  increases   decreases   equals   raw imbalance")
    for residue in STANDARD_RESIDUES:
        data = buckets[residue]
        total = data["increase"] + data["decrease"] + data["equal"]
        imbalance = abs(data["increase"] - data["decrease"])
        print(
            f"{residue:>7}  "
            f"{total:>11,}  "
            f"{data['increase']:>9,}  "
            f"{data['decrease']:>9,}  "
            f"{data['equal']:>7,}  "
            f"{pct(imbalance, total):>13}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modulo-6 diagnostics for consecutive prime gaps.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--limit", type=int, help="Generate all primes up to this integer.")
    mode.add_argument("--count", type=int, help="Generate this many primes.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 5:
        parser.error("--limit must be at least 5")
    if args.count is not None and args.count < 3:
        parser.error("--count must be at least 3")

    return args


def main() -> None:
    args = parse_args()
    primes = sieve(args.limit) if args.limit is not None else primes_by_count(args.count)
    gaps = consecutive_gaps(primes)

    print(f"Primes analyzed : {len(primes):,}")
    print(f"Largest prime   : {primes[-1]:,}")
    print(f"Gaps analyzed   : {len(gaps):,}")
    print("Exceptional initial gap 1 is excluded from modulo-6 diagnostics.")
    print()

    print_gap_residues(gaps)
    print_transition_matrix(gaps)
    print_comparison_by_residue(gaps)
    print()
    print("Status           : residue diagnostics, not a proof")


if __name__ == "__main__":
    main()

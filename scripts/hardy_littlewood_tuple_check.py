from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TupleCheck:
    offsets: tuple[int, ...]
    limit: int
    admissible: bool
    singular_series: float
    observed: int
    predicted: float

    @property
    def ratio(self) -> float:
        if self.predicted == 0:
            return 0.0
        return self.observed / self.predicted


def sieve_bool(limit: int) -> bytearray:
    if limit < 2:
        return bytearray(limit + 1)

    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p * p : limit + 1 : p] = b"\x00" * (((limit - p * p) // p) + 1)
    return is_prime


def primes_up_to(limit: int) -> list[int]:
    flags = sieve_bool(limit)
    return [n for n in range(2, limit + 1) if flags[n]]


def normalize_offsets(offsets: list[int]) -> tuple[int, ...]:
    if not offsets:
        raise ValueError("at least one offset is required")
    if any(offset < 0 for offset in offsets):
        raise ValueError("offsets must be non-negative")
    return tuple(sorted(set(offsets)))


def is_admissible(offsets: tuple[int, ...]) -> bool:
    k = len(offsets)
    for p in primes_up_to(k):
        residues = {offset % p for offset in offsets}
        if len(residues) == p:
            return False
    return True


def singular_series(offsets: tuple[int, ...], prime_cutoff: int) -> float:
    k = len(offsets)
    product = 1.0

    for p in primes_up_to(prime_cutoff):
        residue_count = len({offset % p for offset in offsets})
        local_factor = (1.0 - residue_count / p) / ((1.0 - 1.0 / p) ** k)
        if local_factor == 0.0:
            return 0.0
        product *= local_factor

    return product


def count_prime_tuples(offsets: tuple[int, ...], limit: int) -> int:
    max_offset = max(offsets)
    flags = sieve_bool(limit + max_offset)

    count = 0
    for n in range(2, limit + 1):
        if all(flags[n + offset] for offset in offsets):
            count += 1
    return count


def logarithmic_integral_power(limit: int, power: int, steps: int) -> float:
    if limit <= 2:
        return 0.0

    width = (limit - 2.0) / steps
    total = 0.0
    for i in range(steps):
        midpoint = 2.0 + (i + 0.5) * width
        total += 1.0 / (math.log(midpoint) ** power)
    return total * width


def check_tuple(
    offsets: tuple[int, ...],
    limit: int,
    prime_cutoff: int,
    integral_steps: int,
) -> TupleCheck:
    series = singular_series(offsets, prime_cutoff)
    observed = count_prime_tuples(offsets, limit)
    predicted = series * logarithmic_integral_power(limit, len(offsets), integral_steps)

    return TupleCheck(
        offsets=offsets,
        limit=limit,
        admissible=is_admissible(offsets),
        singular_series=series,
        observed=observed,
        predicted=predicted,
    )


def parse_offsets(raw: str) -> tuple[int, ...]:
    try:
        offsets = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("offsets must be comma-separated integers") from exc

    try:
        return normalize_offsets(offsets)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite Hardy-Littlewood k-tuple sanity check."
    )
    parser.add_argument(
        "--offsets",
        type=parse_offsets,
        default=parse_offsets("0,2"),
        help="Comma-separated tuple offsets, for example 0,2 or 0,2,6.",
    )
    parser.add_argument("--limit", type=int, default=100_000, help="Count n <= limit.")
    parser.add_argument(
        "--prime-cutoff",
        type=int,
        default=10_000,
        help="Prime cutoff for the truncated singular series.",
    )
    parser.add_argument(
        "--integral-steps",
        type=int,
        default=20_000,
        help="Midpoint steps for the logarithmic integral approximation.",
    )
    args = parser.parse_args()

    if args.limit < 2:
        parser.error("--limit must be at least 2")
    if args.prime_cutoff < 2:
        parser.error("--prime-cutoff must be at least 2")
    if args.integral_steps < 100:
        parser.error("--integral-steps must be at least 100")

    return args


def main() -> None:
    args = parse_args()
    result = check_tuple(args.offsets, args.limit, args.prime_cutoff, args.integral_steps)

    print(f"Offsets          : {list(result.offsets)}")
    print(f"Limit            : {result.limit:,}")
    print(f"Admissible       : {result.admissible}")
    print(f"Singular series  : {result.singular_series:.8f}")
    print(f"Observed tuples  : {result.observed:,}")
    print(f"HL prediction    : {result.predicted:,.2f}")
    print(f"Observed/predict : {result.ratio:.6f}")
    print()
    print("Status           : finite numerical check, not a proof")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
import time

import numpy as np


def sieve(limit: int) -> np.ndarray:
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[:2] = False
    for p in range(2, math.isqrt(limit) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    return np.flatnonzero(is_prime)


def estimate_limit_for_nth_prime(count: int) -> int:
    if count < 6:
        return 15
    n = float(count)
    return int(n * (math.log(n) + math.log(math.log(n)))) + 1024


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Larger NumPy run for consecutive-prime-gap comparisons."
    )
    parser.add_argument("--count", type=int, default=1_000_000, help="Number of primes to analyze.")
    args = parser.parse_args()
    if args.count < 3:
        parser.error("--count must be at least 3 to compare consecutive gaps.")

    start = time.time()
    limit = estimate_limit_for_nth_prime(args.count)
    primes = sieve(limit)
    while len(primes) < args.count:
        limit *= 2
        primes = sieve(limit)
    primes = primes[: args.count]

    gaps = np.diff(primes)
    diffs = np.diff(gaps)

    increases = int(np.count_nonzero(diffs > 0))
    decreases = int(np.count_nonzero(diffs < 0))
    equals = int(np.count_nonzero(diffs == 0))
    total = increases + decreases + equals
    strict_total = increases + decreases
    raw_imbalance = abs(increases - decreases) / total
    strict_relative_imbalance = 0.0
    if strict_total:
        strict_relative_imbalance = abs(increases - decreases) / strict_total

    elapsed = time.time() - start

    print(f"Primes analyzed : {len(primes):,}")
    print(f"Largest prime   : {int(primes[-1]):,}")
    print(f"Comparisons     : {total:,}")
    print(f"d_(n+1) > d_n   : {increases:,} ({100 * increases / total:.5f}%)")
    print(f"d_(n+1) < d_n   : {decreases:,} ({100 * decreases / total:.5f}%)")
    print(f"d_(n+1) = d_n   : {equals:,} ({100 * equals / total:.5f}%)")
    print(f"Raw imbalance   : {100 * raw_imbalance:.5f}%")
    print(f"Strict relative : {strict_relative_imbalance:.8f}")
    print(f"Elapsed         : {elapsed:.2f}s")


if __name__ == "__main__":
    main()

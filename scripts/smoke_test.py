from __future__ import annotations

from hardy_littlewood_tuple_check import (
    count_prime_tuples,
    is_admissible,
    normalize_offsets,
    singular_series,
)
from gap_diagnostics import longest_equal_run, pearson, strict_comparisons
from prime_gap_symmetry import compare_from_primes, consecutive_gaps, primes_by_count, sieve
from streaming_large_runner import run_streaming


def test_sieve_small_range() -> None:
    assert sieve(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_count_generation() -> None:
    assert primes_by_count(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_gap_comparisons() -> None:
    primes = [2, 3, 5, 7, 11, 13]
    assert consecutive_gaps(primes) == [1, 2, 2, 4, 2]

    stats = compare_from_primes(primes)
    assert stats.comparisons == 4
    assert stats.increases == 2
    assert stats.decreases == 1
    assert stats.equals == 1


def test_hardy_littlewood_helpers() -> None:
    twin_offsets = normalize_offsets([0, 2])

    assert is_admissible(twin_offsets)
    assert not is_admissible(normalize_offsets([0, 1]))
    assert 1.2 < singular_series(twin_offsets, 1000) < 1.4
    assert count_prime_tuples(twin_offsets, 100) == 8


def test_gap_diagnostics_helpers() -> None:
    gaps = [2, 4, 4, 2, 6, 6, 6]
    stats = strict_comparisons(gaps)

    assert stats.increases == 2
    assert stats.decreases == 1
    assert stats.equals == 3
    assert longest_equal_run(gaps) == (4, 3, 6)
    assert pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_streaming_runner_small_count() -> None:
    stats = run_streaming(
        mode="count",
        requested=10,
        sieve_limit=30,
        segment_size=10,
        quiet=True,
    )

    assert stats.primes_analyzed == 10
    assert stats.largest_prime == 29
    assert stats.comparisons == 8
    assert stats.increases == 5
    assert stats.decreases == 2
    assert stats.equals == 1


def main() -> None:
    test_sieve_small_range()
    test_count_generation()
    test_gap_comparisons()
    test_hardy_littlewood_helpers()
    test_gap_diagnostics_helpers()
    test_streaming_runner_small_count()
    print("smoke tests passed")


if __name__ == "__main__":
    main()

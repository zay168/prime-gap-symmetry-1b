# Numerical results

## Reproducible local runs

Use the standard-library script:

```bash
python scripts/prime_gap_symmetry.py --limit 1000000
python scripts/prime_gap_symmetry.py --count 100000
```

The first command counts primes below a fixed integer limit. The second keeps increasing the sieve window until it has enough primes.

## Reported large run

A previous large run was reported over the first `10^9` primes.

| Metric | Value |
| --- | ---: |
| primes analyzed | `1,000,000,000` |
| largest prime | `22,801,763,489` |
| comparisons | `999,999,998` |
| $d_{n+1} > d_n$ | `48.84886%` |
| $d_{n+1} < d_n$ | `48.85126%` |
| $d_{n+1} = d_n$ | `2.29988%` |
| $\lvert \Pr(d_{n+1} > d_n) - \Pr(d_{n+1} < d_n) \rvert$ | `0.00240%` |

This repository does not include the full raw prime list or a machine-verifiable log for that run. Treat the table as a reported result, not as a CI-verified artifact.

## Caveats

- Strict increase and strict decrease are compared separately from equality.
- Equality events are not negligible at finite scale.
- The standard-library script is intended for reproducibility, not for billion-prime speed.
- The optional NumPy script is still a computational tool, not a proof.

## Large-run runner

Use the streaming runner for large finite computations:

```bash
python scripts/streaming_large_runner.py --count 1000000000 --segment-size 100000000 --checkpoint-every-primes 10000000 --checkpoint-file results/prime_gap_1b.jsonl --json-summary results/prime_gap_1b_summary.json
```

The runner keeps only the current sieve segment plus the previous prime and previous gap. This avoids storing the full prime list in memory and records checksums in the checkpoint/summary output.

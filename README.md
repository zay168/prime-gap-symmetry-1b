# Prime Gap Symmetry

Computational experiments around a simple question about consecutive prime gaps:

> how often is the next prime gap larger than the current one?

For each prime, define its gap as the distance to the next prime. This repository studies the empirical frequencies of three events:

- the next gap is larger than the current gap;
- the next gap is smaller than the current gap;
- the next gap is equal to the current gap.

The repository is intentionally conservative: it contains experiments, a heuristic note, and reproducibility scripts. It does **not** claim a proof of the prime-gap symmetry statement.

## Status

This is an exploratory computational project.

What is solid:

- exact integer experiments for small and medium ranges;
- a clean Python script for reproducing the basic counts;
- a streaming segmented runner for larger computations without storing all primes;
- a documented reported large run over the first `10^9` primes;
- an informal heuristic based on the Cramer-Gallagher model.

What is not claimed:

- no unconditional theorem;
- no proof of Hardy-Littlewood;
- no completed Lean formalization;
- no evidence that a Fourier/Riemann-zero pattern has been established.

## Quick start

```bash
python scripts/prime_gap_symmetry.py --limit 1000000
```

The script prints the number of primes below the limit, the strict comparison counts, and a small residue-class breakdown modulo 6.

For a count-based run:

```bash
python scripts/prime_gap_symmetry.py --count 100000
```

For larger runs without storing the full prime list:

```bash
python scripts/streaming_large_runner.py --count 1000000 --segment-size 5000000
```

For a long run with reproducibility logs:

```bash
python scripts/streaming_large_runner.py --count 1000000000 --segment-size 100000000 --checkpoint-every-primes 10000000 --checkpoint-file results/prime_gap_1b.jsonl --json-summary results/prime_gap_1b_summary.json
```

## Repository structure

```text
scripts/
  prime_gap_symmetry.py        Reproducible CPU experiment.
  large_numpy_verification.py  Optional larger NumPy experiment.
  streaming_large_runner.py    Streaming segmented runner for large runs.
  hardy_littlewood_tuple_check.py
                               Finite k-tuple sanity check.
  mod6_gap_analysis.py         Residue diagnostics for prime gaps.
  gap_diagnostics.py           Correlation, window, and conditional diagnostics.

docs/
  research_note.md             Heuristic model and mathematical context.
  results.md                   Reported numerical results and caveats.
  script_audit.md              Why older numbered scripts were removed.
  formalization_status.md      Why the previous Lean skeleton was removed.
  archive.md                   What was removed during cleanup and why.
```

## Main empirical observation

In large computations the strict increase and strict decrease frequencies are very close. Equality events remain visible at finite scale, so the raw strict frequencies need not be close to `50%` individually.

The reported large run over the first `10^9` primes gave:

| Event | Frequency |
| --- | ---: |
| next gap larger | `48.84886%` |
| next gap smaller | `48.85126%` |
| next gap equal | `2.29988%` |
| raw larger/smaller imbalance | `0.00240%` |

See [docs/results.md](docs/results.md).

## Heuristic

Under a strong random model for normalized prime gaps, two consecutive normalized gaps behave like exchangeable continuous random variables. Exchangeability gives

```text
P(Y > X) = P(Y < X).
```

This is a useful heuristic explanation for the near equality of strict increase and strict decrease frequencies. Turning this into a theorem about consecutive prime gaps is a much harder number-theoretic problem and is not done here.

See [docs/research_note.md](docs/research_note.md).

## Requirements

The main script uses only the Python standard library.

The optional NumPy scripts require:

```bash
pip install -r requirements.txt
```

Additional finite checks:

```bash
python scripts/hardy_littlewood_tuple_check.py --offsets 0,2 --limit 100000
python scripts/mod6_gap_analysis.py --count 100000
python scripts/gap_diagnostics.py --count 100000
```

## Citation

```bibtex
@software{alsarraf_prime_gap_symmetry_2026,
  author = {Al Sarraf, Zayd},
  title = {Prime Gap Symmetry: Computational Experiments},
  year = {2026},
  url = {https://github.com/zay168/prime-gap-symmetry-1b}
}
```

## License

Code and documentation are released under the MIT License.

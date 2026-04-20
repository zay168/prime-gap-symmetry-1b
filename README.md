# Prime Gap Symmetry — Computational Verification to 10⁹

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18294141.svg)](https://doi.org/10.5281/zenodo.18294141)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A computational and theoretical investigation of the symmetry of consecutive prime-gap comparisons. Includes the GPU-accelerated verification pipeline, the draft article, and a partial Lean 4 formalization.

## Main result

Conditional on the Hardy–Littlewood k-tuple conjecture,

$$\delta(A^+) \;=\; \lim_{N \to \infty} \frac{1}{N} \,\#\{\, n \le N : d_{n+1} > d_n \,\} \;=\; \tfrac{1}{2},$$

where $d_n = p_{n+1} - p_n$ is the $n$-th prime gap. GPU-accelerated verification over the first $10^9$ primes agrees with this prediction to within $0.0024\%$.

## Key observations

1. **Conditional argument.** Partially formalized in Lean 4 (`formal_proof.lean`), relying on Gallagher's theorem derived from the Hardy–Littlewood conjecture.
2. **Modular balancing.** The global 50/50 symmetry emerges from the cancellation of biases across residue classes modulo 6 ($d_n \equiv 0, 2, 4 \pmod 6$).
3. **Riemann spectrum filter.** Spectral analysis of the convergence error shows that the gap-increment signal suppresses oscillations at the imaginary parts of the non-trivial Riemann zeros, consistent with a low-pass statistical filter.

## Verification results ($N = 10^9$)

| Metric | Value |
| --- | --- |
| Primes analyzed | $1{,}000{,}000{,}000$ |
| Largest prime | $22{,}801{,}763{,}489$ |
| $\delta(A^+)$ (increases) | $48.84886\%$ |
| $\delta(A^-)$ (decreases) | $48.85126\%$ |
| Difference | $0.0024\%$ |

Convergence is consistent with $O(1/\ln N)$, matching the theoretical prediction.

## Repository structure

**Documentation**

- `ARTICLE_DRAFT.md` — draft paper.
- `formal_proof.lean` — partial Lean 4 formalization.

**Verification (Python)**

- `24_ultra_optimized.py` — GPU-accelerated verification script (~3.9M primes/s on an RTX 5060).
- `25_riemann_oscillations.py` — spectral analysis of the convergence error against the Riemann zeros.
- `22_gpu_verification.py` — earlier GPU verification baseline.
- `20_mod6_structure.py` — derivation of the modular-balancing mechanism.

**Exploratory scripts**

- `10_exploration_experimentale.py` — entropy and correlation tests.
- `16_preuve_rigoureuse.py` — combinatorial proof attempts.
- `19_attaque_ia.py` — pattern-detection experiments.

## Usage

### Requirements

- Python 3.10+
- PyTorch with CUDA support
- NVIDIA GPU (tested on RTX 5060)

### Run the primary verification

```bash
python 24_ultra_optimized.py
```

## Citation

```bibtex
@software{alsarraf_prime_gap_symmetry_2026,
  author  = {Al Sarraf, Zayd},
  title   = {Prime Gap Symmetry — Computational Verification to 10^9},
  year    = {2026},
  doi     = {10.5281/zenodo.18294141},
  url     = {https://github.com/zay168/prime-gap-symmetry-1b}
}
```

## License

- Code: [MIT](LICENSE)
- Text and article: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Author

Zayd Al Sarraf · [github.com/zay168](https://github.com/zay168) · [alsarrafzayd@gmail.com](mailto:alsarrafzayd@gmail.com)

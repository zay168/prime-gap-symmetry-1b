# Script audit

This document records why the old numbered Python scripts were removed from the active repository.

The central issue was not that every script was useless. Several contained workable computational ideas. The problem was that code, narration, speculation, and proof language were mixed together, making the repository look much less credible than the underlying experiments.

## Main code-level problems

- Many scripts had fixed heavy constants instead of command-line parameters.
- GPU scripts executed CUDA calls at import time and failed immediately on machines without CUDA.
- Several scripts printed mathematical conclusions that the code did not establish.
- Equality events were often folded into both `A+` and `A-`, then compared directly with `1/2`.
- Some scripts used finite-window statistics or fitted curves as if they implied asymptotic results.
- Generated files and cache artifacts were tracked in Git.

## Forensic classification

| Old file | Verdict | Reason | Replacement |
| --- | --- | --- | --- |
| `07_verification_numerique.py` | salvageable prototype | Basic counts are useful, but non-strict `>=` and `<=` were presented too close to `1/2`. | `scripts/prime_gap_symmetry.py` |
| `08_visualisation.py` | salvageable later | Plotting is useful, but it writes fixed PNG names and requires `matplotlib`. | none yet |
| `10_exploration_experimentale.py` | partial prototype | Some diagnostics are useful, but the summaries overinterpret finite statistics. | future diagnostics only |
| `11_symetrie_approfondie.py` | partial prototype | Conditional-gap analysis is interesting, but the martingale framing is too strong. | future diagnostics only |
| `13_attaque_massive.py` | remove | Treats stability tests, extrapolation, and finite checks as if they formed a proof strategy. | removed |
| `14_approche_asymptotique.py` | remove | Mostly narrative output and hardcoded conclusions. | removed |
| `15_resolution_mystere.py` | remove | Derives a simplified exponential-drift model, then presents it as an asymptotic proof. | removed |
| `16_preuve_rigoureuse.py` | remove | Prints a conditional proof, but skips the hard joint-distribution and dependence assumptions. | `docs/research_note.md` |
| `17_attaque_hardy_littlewood.py` | salvage core only | The singular-series and tuple-counting ideas are real, but the script is theatrical and concludes too much. | `scripts/hardy_littlewood_tuple_check.py` |
| `18_approche_directe.py` | remove | Mostly explanatory text about why the direct Poisson proof is not available. | removed |
| `19_attaque_ia.py` | remove | Pattern mining without validation, mixed with proof-search language. | removed |
| `20_mod6_structure.py` | salvage core only | Residue diagnostics are useful, but the script claims a proof path from mod 6 symmetry. | `scripts/mod6_gap_analysis.py` |
| `21_revolution.py` | remove | Conceptual brainstorming, not a reproducible research script. | removed |
| `22_gpu_verification.py` | remove | CUDA at import time, fixed 200M sieve, no CLI, non-portable. | removed |
| `23_extreme_verification.py` | remove | Attempts a 1B-prime run while storing huge lists in memory. | removed |
| `24_ultra_optimized.py` | rewritten | Better idea than `23`, but still top-level CUDA, fixed 1B target, and large concatenated arrays. | `scripts/streaming_large_runner.py` |
| `25_riemann_oscillations.py` | remove | Spectral experiment with weak statistical control and fixed output file. | removed |

## Script 17 specifically

The old Hardy-Littlewood script had one useful nucleus:

- admissibility testing;
- truncated singular series computation;
- finite counting of prime tuples.

But it also had serious credibility problems:

- the title framed the script as an attack on Hardy-Littlewood;
- the output said the conjecture was numerically exact or almost certainly true;
- the prediction used a finite truncated singular series without reporting the truncation caveat clearly;
- it mixed real finite checks with proof-structure narration;
- it suggested new proof directions without any actual mathematical bridge.

The replacement keeps only the finite numerical check:

```bash
python scripts/hardy_littlewood_tuple_check.py --offsets 0,2 --limit 100000
```

Its output explicitly says that the result is a finite numerical check, not a proof.

## Other scripts were affected too

`17_attaque_hardy_littlewood.py` was not an isolated problem. The same pattern appears across the old scripts in several forms.

### `10_exploration_experimentale.py`

Useful parts:

- Pearson correlation between consecutive gaps;
- entropy and repeated-pattern diagnostics;
- local transition tables.

Problems:

- `markov_transition_matrix` clips gaps above `max_gap`, but then weights the probabilities using the original unclipped gap values. All large gaps therefore fall back to a default probability of `0.5`, which distorts the reported global transition estimate.
- The Fourier test uses only a few low frequencies and no null model, so "no periodicity" or "dominant frequency" should not be treated as evidence.
- The summary says the symmetry of `d_{n+1} - d_n` naturally implies the desired density. That only follows from an exchangeability assumption that the script does not establish.
- The convergence check compares non-strict `>=` frequencies with `1/2`, which is misleading when equality events are still visible.

Replacement: `scripts/gap_diagnostics.py`.

### `11_symetrie_approfondie.py`

Useful parts:

- sign test after removing equality events;
- conditional next-gap behavior by current gap value.

Problems:

- A sign test can fail to reject asymmetry without proving symmetry.
- The "martingale" framing is too strong. The observed conditional table shows small gaps tend to increase and large gaps tend to decrease; that is regression toward the local mean, not evidence that `E[d_{n+1} | d_n = g] = g`.
- The final narrative says the effects cancel globally, but this cancellation is empirical and range-dependent.

Replacement: `scripts/gap_diagnostics.py`.

### `13_attaque_massive.py`

Problems:

- Finite stability tests are presented as if they were a proof strategy.
- Overlapping window tests are treated as independent when estimating expected anomalies.
- The "numerical induction" section is not induction; it is a running average.
- The extrapolation fits `|delta(N) - 1/2|` over a few checkpoints and interprets the fitted exponent asymptotically. That is not defensible.
- It again uses non-strict `>=` in places where strict comparisons should be separated from equality.

Replacement: removed; the finite-window part is now in `scripts/gap_diagnostics.py`.

### `14_approche_asymptotique.py`

Problems:

- The reversibility argument is asserted from intuition, not derived.
- The claim that equality events have a positive limiting density is unsupported and contradicted by the reported finite trend.
- It mixes a valid identity,
  `P(>=) = P(>) + P(=)`,
  with speculative assumptions about the limits.
- It hardcodes old numerical values and then reasons from them.

Replacement: removed; the honest version is the exchangeability heuristic in `docs/research_note.md`.

### `15_resolution_mystere.py`

Problems:

- The exponential-drift model is useful as a toy model, but the script turns it into an asymptotic proof.
- It assumes independence of consecutive normalized gaps.
- The normalized comparison changes the treatment of raw equal gaps because `d_n / log(p_n)` and `d_{n+1} / log(p_{n+1})` are almost never exactly equal. This is a different statistic, not a repaired version of the original one.
- It invokes a law of large numbers without proving the required dependence conditions.

Replacement: `scripts/gap_diagnostics.py` reports normalized comparisons, but labels them only as diagnostics.

### `16_preuve_rigoureuse.py`

Problems:

- Gallagher-style Poisson behavior is used as if it gave i.i.d. consecutive gaps.
- The hard missing object is the joint limiting law of consecutive normalized gaps, not only the marginal distribution of one gap.
- The script states a conditional theorem with stronger assumptions than it names.
- Monte Carlo simulation of exponential variables verifies a probability identity, but it does not verify anything about primes.

Replacement: `docs/research_note.md` states the missing joint/exchangeability assumption explicitly.

### `18_approche_directe.py`

Problems:

- Mostly expository output, not a research script.
- The Monte Carlo model `P(n prime) = 1/log(n)` ignores residue constraints and known prime correlations.
- It ends by claiming a conditional proof exists, but the previous scripts did not establish the required assumptions.

Replacement: removed.

### `19_attaque_ia.py`

Problems:

- Pattern mining is done without train/test separation, null models, or multiple-testing control.
- The k-tuple section uses hardcoded constants instead of the singular series computation.
- Several detected "patterns" are basic consequences of modular arithmetic or the prime number theorem.
- The branding around "AI" makes the script look less rigorous than the actual computations.

Replacement: removed; validated finite diagnostics live in `scripts/gap_diagnostics.py` and `scripts/mod6_gap_analysis.py`.

### `20_mod6_structure.py`

Useful part:

- residue counts and transition matrices modulo 6.

Problems:

- The script claims a path toward proof from modulo-6 symmetry.
- New diagnostics directly show that the residue classes are not individually balanced at finite scale. On the first `100000` primes:
  - current gap `0 mod 6`: raw imbalance `12.48689%`;
  - current gap `2 mod 6`: raw imbalance `20.11092%`;
  - current gap `4 mod 6`: raw imbalance `1.61913%`.
- The global strict balance is therefore a cancellation across classes and gap sizes, not a simple "perfect symmetry in each class".

Replacement: `scripts/mod6_gap_analysis.py`.

### `21_revolution.py`

Problems:

- Conceptual brainstorming is mixed with code output.
- Topological, physical, algebraic, and proof-assistant analogies are not tied to a concrete theorem or a reproducible test.
- The script gives a strong impression of originality without a verified mathematical mechanism.

Replacement: removed.

### `22_gpu_verification.py`, `23_extreme_verification.py`, `24_ultra_optimized.py`

Problems:

- CUDA is queried at import time, so the scripts are not portable.
- All three have fixed large targets and no `argparse`.
- The `23` version tries to store a billion primes or huge intermediate lists in memory.
- The `24` version is closer to a real large-run prototype, but it still concatenates large arrays and has top-level GPU execution.
- None of them records enough metadata for a verifiable large computation: machine, commit, Python/PyTorch versions, elapsed phases, raw counts, and reproducibility log.

Replacement: `scripts/streaming_large_runner.py`, a segmented streaming counter with optional JSONL checkpoints and no full-prime-list storage. `scripts/large_numpy_verification.py` remains only as a small portable cross-check.

### `25_riemann_oscillations.py`

Problems:

- The signal is cumulative and non-stationary, so a direct periodogram is hard to interpret.
- Resampling in `log(N)` is arbitrary and lacks sensitivity checks.
- There is no null model or multiple-testing correction.
- Frequencies found by periodogram are compared visually to Riemann-zero ordinates, which is not evidence of a real connection.

Replacement: removed.

## Policy for active scripts

An active script should satisfy all of these:

- has `argparse` and explicit parameters;
- has no expensive work at import time;
- runs on a normal CPU unless marked optional;
- writes no files unless given an output path;
- separates strict `>`, strict `<`, and equality;
- prints measurements, not proof claims.

Anything that fails these rules should live outside the active repository until rewritten.

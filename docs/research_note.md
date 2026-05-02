# Research note

## Question

Let $p_n$ be the $n$-th prime and define the prime gap

$$
d_n = p_{n+1} - p_n.
$$

The computational question studied here is:

How often is $d_{n+1} > d_n$ compared with $d_{n+1} < d_n$?

Equality events are treated separately.

## Heuristic model

A common heuristic in prime-gap questions is that normalized gaps

$$
g_n = \frac{d_n}{\log p_n}
$$

behave, at large scale, like samples from an exponential distribution. In the strongest simplified version of the model, two neighboring normalized gaps are treated as exchangeable continuous random variables `X` and `Y`.

Exchangeability alone gives

$$
\Pr(Y > X) = \Pr(Y < X).
$$

Continuity gives $\Pr(Y = X) = 0$ in the model, so each strict probability is $1/2$.

This explains why one might expect the strict increase and strict decrease counts to be close in large computations.

## What this does not prove

This model is not a proof about consecutive prime gaps.

The missing step is the hard one: proving enough joint distribution and dependence control for consecutive normalized prime gaps. This is related to deep prime-tuple heuristics and is not established here.

In particular, this repository does not prove Hardy-Littlewood, does not derive Gallagher's theorem, and does not turn the random model into an unconditional theorem.

## What is worth testing

The scripts are useful for:

- checking the strict imbalance between increase and decrease counts;
- tracking equality events at finite scale;
- comparing behavior across residue classes modulo 6;
- testing whether convergence appears stable under larger ranges.

The output should be read as computational evidence and model-checking, not as a proof.

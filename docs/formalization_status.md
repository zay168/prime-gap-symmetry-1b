# Formalization status

Earlier versions of this repository contained Lean files that used placeholders, axioms, or string-valued "theorems" to represent the argument. That was misleading for a public repository, so the Lean skeleton has been removed from the main presentation.

Current status:

- there is no completed Lean proof in this repository;
- the probabilistic heuristic is described informally in [research_note.md](research_note.md);
- a real formalization would need precise definitions of natural density, prime gaps, the probabilistic model, and the conditional number-theoretic assumptions;
- any future Lean work should live in a separate `lean/` directory and compile without pretending that placeholders are proofs.

This is a deliberate credibility choice: a smaller honest repository is stronger than a larger one that overstates its formal content.

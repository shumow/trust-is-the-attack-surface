# Revision plan

This branch tracks review-driven revisions that make the repository easier to
reproduce and harder to over-read.

1. **Reproducibility friction.** Ensure demos create `results/` on demand, add a
   single reproduction runner, and document expected commands and runtimes.
2. **Tests around the mathematical core.** Add unit tests for quine combinatorics,
   closed-form formulas, result-path behavior, and optional slow regression tests
   for the headline correlations.
3. **Claim boundaries.** Label claims as proved on the toy, measured on the toy,
   measured on sample prose through the toy, or conjectured for transformers.
4. **Real-text cache experiment.** Commit a small natural-prose fixture and a demo
   that reports cache benefit and reliance under char- and word-level tokenization.
5. **Literature review.** Done in `LITERATURE_REVIEW.md`: add a short related-work
   section covering cache models, induction heads, indirect prompt injection,
   context poisoning, reinforced urns, and the cryptanalytic analogy. This should
   separate cited background from this repo's own contribution.
6. **Sensitivity sweeps.** Run multiple seeds and vary vocabulary size, document
   length, Dirichlet strength, source concentration, and cache order.
   *Done.* `demo_sensitivity.py` varies `V`, `G_CONC`, `D_CONC`, length, and
   Dirichlet strength one at a time over 3 seeds; the trust-saturation dichotomy
   holds in all 11 configs (`results/sensitivity.png`). `demo_joint_sweep.py` varies
   `V` × cache order jointly (orders 1–4), and reliance-when-useless collapses onto
   the single density axis `V^k` (`results/joint_sweep.png`). Remaining: measurement
   off the synthetic source (the transformer-validation sibling, item 9).
7. **Dual-use framing for contagion.** Keep the mechanism study, but pair attack
   cost measurements with detection and mitigation measurements.
8. **Publication cleanup.** Align README, Markdown articles, LaTeX articles, and
   generated figures so they make the same scoped claims.
9. **Transformer validation sibling.** Continue the separate real-transformer
   validation effort; keep this repo explicit that those results are external until
   they are measured and cited here.

# Trust Is the Attack Surface

A cryptanalytic reading of context poisoning, worked on a toy global+cache Markov
model. The thesis: in-context **exploitability tracks trust** (the weight a model
places on its context), **not usefulness** — and the two only coincide when contexts
are sparse enough that trust must be earned. See
[trust_is_the_attack_surface.md](trust_is_the_attack_surface.md) for the article and
[docs_06_conservation_law.md](docs_06_conservation_law.md) for the worked write-up
with measurements.

A second line of work, **context contagion**, extends this from a single locked token
to whole self-reproducing token strings (quines): which strings reproduce, how cheaply
one can be planted in an empty or already-warmed cache, and how it spreads between
caches as a context-window worm — culminating in a re-run under a longest-match
(induction-head) surrogate, where contagion is cheapest and most transmissible. See the
standalone article [latex/contagious_context.tex](latex/contagious_context.tex), the
running results log [contagion_notes.md](contagion_notes.md), and
[transformer_validation_plan.md](transformer_validation_plan.md) for the plan to test it
on real transformers.

This repository is **independent** of, and depends on, the toy model library it
analyzes.

## Relationship to the substrate library

The Markov global+cache model lives in a separate library,
[`markov-cache`](https://github.com/shumow/markov-models-aux-context-caching), and is
consumed here as an external, versioned dependency. This repo is the **poisoning /
conservation-law analysis layer** built on top of it; the two are maintained
independently. See
[sister_library_packaging_recommendations.md](sister_library_packaging_recommendations.md)
for the packaging rationale.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the substrate library to a published tag, so a fresh clone
is self-contained:

```
markov-cache[demos] @ git+https://github.com/shumow/markov-models-aux-context-caching@v0.1.0
```

To hack on the substrate alongside this repo, install it editable from a local
sibling checkout instead (see the commented line in
[requirements.txt](requirements.txt)):

```bash
pip install -e "../Markov Models with Aux Context Caching[demos]"
```

## Reproduce

```bash
python demo_conservation.py
```

Writes `results/conservation_law.png` and `results/condensation.png` and prints the
per-cache-order correlation table. Expected: order-3 (sparse / earned trust)
`corr(benefit, propagated) ≈ 0.97`; order-1 (dense / saturated trust) `≈ 0.5`;
condensation knee at cache reliance ≈ 0.9.

The context-contagion results are reproduced by the `demo_contagion*.py` scripts, each
writing its figure(s) to `results/` (for example `demo_contagion.py` for the
reproduction law, `demo_contagion_worm.py` for the worm, `demo_contagion_ppm.py` for the
induction surrogate); see [contagion_notes.md](contagion_notes.md) for what each shows.

## Papers

Two formal write-ups live in [`latex/`](latex/). Build either with a standard TeX
distribution (two passes resolve cross-references; the bibliography is inline, so no
`bibtex` step):

```bash
cd latex
pdflatex trust_is_the_attack_surface.tex && pdflatex trust_is_the_attack_surface.tex
pdflatex contagious_context.tex          && pdflatex contagious_context.tex
```

`contagious_context.tex` is the context-contagion article and references the first.

## Layout

| File | What |
|---|---|
| `trust_is_the_attack_surface.md` | The trust/usefulness article (essay form). |
| `latex/trust_is_the_attack_surface.tex` | The trust/usefulness results as a math/CS article (built PDF alongside). |
| `docs_06_conservation_law.md` | Worked write-up + the cryptanalysis dictionary. |
| `demo_conservation.py` | Reproduces the trust/usefulness figures and correlation table. |
| `latex/contagious_context.tex` | The **context-contagion** article (built PDF alongside). |
| `contagion.py` | Library for the contagion line (quine constructors, reproduction, attacker cost). |
| `demo_contagion*.py` | The contagion experiments (payload, delivery, worm, induction surrogate). |
| `contagion_notes.md` | Running results log for the contagion line. |
| `transformer_validation_plan.md` | Plan for testing context contagion on real transformers. |
| `llm_detection_research_proposal.md` | Related research proposal. |
| `sister_library_packaging_recommendations.md` | Why/how the substrate is a library. |
| `DEV_PLAN.md` | Packaging + wiring dev plan and its execution log. |

# Trust Is the Attack Surface

A cryptanalytic reading of context poisoning, worked on a toy global+cache Markov
model. The thesis: in-context **exploitability tracks trust** (the weight a model
places on its context), **not usefulness** — and the two only coincide when contexts
are sparse enough that trust must be earned. See
[trust_is_the_attack_surface.md](trust_is_the_attack_surface.md) for the article and
[docs_06_conservation_law.md](docs_06_conservation_law.md) for the worked write-up
with measurements.

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

## Layout

| File | What |
|---|---|
| `trust_is_the_attack_surface.md` | The article. |
| `docs_06_conservation_law.md` | Worked write-up + the cryptanalysis dictionary. |
| `llm_detection_research_proposal.md` | Related research proposal. |
| `demo_conservation.py` | Reproduces the figures and the correlation table. |
| `sister_library_packaging_recommendations.md` | Why/how the substrate is a library. |
| `DEV_PLAN.md` | Packaging + wiring dev plan and its execution log. |

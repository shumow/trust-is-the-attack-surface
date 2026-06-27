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

## Evidence status

This repo now uses four labels for its claims:

| Label | Meaning |
|---|---|
| **Proved on toy** | A combinatorial or closed-form statement about the global+cache Markov substrate. |
| **Measured on toy** | A deterministic demo result reproduced by scripts in this repo. |
| **Measured on sample prose through toy** | A sanity check using the same toy substrate on `data/sample_prose.txt`; useful for calibration, not a claim about transformers. |
| **Conjectured for transformers** | A hypothesis for real LLMs, not established here. See `transformer_validation_plan.md`. |

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
python run_reproductions.py
```

The default runner executes the central demos and creates `results/` if needed:
`demo_conservation.py`, `demo_contagion.py`, `demo_contagion_ppm.py`, and
`demo_real_text_cache.py`. For the full figure suite, run:

```bash
python run_reproductions.py --full
```

The main conservation demo writes `results/conservation_law.png` and
`results/condensation.png` and prints the per-cache-order correlation table.
Expected: order-3 (sparse / earned trust) `corr(benefit, propagated) ≈ 0.97`;
order-1 (dense / saturated trust) `≈ 0.5`; condensation knee at cache reliance
≈ 0.9.

The context-contagion results are reproduced by the `demo_contagion*.py` scripts, each
writing its figure(s) to `results/` (for example `demo_contagion.py` for the
reproduction law, `demo_contagion_worm.py` for the worm, `demo_contagion_ppm.py` for the
induction surrogate); see [contagion_notes.md](contagion_notes.md) for what each shows.

`demo_real_text_cache.py` is a small calibration run on a committed prose fixture. It
does **not** measure transformers, and it is a single fixture under one
configuration (`data/sample_prose.txt`, the `TOKEN_LEVELS` doc lengths, a 0.65
split) — read the *direction*, not the digits, and re-run it on your own text
before leaning on any number. The qualitative pattern it shows: a dense char-level
order-1 cache **saturates** (`final_reliance` ~0.85–0.9) while *hurting* prediction
(`benefit` strongly negative, ~ −0.8 bits), whereas a sparse word-level order-3
cache stays **near-untrusted** (`final_reliance` of order 0.01–0.02) and roughly
neutral on benefit. That is the same earned-vs-saturated split as the synthetic
sweep, which is the only claim this fixture is meant to support.

`demo_sensitivity.py` checks that the headline is not an artifact of the baseline
source configuration. It varies `V`, `G_CONC`, `D_CONC`, document length, and
Dirichlet strength one at a time (3 seeds, 11 configs) and reports the trust-
saturation dichotomy for each, writing `results/sensitivity.png` and `.csv`. Result:
the reliance dichotomy holds in all 11 (dense order-1 trusted at 0.77–0.94 with no
structure to learn; sparse order-3 at 0.00–0.11), and order-1's `corr(benefit,
propagated)` is unstable around zero while order-3's stays high — the decoupling is
robust. Runs in ~3 min; included in `run_reproductions.py --full`.

`demo_joint_sweep.py` closes the one-at-a-time caveat by varying vocabulary `V` and
cache order `k` *jointly* (`{8,16,32,64}` × `{1,2,3,4}`, 3 seeds). Since the number
of possible order-`k` contexts is `V^k`, it tests whether `V` and `k` matter only
through that product. They do: reliance-when-useless falls as a clean sigmoid in
`log V^k` and collapses across `(V,k)` pairs of equal `V^k` (the three `V^k = 4096`
configs agree to within 0.08), so the qualitative "sparse vs dense" contrast becomes
a one-parameter density law. Writes `results/joint_sweep.png` and `.csv`; ~3 min,
included in `--full`.

## Test

```bash
python -m unittest discover                                  # fast: ~2s
RUN_SLOW_TESTS=1 python -m unittest tests.test_slow_regression  # full single-seed sweep
```

The default `discover` suite covers the *proved-on-toy* math (de Bruijn
minimality, the closed-form reproduction / min-repetition formulas) **and** a fast,
multi-seed guard on the headline mechanism — that order-1 trust saturates
independent of `pi_ctx` while order-3 trust is earned from recurrence
(`tests/test_core.py::TrustSaturationDichotomyTests`). It asserts the qualitative
dichotomy, not seed-specific magnitudes. The opt-in slow test pins the exact
single-seed correlation numbers (`corr(benefit, propagated) ≈ 0.97` / `≈ 0.52`)
from the full `demo_conservation` sweep; treat those magnitudes as `SEED=7`-specific.

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
| `LITERATURE_REVIEW.md` | Related-work summary and citation-fidelity audit. |
| `demo_conservation.py` | Reproduces the trust/usefulness figures and correlation table. |
| `demo_real_text_cache.py` | Sanity-checks toy-cache reliance on committed sample prose. |
| `demo_sensitivity.py` | Source-parameter robustness sweep for the trust-saturation dichotomy. |
| `demo_joint_sweep.py` | Joint `V` × cache-order sweep; reliance collapses onto the `V^k` density axis. |
| `latex/contagious_context.tex` | The **context-contagion** article (built PDF alongside). |
| `contagion.py` | Library for the contagion line (quine constructors, reproduction, attacker cost). |
| `demo_contagion*.py` | The contagion experiments (payload, delivery, worm, induction surrogate). |
| `contagion_notes.md` | Running results log for the contagion line. |
| `transformer_validation_plan.md` | Plan for testing context contagion on real transformers. |
| `llm_detection_research_proposal.md` | Related research proposal. |
| `sister_library_packaging_recommendations.md` | Why/how the substrate is a library. |
| `DEV_PLAN.md` | Packaging + wiring dev plan and its execution log. |
| `REVISION_PLAN.md` | Review-driven cleanup and validation plan. |

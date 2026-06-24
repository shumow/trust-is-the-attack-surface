# Packaging recommendations for `markov-models-aux-context-caching`

How to make the sister repo
([shumow/markov-models-aux-context-caching](https://github.com/shumow/markov-models-aux-context-caching))
easily consumable by this project (`TrustIsTheAttackSurface`) and other downstream
work.

> **Scope decision (settled):** `TrustIsTheAttackSurface` is a **permanently
> independent downstream consumer** — it will *not* be merged back into the sister
> repo as a "Stage 4." Everything below commits to that: a stable, versioned library
> boundary consumed via a pinned dependency. Submodule/in-tree options are recorded
> only for completeness and are not recommended.

---

## TL;DR

Package it as a **minimal, single-module Python library**. The core
(`markov_cache.py`) is already library-shaped — pure numpy, no module-level side
effects, no `__main__` block. Add a `pyproject.toml`, keep the demos as scripts,
demote demo/GUI dependencies to optional extras, and consume it here as a
**pinned git dependency** rather than a copy.

---

## Why this is the right call

### What is library code vs. application code

| File | Nature | Ship in package? |
|---|---|---|
| `markov_cache.py` | Pure model: `MarkovCounts`, `BackoffModel`, `CachePredictor`, `PpmPredictor`, plus the functions this project imports (`peaky_transition_matrix`, `make_corpus`, `sample_document`, `global_dist`, `combine_dirichlet`, `make_predictor`, `evaluate`, `evaluate_predictor`, `generate`, …). numpy-only, no side effects at import. | **Yes** |
| `text_adapter.py` | Real-text tokenizer/adapter. Library-shaped. This project's next step (`docs_06` Honesty Flags) explicitly wants it. | **Yes** |
| `demo_*.py` | Application scripts: write figures to relative `results/` paths, assume cwd = repo root. | No — keep as scripts |
| `gui_app.py` | Streamlit app. | No — keep as script |
| `verify_reproducibility.py` | Harness. | No — keep as script |

### Current dependency situation

- This project's [demo_conservation.py:65](demo_conservation.py:65) does
  `from markov_cache import (...)` — **all 11 imported symbols exist** in the sister
  repo's `markov_cache.py`, but there is **no local copy**, so the script cannot run
  standalone today. Packaging fixes this cleanly.
- The sister repo has **no `pyproject.toml`/`setup.py`, no `__all__`, no package
  boundary**. Dependencies are listed loosely in `requirements.txt`
  (numpy, matplotlib, streamlit).

---

## Concrete changes to the sister repo

### 1. Add a `pyproject.toml`

Expose `markov_cache` (and `text_adapter`) as installable top-level modules, with
**numpy as the only runtime dependency**. Demote demo/GUI deps to extras.

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "markov-cache"
version = "0.1.0"
description = "Toy global+cache Markov model: parametric memory vs. context window."
requires-python = ">=3.9"
dependencies = ["numpy"]

[project.optional-dependencies]
demos = ["matplotlib"]
gui   = ["streamlit", "matplotlib"]

[tool.setuptools]
py-modules = ["markov_cache", "text_adapter"]
```

This keeps a consumer that only wants the model from dragging in Streamlit.

### 2. Add an explicit `__all__` to `markov_cache.py`

Documents the public surface this project depends on and makes the library boundary
intentional rather than incidental. At minimum include the symbols
`demo_conservation.py` imports.

### 3. Tag a release

Cut a git tag (e.g. `v0.1.0`) so downstream analysis pins a fixed version of the
substrate and stays reproducible.

---

## How this project consumes it

Declare a **pinned git dependency** (no PyPI publish required):

```
pip install "markov-cache @ git+https://github.com/shumow/markov-models-aux-context-caching@v0.1.0"
```

Then `from markov_cache import ...` works from anywhere, and the experimental layer
(`demo_conservation.py`) lives here while the toy model lives there — matching the
real conceptual split (substrate vs. independent poisoning analysis).

Pin a **specific tag**, not a branch or `HEAD`, so a future change upstream can never
silently alter the substrate underneath this project's published results. Record the
pinned version in this repo (e.g. a `requirements.txt` / `pyproject.toml` here) and
treat upgrading it as a deliberate act: bump the pin, re-run `demo_conservation.py`,
confirm the figures/correlations still hold, then commit the new pin.

---

## Alternatives considered

| Approach | Verdict |
|---|---|
| **Pinned `git+https` install** (above) | **Recommended.** Nothing to publish, reproducible, clean imports, and the right fit for an indefinitely independent consumer. |
| Publish to PyPI | Optional upgrade. Only worth it if the sister repo gains consumers beyond this project; for a single downstream, the pinned git tag is enough. |
| Git submodule | Not recommended. Made sense only under an eventual-merge plan, which is now ruled out. Couples the two repos' working trees and complicates imports for no benefit here. |
| Vendoring / copying `markov_cache.py` here | **Avoid.** Forks the substrate; the two copies will drift — the worst outcome for a long-lived independent project. |

---

## Living with an indefinitely independent dependency

Because the two repos will never reconverge, treat the sister library as a stable
external dependency:

- **Pin a tag and own the upgrade cadence.** This project decides when to adopt a
  new substrate version; upstream changes never arrive implicitly.
- **Keep the experimental surface narrow.** This project depends only on the symbols
  in `markov_cache.__all__` (and `text_adapter`). If it starts reaching into
  internals, fold the needed helper into the library and re-export it rather than
  importing private names — that keeps the boundary maintainable across versions.
- **No upstream changes required for this project's sake beyond §1–§3.** Once the
  library boundary exists and is tagged, this repo is self-sufficient.

---

## Don't over-package

It is one ~23 KB module. A single-module distribution (or a tiny `markov_cache/`
package with `__init__.py` re-exporting) is sufficient — resist a deep package tree.

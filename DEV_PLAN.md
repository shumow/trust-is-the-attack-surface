# Dev Plan: package the sister library and wire it into this project

Executes the recommendations in
[sister_library_packaging_recommendations.md](sister_library_packaging_recommendations.md),
for the **permanently independent** downstream case.

## Goal / definition of done

`demo_conservation.py` runs from this project against `markov_cache` installed as a
real, importable library (not a copied file), producing
`results/conservation_law.png` and `results/condensation.png`, with the printed
correlation table matching the documented claims (order-3 `corr(benefit,propagated) ≈ 0.97`,
order-1 `≈ 0.5`). The sister repo carries a clean, tagged, pip-installable library
boundary.

## Facts established during recon

- Sister repo is already cloned locally (clean tree, on `origin/HEAD`) at
  `/Users/dbone/Claude/Projects/Markov Models with Aux Context Caching`.
- `markov_cache.py`: no `pyproject.toml`/`setup.py`, no `__all__`, no `__main__`
  block. Pure numpy. Public surface = 13 functions + 4 classes (full list in §A2).
- This project's `demo_conservation.py` imports 11 of those symbols.
- Toolchain: Python 3.9.6, pip 21.2.4, git 2.50.1, `venv` available.

---

## Workstream A — sister repo packaging (local clone)

- **A1** Confirm clone state. *(done in recon)*
- **A2** Add `__all__` to `markov_cache.py` covering the public API:
  `peaky_transition_matrix, sample_document, make_corpus, global_dist,
  combine_linear, combine_dirichlet, evaluate, reliance_curve, make_predictor,
  generate, suggest, evaluate_predictor, tune_ppm, MarkovCounts, BackoffModel,
  CachePredictor, PpmPredictor`.
- **A3** Add `pyproject.toml`: name `markov-cache`, runtime dep `numpy`, extras
  `demos=[matplotlib]` / `gui=[streamlit,matplotlib]`, `py-modules =
  ["markov_cache", "text_adapter"]`.
- **A4** Build/install sanity check into this project's venv.

## Workstream B — this project wiring

- **B1** Create `.venv` in this project.
- **B2** Install the sister lib **editable from the local clone** for development
  (`pip install -e <clone>`), plus the `demos` extra (matplotlib) needed by
  `demo_conservation.py`.
- **B3** Record the dependency in this repo:
  - `requirements.txt` with the editable local path for dev **and** a commented
    canonical pinned line `markov-cache @ git+https://…@v0.1.0` for the published
    consume path.
- **B4** Create `results/` directory.

## Workstream C — verify end-to-end

- **C1** Run `python demo_conservation.py`.
- **C2** Confirm both PNGs are written and the correlation table matches the
  documented regimes. Capture the printed numbers.

## Workstream D — handoff (user actions, outward)

Not performed automatically (pushing/tagging are outward + the user's call):
1. Review the two new files in the sister clone (`__all__`, `pyproject.toml`).
2. `git commit` + `git tag v0.1.0` + `git push --follow-tags` in the sister repo.
3. Switch this project's `requirements.txt` from the editable local path to the
   pinned `git+https://…@v0.1.0` line.

---

## Progress log

- [x] A1 recon
- [x] A2 `__all__` added to `markov_cache.py` (17 symbols)
- [x] A3 `pyproject.toml` added to sister clone
- [x] A4 build sanity — editable wheel built, `import markov_cache` + `text_adapter` OK
- [x] B1 `.venv` created (pip auto-upgraded 21.2.4 → 26.0.1)
- [x] B2 `pip install -e "../Markov Models with Aux Context Caching[demos]"`
- [x] B3 `requirements.txt` written (local-dev pin active; canonical git+tag pin documented)
- [x] B4 `results/` created
- [x] C1 ran `demo_conservation.py`
- [x] C2 verified — figures written; order-3 corr(benefit,propagated)=**+0.968**,
      order-1=**+0.520**, condensation knee at reliance≈0.9. Matches docs.

### Results captured (this run)
- `results/conservation_law.png`, `results/condensation.png` written.
- order 1: corr(benefit,propagated)=+0.520, corr(benefit,reliance)=+0.564, corr(benefit,static)=−0.990
- order 3: corr(benefit,propagated)=+0.968, corr(benefit,reliance)=+0.933, corr(benefit,static)=−0.936

### Remaining: Workstream D (user / outward actions — not done automatically)
In the sister clone `../Markov Models with Aux Context Caching` (2 uncommitted files:
`pyproject.toml`, `__all__` edit in `markov_cache.py`):
1. Review the diff.
2. `git commit -am "Package as installable library (pyproject + __all__)"`
3. `git tag v0.1.0 && git push --follow-tags`
4. In this repo, switch `requirements.txt` from the editable local line to the
   canonical `markov-cache[demos] @ git+https://…@v0.1.0` pin.

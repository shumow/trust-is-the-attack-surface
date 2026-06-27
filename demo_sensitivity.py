"""Sensitivity sweep: is the trust-saturation dichotomy robust to the source's
free parameters? (REVISION_PLAN item 6, first cut.)

`demo_conservation.py` establishes the headline at *one* configuration and a single
seed: a dense order-1 cache saturates trust independent of usefulness
(`corr(benefit, propagated) ~ 0.5`), while a sparse order-3 cache earns it
(`~ 0.97`). The honest open question -- flagged in `LITERATURE_REVIEW.md` under
"Omissions and competing explanations" -- is whether that split is a property of the
mechanism or an artifact of the chosen `V`, `G_CONC`, `D_CONC`, document length, and
Dirichlet strength.

This script varies each of those one at a time around the conservation baseline,
averages over a few seeds, and reports for each configuration:

  rel1@0   order-1 cache reliance at pi_ctx = 0  (dense; should be HIGH -- free trust)
  rel3@0   order-3 cache reliance at pi_ctx = 0  (sparse; should be LOW -- earned)
  cprop1   corr(benefit, propagated leverage) across the pi grid, order 1  (should be LOW)
  cprop3   corr(benefit, propagated leverage) across the pi grid, order 3  (should be HIGH)

and a qualitative PASS/FAIL on the dichotomy (dense trusted while useless; sparse not;
decoupling stronger for the dense cache). The point is not to defend specific
magnitudes -- it is to find the regions of source-parameter space where the
mechanism holds and, more usefully, where it breaks.

Run:  python3 demo_sensitivity.py            # full OAT sweep, ~1-2 min
      python3 demo_sensitivity.py --quick    # 1 seed, coarse pi grid, fast smoke
Writes results/sensitivity.csv and results/sensitivity.png and prints a table.
"""
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo_utils import result_path
from markov_cache import (
    peaky_transition_matrix, make_corpus, BackoffModel,
    evaluate, evaluate_predictor, make_predictor,
)
from demo_conservation import propagated_leverage

# --------------------------------------------------------------------------- #
# Baseline = the conservation-demo source, slightly trimmed for sweep speed.
# --------------------------------------------------------------------------- #
BASELINE = dict(V=32, g_conc=1.0, d_conc=0.05, length=300, dir_a=1.0)
GLOBAL_ORDER = 2
G_ALPHA = 0.1
N_TRAIN, N_TEST = 150, 16
ORDERS = (1, 3)

# One-at-a-time variations around the baseline (baseline value included in each).
AXES = {
    "V":      [16, 32, 64],
    "g_conc": [0.5, 1.0, 2.0],
    "d_conc": [0.02, 0.05, 0.10],
    "length": [150, 300, 600],
    "dir_a":  [0.5, 1.0, 2.0],
}


def build_configs():
    """One-at-a-time configs around BASELINE, de-duplicated, baseline first."""
    seen = {tuple(sorted(BASELINE.items()))}
    configs = [("baseline", dict(BASELINE))]
    for axis, values in AXES.items():
        for v in values:
            if v == BASELINE[axis]:
                continue
            cfg = dict(BASELINE)
            cfg[axis] = v
            key = tuple(sorted(cfg.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append((f"{axis}={v}", cfg))
    return configs


def measure_config(cfg, pi_grid, seeds):
    """Return per-order dicts of pi-grid curves (benefit, reliance, propagated),
    averaged over seeds."""
    V, g_conc, d_conc, length, dir_a = (
        cfg["V"], cfg["g_conc"], cfg["d_conc"], cfg["length"], cfg["dir_a"]
    )
    # accumulate sums over seeds: curves[order][quantity] -> array over pi_grid
    acc = {o: {q: np.zeros(len(pi_grid)) for q in ("benefit", "reliance", "prop")}
           for o in ORDERS}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        G = peaky_transition_matrix(V, g_conc, rng)
        for j, pi in enumerate(pi_grid):
            train = make_corpus(G, pi, N_TRAIN, length, d_conc, rng)
            test = make_corpus(G, pi, N_TEST, length, d_conc, rng)
            gm = BackoffModel(GLOBAL_ORDER, V).fit(train)
            for o in ORDERS:
                bg, _ = evaluate(test, gm, cache_order=o, method="global",
                                 weight=0.0, g_alpha=G_ALPHA)
                bc, _ = evaluate(test, gm, cache_order=o, method="dirichlet",
                                 weight=dir_a, g_alpha=G_ALPHA)
                _, _, rel = evaluate_predictor(
                    test, make_predictor(gm, approach="dirichlet", cache_order=o,
                                         weight=dir_a, g_alpha=G_ALPHA),
                    reliance=True)
                prop = propagated_leverage(test, gm, cache_order=o, a=dir_a,
                                           g_alpha=G_ALPHA)
                acc[o]["benefit"][j] += (bg - bc)
                acc[o]["reliance"][j] += float(rel[-1])
                acc[o]["prop"][j] += prop
    n = len(seeds)
    return {o: {q: acc[o][q] / n for q in acc[o]} for o in ORDERS}


def _corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize(curves):
    """Reduce per-order pi curves to the four headline scalars + a verdict."""
    o1, o3 = curves[1], curves[3]
    rel1_0, rel3_0 = float(o1["reliance"][0]), float(o3["reliance"][0])
    cprop1 = _corr(o1["benefit"], o1["prop"])
    cprop3 = _corr(o3["benefit"], o3["prop"])
    # Qualitative dichotomy: dense cache trusted while useless (pi=0), sparse cache
    # not, and the benefit<->exploitability coupling weaker for the dense cache.
    dense_trusted_when_useless = rel1_0 > 0.60
    sparse_earns_trust = rel3_0 < 0.30
    decoupling_dir = (np.isnan(cprop1) or np.isnan(cprop3) or cprop3 > cprop1 + 0.05)
    verdict = dense_trusted_when_useless and sparse_earns_trust and decoupling_dir
    return dict(rel1_0=rel1_0, rel3_0=rel3_0, cprop1=cprop1, cprop3=cprop3,
                verdict=verdict)


def run(pi_grid, seeds):
    rows = []
    for label, cfg in build_configs():
        curves = measure_config(cfg, pi_grid, seeds)
        s = summarize(curves)
        rows.append(dict(label=label, **cfg, **s))
        cp1 = "  nan" if np.isnan(s["cprop1"]) else f"{s['cprop1']:+.2f}"
        cp3 = "  nan" if np.isnan(s["cprop3"]) else f"{s['cprop3']:+.2f}"
        print(f"{label:>12}  rel1@0={s['rel1_0']:.2f}  rel3@0={s['rel3_0']:.2f}  "
              f"corr(b,prop) o1={cp1} o3={cp3}  "
              f"{'PASS' if s['verdict'] else 'FAIL'}")
    return rows


def write_csv(rows):
    path = result_path("sensitivity.csv")
    fields = ["label", "V", "g_conc", "d_conc", "length", "dir_a",
              "rel1_0", "rel3_0", "cprop1", "cprop3", "verdict"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"wrote {path}")


def plot(rows):
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.7 * len(rows)), 7.5), sharex=True)

    # Panel 1: reliance at pi=0 -- the cheap, robust saturation signal.
    axes[0].bar(x - 0.2, [r["rel1_0"] for r in rows], 0.38,
                label="order-1 reliance @ pi=0 (dense)")
    axes[0].bar(x + 0.2, [r["rel3_0"] for r in rows], 0.38,
                label="order-3 reliance @ pi=0 (sparse)")
    axes[0].axhline(0.60, color="green", lw=0.8, ls="--", alpha=0.7)
    axes[0].axhline(0.30, color="red", lw=0.8, ls="--", alpha=0.7)
    axes[0].set_ylabel("cache reliance at pi_ctx = 0")
    axes[0].set_title("Trust when context is useless: dense saturates, sparse does not")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")

    # Panel 2: corr(benefit, propagated) -- the decoupling, per order.
    c1 = [0.0 if np.isnan(r["cprop1"]) else r["cprop1"] for r in rows]
    c3 = [0.0 if np.isnan(r["cprop3"]) else r["cprop3"] for r in rows]
    axes[1].bar(x - 0.2, c1, 0.38, label="order-1 corr(benefit, propagated)")
    axes[1].bar(x + 0.2, c3, 0.38, label="order-3 corr(benefit, propagated)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("Pearson r over pi grid"); axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_title("Usefulness<->exploitability coupling: weaker for the dense cache")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    path = result_path("sensitivity.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="1 seed and a coarse 3-point pi grid (fast smoke run)")
    args = ap.parse_args()
    if args.quick:
        pi_grid, seeds = np.array([0.0, 0.45, 0.9]), (1,)
    else:
        pi_grid, seeds = np.linspace(0.0, 0.9, 5), (1, 2, 3)

    print(f"=== Sensitivity sweep (seeds={seeds}, pi_grid={np.round(pi_grid,3)}) ===")
    print("baseline:", BASELINE)
    rows = run(pi_grid, seeds)
    n_pass = sum(r["verdict"] for r in rows)
    print(f"\ndichotomy holds for {n_pass}/{len(rows)} configurations")
    write_csv(rows)
    plot(rows)


if __name__ == "__main__":
    main()

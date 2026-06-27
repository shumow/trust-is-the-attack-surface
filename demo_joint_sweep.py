"""Joint V x cache-order sweep: is the trust-saturation dichotomy governed by a
single density axis? (REVISION_PLAN item 6, closing the one-at-a-time caveat.)

`demo_sensitivity.py` varies the source parameters one at a time and finds the
order-1-vs-order-3 dichotomy robust. This script varies the *two* knobs that the
mechanism says jointly set context density -- vocabulary `V` and cache order `k` --
together, because the number of distinct order-`k` contexts is `V**k`. The repo's
claim is that trust saturates when contexts are dense (few possible contexts, each
filled regardless of usefulness) and is earned when they are sparse. If that is the
real variable, then `V` and `k` should matter *only through their product* `V**k`:
reliance-when-useless and the usefulness<->exploitability coupling should collapse
onto a single curve in `V**k`, not depend on `V` and `k` separately.

What it measures, per (V, k), averaged over seeds and swept over a pi_ctx grid:

  rel0      cache reliance at pi_ctx = 0  (trust handed out when context is useless)
  ben0      benefit (bits saved) at pi_ctx = 0  (negative => the cache is hurting)
  cprop     corr(benefit, propagated leverage) across the pi grid (the coupling)

Headline read: rel0 should fall monotonically from ~1 (dense, V**k small) to ~0
(sparse, V**k large) and collapse across (V, k) pairs with equal V**k; cprop should
be unstable/low where rel0 is high (decoupled) and high where rel0 is low (earned).

Run:  python3 demo_joint_sweep.py            # full grid, a few minutes
      python3 demo_joint_sweep.py --quick    # small grid, 1 seed, fast smoke
Writes results/joint_sweep.csv and results/joint_sweep.png and prints a table.
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

# Fixed source (the conservation baseline minus the two knobs we are sweeping).
G_CONC, D_CONC = 1.0, 0.05
GLOBAL_ORDER, G_ALPHA, DIR_A = 2, 0.1, 1.0
N_TRAIN, N_TEST, LENGTH = 150, 16, 300

V_GRID = (8, 16, 32, 64)
K_GRID = (1, 2, 3, 4)


def _corr(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def measure_cell(V, k, pi_grid, seeds):
    """Average over seeds: per-pi benefit/reliance/propagated curves for cache order k."""
    benefit = np.zeros(len(pi_grid))
    reliance = np.zeros(len(pi_grid))
    prop = np.zeros(len(pi_grid))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        G = peaky_transition_matrix(V, G_CONC, rng)
        for j, pi in enumerate(pi_grid):
            train = make_corpus(G, pi, N_TRAIN, LENGTH, D_CONC, rng)
            test = make_corpus(G, pi, N_TEST, LENGTH, D_CONC, rng)
            gm = BackoffModel(GLOBAL_ORDER, V).fit(train)
            bg, _ = evaluate(test, gm, cache_order=k, method="global",
                             weight=0.0, g_alpha=G_ALPHA)
            bc, _ = evaluate(test, gm, cache_order=k, method="dirichlet",
                             weight=DIR_A, g_alpha=G_ALPHA)
            _, _, rel = evaluate_predictor(
                test, make_predictor(gm, approach="dirichlet", cache_order=k,
                                     weight=DIR_A, g_alpha=G_ALPHA),
                reliance=True)
            benefit[j] += (bg - bc)
            reliance[j] += float(rel[-1])
            prop[j] += propagated_leverage(test, gm, cache_order=k, a=DIR_A,
                                           g_alpha=G_ALPHA)
    n = len(seeds)
    benefit, reliance, prop = benefit / n, reliance / n, prop / n
    return dict(
        rel0=float(reliance[0]),
        ben0=float(benefit[0]),
        cprop=_corr(benefit, prop),
    )


def run(v_grid, k_grid, pi_grid, seeds):
    rows = []
    for V in v_grid:
        for k in k_grid:
            m = measure_cell(V, k, pi_grid, seeds)
            nctx = V ** k
            rows.append(dict(V=V, k=k, n_contexts=nctx, **m))
            cp = "  nan" if np.isnan(m["cprop"]) else f"{m['cprop']:+.2f}"
            print(f"V={V:>3} k={k}  V^k={nctx:>10}  "
                  f"rel0={m['rel0']:.2f}  ben0={m['ben0']:+.2f}  corr(b,prop)={cp}")
    return rows


def _collapse_quality(rows):
    """How well does rel0 collapse onto a single function of log(V^k)?

    Compare the spread of rel0 *within* iso-density groups (same V^k, different
    (V,k)) against the overall spread. Small within-group spread => collapse.
    Returns (max within-group std, overall std)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[r["n_contexts"]].append(r["rel0"])
    within = [np.std(v) for v in groups.values() if len(v) > 1]
    overall = np.std([r["rel0"] for r in rows])
    return (float(max(within)) if within else 0.0, float(overall))


def write_csv(rows):
    path = result_path("joint_sweep.csv")
    fields = ["V", "k", "n_contexts", "rel0", "ben0", "cprop"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({key: r[key] for key in fields})
    print(f"wrote {path}")


def plot(rows, v_grid, k_grid):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: collapse of rel0 onto V^k. One marker per V, x = log10(V^k).
    ax = axes[0]
    markers = {8: "o", 16: "s", 32: "^", 64: "D"}
    for V in v_grid:
        sub = sorted((r for r in rows if r["V"] == V), key=lambda r: r["n_contexts"])
        ax.plot([np.log10(r["n_contexts"]) for r in sub], [r["rel0"] for r in sub],
                markers.get(V, "o") + "-", label=f"V={V}", alpha=0.85)
    ax.axhline(0.60, color="green", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0.30, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"$\log_{10}(V^{k})$  (number of possible contexts)")
    ax.set_ylabel("cache reliance at $\\pi_{ctx}=0$")
    ax.set_title("Trust-when-useless collapses onto context count")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: coupling vs density on the same x-axis.
    ax = axes[1]
    for V in v_grid:
        sub = sorted((r for r in rows if r["V"] == V), key=lambda r: r["n_contexts"])
        xs = [np.log10(r["n_contexts"]) for r in sub]
        ys = [np.nan if np.isnan(r["cprop"]) else r["cprop"] for r in sub]
        ax.plot(xs, ys, markers.get(V, "o") + "-", label=f"V={V}", alpha=0.85)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\log_{10}(V^{k})$")
    ax.set_ylabel("corr(benefit, propagated) over $\\pi$ grid")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Coupling: unstable when dense, earned when sparse")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: rel0 heatmap over the V x k grid.
    ax = axes[2]
    grid = np.full((len(k_grid), len(v_grid)), np.nan)
    for r in rows:
        grid[k_grid.index(r["k"]), v_grid.index(r["V"])] = r["rel0"]
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(v_grid))); ax.set_xticklabels(v_grid)
    ax.set_yticks(range(len(k_grid))); ax.set_yticklabels(k_grid)
    ax.set_xlabel("vocabulary $V$"); ax.set_ylabel("cache order $k$")
    ax.set_title("reliance at $\\pi_{ctx}=0$")
    for i in range(len(k_grid)):
        for j in range(len(v_grid)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white" if grid[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = result_path("joint_sweep.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small 3x3 grid, 1 seed, coarse pi grid (fast smoke run)")
    args = ap.parse_args()
    if args.quick:
        v_grid, k_grid = (8, 16, 32), (1, 2, 3)
        pi_grid, seeds = np.array([0.0, 0.45, 0.9]), (1,)
    else:
        v_grid, k_grid = V_GRID, K_GRID
        pi_grid, seeds = np.linspace(0.0, 0.9, 5), (1, 2, 3)

    print(f"=== Joint V x order sweep (seeds={seeds}, "
          f"pi_grid={np.round(pi_grid, 3)}) ===")
    rows = run(v_grid, k_grid, pi_grid, seeds)
    within, overall = _collapse_quality(rows)
    print(f"\nrel0 collapse onto V^k: max within-iso-density std = {within:.3f} "
          f"vs overall std = {overall:.3f}  "
          f"({'collapses' if within < 0.15 and overall > 0.2 else 'see figure'})")
    write_csv(rows)
    plot(rows, list(v_grid), list(k_grid))


if __name__ == "__main__":
    main()

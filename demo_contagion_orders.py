"""
Milestones 1.2-1.5: the order ladder of exact quines (orders 2..5).

For each order k we use the canonical maximal order-k quine -- a BINARY de Bruijn
cycle B(2,k), period 2**k -- embedded on the global model's two rarest symbols
(adversarial: the global would essentially never emit it). Holding the alphabet at
2 isolates the effect of ORDER. We sweep repetitions (-> reliance) and measure:

  * per-step transition fidelity  -- expected to collapse onto the SAME closed form
    p_step = rho + (1-rho)*pg for every order (per-step contagion is order-blind);
  * run length in PERIODS          -- expected to fall fast with order: surviving one
    period needs p_step**(2**k), doubly exponential in k -> the brittleness ladder;
  * emergent dominant period       -- on failure, does the walk settle onto the full
    period 2**k or collapse onto a proper DIVISOR sub-cycle (2,4,8,...)? This is where
    period divisibility / order primality would announce themselves.

Run:  python3 demo_contagion_orders.py
Writes results/contagion_orders.png and results/contagion_collapse.png and prints tables.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from markov_cache import (peaky_transition_matrix, make_corpus, BackoffModel,
                          global_dist, make_predictor)
import contagion as C

# --------------------------------------------------------------------------- #
V            = 32
G_CONC       = 1.0
D_CONC       = 0.05
GLOBAL_ORDER = 2
G_ALPHA      = 0.1
DIR_A        = 1.0
PI_TRAIN     = 0.3
N_TRAIN, LEN_TRAIN = 200, 400
ORDERS       = [2, 3, 4, 5]
ALPHA_B      = 2          # binary de Bruijn: isolates order, period = 2**k
REPS_GRID    = [5, 8, 13, 21, 34, 55, 89, 144, 233]
N_TRIALS     = 20
SEED         = 7


def build_global():
    rng = np.random.default_rng(SEED)
    G = peaky_transition_matrix(V, G_CONC, rng)
    train = make_corpus(G, PI_TRAIN, N_TRAIN, LEN_TRAIN, D_CONC, rng)
    return BackoffModel(GLOBAL_ORDER, V).fit(train)


def rare_symbols(gm, b):
    base = global_dist(gm, [], alpha=G_ALPHA)
    return [int(s) for s in np.argsort(base)[:b]]


def order_quine(gm, k, symbols):
    """Binary de Bruijn B(2,k) relabeled onto `symbols`; an exact order-k quine."""
    raw = C.debruijn_cycle(ALPHA_B, k)
    S = [symbols[x] for x in raw]
    assert C.minimal_order(S) == k, f"expected minimal order {k}, got {C.minimal_order(S)}"
    return S


def mean_pg_correct(gm, S, k):
    p = len(S)
    vals = []
    for i in range(p):
        window = [S[(i + j) % p] for j in range(k)]
        nxt = S[(i + k) % p]
        vals.append(global_dist(gm, window, alpha=G_ALPHA)[nxt])
    return float(np.mean(vals))


def run():
    gm = build_global()
    syms = rare_symbols(gm, ALPHA_B)
    rng = np.random.default_rng(SEED + 5)
    data = {}
    for k in ORDERS:
        S = order_quine(gm, k, syms)
        period = len(S)
        pg_corr = mean_pg_correct(gm, S, k)
        gen_len = max(400, 50 * period)
        print(f"\n--- order {k}:  period 2^{k}={period}  (prime period? "
              f"{'yes' if period in (2,3,5,7,11,13,17,19,23,29,31) else 'no'})  "
              f"pg(correct)={pg_corr:.4f} ---")
        rows = []
        for reps in REPS_GRID:
            pred = make_predictor(gm, approach='dirichlet', cache_order=k,
                                  weight=DIR_A, g_alpha=G_ALPHA)
            m = C.reproduce(pred, S, reps, rng, k=k, gen_len=gen_len, n_trials=N_TRIALS)
            theory = C.predicted_per_step(m['reliance'], pg_corr)
            rows.append((reps, m['reliance'], m['transition_fidelity'], theory,
                         m['run_periods'], m['occupancy']))
            print(f"  reps={reps:3d}  rho={m['reliance']:.3f}  "
                  f"step_fid={m['transition_fidelity']:.3f}  theory={theory:.3f}  "
                  f"run_periods={m['run_periods']:6.2f}  occ={m['occupancy']:.3f}")
        data[k] = dict(S=S, period=period, pg=pg_corr, rows=np.array(rows, float))
    return gm, data


def collapse_spectrum(gm, data, reps=233):
    """At high reliance, where does each order's quine settle when it slips?
    Tally the emergent dominant period (with score >= 0.6) against divisors of 2^k."""
    rng = np.random.default_rng(SEED + 9)
    spectra = {}
    print("\n=== Collapse spectrum (emergent dominant period at high reliance) ===")
    for k in ORDERS:
        S, period = data[k]['S'], data[k]['period']
        pred = make_predictor(gm, approach='dirichlet', cache_order=k,
                              weight=DIR_A, g_alpha=G_ALPHA)
        m = C.reproduce(pred, S, reps, rng, k=k,
                        gen_len=max(400, 50 * period), n_trials=60)
        ds = [d for (d, s) in m['periods'] if s >= 0.6]
        spectra[k] = (ds, period)
        divisors = [d for d in range(1, period + 1) if period % d == 0]
        from collections import Counter
        cnt = Counter(ds)
        on_full = cnt.get(period, 0)
        on_div = sum(v for d, v in cnt.items() if d != period and period % d == 0)
        off = sum(v for d, v in cnt.items() if period % d != 0)
        print(f"  order {k} (period {period}, divisors {divisors}): "
              f"full={on_full}  proper-divisor={on_div}  off-lattice={off}  "
              f"| top: {cnt.most_common(4)}")
    return spectra


def plot_orders(data):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(ORDERS)))
    for k, col in zip(ORDERS, colors):
        reps, rho, fid, theory, runp, occ = data[k]['rows'].T
        ax.plot(rho, fid, 'o-', color=col, lw=1.8, label=f'order {k} (period {2**k})')
        ax.plot(rho, theory, '--', color=col, lw=1.0, alpha=0.6)
        ax2.semilogy(rho, np.clip(runp, 1e-2, None), 'o-', color=col, lw=1.8,
                     label=f'order {k}')
    ax.set_xlabel(r'reliance $\rho$'); ax.set_ylabel('per-step transition fidelity')
    ax.set_title('Per-step contagion is order-blind\n(all orders track $\\rho+(1-\\rho)p_g$, dashed)')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3); ax.set_ylim(0, 1.04)
    ax2.axhline(1.0, color='k', lw=0.5, ls=':')
    ax2.set_xlabel(r'reliance $\rho$'); ax2.set_ylabel('reproduced periods before first slip')
    ax2.set_title('Brittleness ladder: surviving one period\nneeds $p_{step}^{2^k}$ (doubly exp. in order)')
    ax2.legend(fontsize=8, loc='upper left'); ax2.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig('results/contagion_orders.png', dpi=130)
    print("\nwrote results/contagion_orders.png")


def plot_collapse(spectra):
    fig, axes = plt.subplots(1, len(ORDERS), figsize=(3.4 * len(ORDERS), 3.6),
                             sharey=True)
    for ax, k in zip(axes, ORDERS):
        ds, period = spectra[k]
        bins = np.arange(1, 2 * period + 2) - 0.5
        ax.hist(ds, bins=bins, color='slategray')
        for d in range(1, period + 1):
            if period % d == 0:
                ax.axvline(d, color='crimson', lw=1.0, alpha=0.6,
                           ls='-' if d == period else '--')
        ax.set_title(f'order {k}, period {period}\n(divisors dashed, full solid)', fontsize=9)
        ax.set_xlabel('emergent period')
    axes[0].set_ylabel('count (60 trials)')
    fig.suptitle('Where the quine settles when it slips: full period vs. divisor sub-cycles',
                 fontsize=11)
    fig.tight_layout(); fig.savefig('results/contagion_collapse.png', dpi=130)
    print("wrote results/contagion_collapse.png")


if __name__ == "__main__":
    print("=== Order ladder of exact quines (orders 2..5) ===")
    gm, data = run()
    plot_orders(data)
    spectra = collapse_spectrum(gm, data)
    plot_collapse(spectra)
    print("\nDone.")

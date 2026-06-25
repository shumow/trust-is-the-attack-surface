"""
Milestone 1.6 / 2: approximate quines and the divisibility / prime-period question.

The window-distinctness theorem (1.2-1.5) showed EXACT quines have no proper
sub-cycle, so divisibility is irrelevant for them. It becomes relevant only for
APPROXIMATE quines, where a divisor sub-cycle can be a competing attractor. We test
that with a quasi-periodic payload: m blocks sharing a period-d motif, distinguished
only by a leading tag. Genuine period p = m*d; resolving the blocks needs cache order
>= d; below that the payload collapses onto the period-d sub-cycle (a divisor of p).

Experiment A (order threshold, fixed composite p): sweep cache order k; emergent
generated period should jump from d (k<d, collapsed) to p (k>=d, resolved) at k=d.

Experiment B (divisibility sweep, fixed order 1): build each p's payload from its
LARGEST proper divisor (= p / smallest prime factor). Prediction: composite p settles
on that largest proper divisor; PRIME p has no sub-motif (d=1, exact rainbow) and
reproduces the full period p. Prime period == harmonic protection.

Run:  python3 demo_contagion_divis.py
Writes results/contagion_divis_order.png and results/contagion_divis_sweep.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

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
REPS         = 144
N_TRIALS     = 40
SCORE_MIN    = 0.6        # min autocorr score to call a run "settled"
SEED         = 7


def build_global():
    rng = np.random.default_rng(SEED)
    G = peaky_transition_matrix(V, G_CONC, rng)
    train = make_corpus(G, PI_TRAIN, N_TRAIN, LEN_TRAIN, D_CONC, rng)
    return BackoffModel(GLOBAL_ORDER, V).fit(train)


def rare_symbols(gm, n):
    base = global_dist(gm, [], alpha=G_ALPHA)
    return [int(s) for s in np.argsort(base)[:n]]


def make_payload(gm, p):
    """Quasi-periodic payload for period p, built from p's largest proper divisor.
    Composite p -> period-d motif with m=spf blocks; prime p -> exact rainbow."""
    spf = C.smallest_prime_factor(p)        # m = number of blocks
    m, d = spf, p // spf                     # d = largest proper divisor (=p if prime->m=p,d=1)
    syms = rare_symbols(gm, (d - 1) + m)     # motif_tail (d-1) + tags (m), all distinct
    motif_tail = syms[:d - 1]
    tags = syms[d - 1:d - 1 + m]
    S = C.quasiperiodic_quine(tags, motif_tail)
    assert C.true_period(S) == p, (p, C.true_period(S))
    return S, d, m


def settle_periods(gm, S, p, k, rng, n_trials=N_TRIALS, reps=REPS):
    """Stream the payload into an order-k cache, generate, and return the list of
    settled dominant periods (those with autocorr score >= SCORE_MIN)."""
    pred = make_predictor(gm, approach='dirichlet', cache_order=k,
                          weight=DIR_A, g_alpha=G_ALPHA)
    m = C.reproduce(pred, S, reps, rng, k=k, gen_len=max(400, 50 * p),
                    n_trials=n_trials)
    return [d for (d, s) in m['periods'] if s >= SCORE_MIN]


# --------------------------------------------------------------------------- #
def experiment_A(gm, p=12, reps_A=500):
    """Order threshold: emergent period vs cache order, fixed composite p with m=2
    blocks (binary resolution: collapse to d, or full 2d). High reps so the
    brittleness ladder doesn't starve high-order full reproduction."""
    S, d, m = make_payload(gm, p)
    min_ord = C.minimal_order(S)
    print(f"\n=== Experiment A: order threshold (p={p}, motif d={d}, blocks m={m}, "
          f"minimal_order={min_ord}, reps={reps_A}) ===")
    rng = np.random.default_rng(SEED + 11)
    orders = list(range(1, d + 2))
    med, frac_full, frac_div = [], [], []
    for k in orders:
        ds = settle_periods(gm, S, p, k, rng, reps=reps_A)
        cnt = Counter(ds)
        full = cnt.get(p, 0)
        div = sum(v for dd, v in cnt.items() if dd != p and p % dd == 0 and dd > 1)
        mode = cnt.most_common(1)[0][0] if cnt else 0
        med.append(mode); frac_full.append(full / N_TRIALS); frac_div.append(div / N_TRIALS)
        print(f"  order k={k:2d}  modal_period={mode:3d}  "
              f"frac_full(p={p})={full/N_TRIALS:.2f}  frac_divisor={div/N_TRIALS:.2f}  "
              f"top={cnt.most_common(3)}")
    return dict(p=p, d=d, m=m, min_ord=min_ord, orders=orders,
                mode=med, frac_full=frac_full, frac_div=frac_div)


def experiment_B(gm, periods=(7, 8, 9, 11, 12, 13, 15, 16, 25)):
    """Divisibility sweep at order 1: emergent period vs p, prime vs composite."""
    print("\n=== Experiment B: divisibility sweep at cache order 1 ===")
    rng = np.random.default_rng(SEED + 23)
    rows = []
    for p in periods:
        S, d, m = make_payload(gm, p)
        spf = C.smallest_prime_factor(p)
        is_prime = (spf == p)
        ds = settle_periods(gm, S, p, k=1, rng=rng)
        cnt = Counter(ds)
        mode = cnt.most_common(1)[0][0] if cnt else 0
        lpd = 1 if is_prime else p // spf       # largest proper divisor
        rows.append((p, is_prime, lpd, mode, cnt.get(p, 0) / N_TRIALS,
                     (cnt.get(lpd, 0) / N_TRIALS) if not is_prime else 0.0))
        tag = "PRIME" if is_prime else f"composite (lpd={lpd})"
        print(f"  p={p:2d} {tag:20s}  modal_period={mode:3d}  "
              f"frac_full={cnt.get(p,0)/N_TRIALS:.2f}  "
              f"frac_lpd={(cnt.get(lpd,0)/N_TRIALS) if not is_prime else 0:.2f}  "
              f"top={cnt.most_common(3)}")
    return rows


def plot_A(A):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(A['orders'], A['frac_div'], 'o-', color='seagreen', lw=2,
            label=f"divisor collapse (period $d={A['d']}$)")
    ax.plot(A['orders'], A['frac_full'], 's-', color='crimson', lw=2,
            label=f"full reproduction (period $p={A['p']}$)")
    ax.axvline(A['min_ord'], color='gray', ls='-.', lw=1.2, alpha=0.8,
               label=f"resolving order $k=d={A['min_ord']}$")
    ax.set_xlabel('cache order $k$'); ax.set_ylabel('fraction of runs')
    ax.set_title(f"Order threshold ($p={A['p']}=d\\cdot m$, $d={A['d']}$, $m={A['m']}$):\n"
                 f"the divisor harmonic vanishes exactly at $k=d$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.0)
    fig.tight_layout(); fig.savefig('results/contagion_divis_order.png', dpi=130)
    print("\nwrote results/contagion_divis_order.png")


def plot_B(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for p, is_prime, lpd, mode, ff, fl in rows:
        col = 'crimson' if is_prime else 'slategray'
        ax.scatter(p, mode, s=80, color=col, zorder=3,
                   marker='*' if is_prime else 'o')
    ax.plot([r[0] for r in rows], [r[0] for r in rows], 'k--', lw=0.8, alpha=0.5,
            label='full period (y=p)')
    # largest-proper-divisor reference for composites
    comp = [(p, lpd) for (p, isp, lpd, *_ ) in rows if not isp]
    ax.scatter([p for p, _ in comp], [lpd for _, lpd in comp], facecolors='none',
               edgecolors='seagreen', s=140, lw=1.4, zorder=2,
               label='largest proper divisor (composite)')
    ax.scatter([], [], color='crimson', marker='*', s=80, label='prime p (reproduces p)')
    ax.scatter([], [], color='slategray', marker='o', s=80, label='composite p (collapses)')
    ax.set_xlabel('payload period $p$'); ax.set_ylabel('modal emergent period (order-1 cache)')
    ax.set_title('Divisibility gates collapse: composites fall to their largest proper\n'
                 'divisor; primes have no sub-motif and reproduce the full period')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('results/contagion_divis_sweep.png', dpi=130)
    print("wrote results/contagion_divis_sweep.png")


if __name__ == "__main__":
    gm = build_global()
    A = experiment_A(gm, p=15)
    plot_A(A)
    rows = experiment_B(gm)
    plot_B(rows)
    print("\nDone.")

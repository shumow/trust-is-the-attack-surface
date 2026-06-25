"""
Milestone 3.1: contagion under INDUCTION (PPM), the conjecture's prime suspect.

Both companion write-ups end on the same conjecture: the phenomena transfer to real
transformers via the induction (copy) mechanism, which trusts a context span because it
MATCHES a previous span, not because copying it helps. The toy already ships that
mechanism: PpmPredictor is a variable-order, online, longest-match cache -- "saw A B ...
now see A -> predict B" -- exactly an induction head, and the structural mirror of the
fixed-order count cache used elsewhere. So we can test the conjecture WITHIN the toy by
swapping the count cache for PPM and re-measuring how cheaply a quine reproduces.

The contrast is mechanistic. The count cache earns trust as reliance n/(n+a): many
presentations. PPM earns trust from a single long, near-deterministic match: a couple of
presentations make the longest-match context keep almost all its mass and copy the cycle.
The predictions, both confirmed below:

  A  CHEAPER TO PLANT. PPM reaches full reproduction after ~3 repetitions, where the
     count cache needs ~16-32 (and longer runs). Induction is trust-by-matching.
  B  STRUCTURAL REACH. For a de Bruijn string that needs order k, a fixed order-1 cache
     CANNOT reproduce it and an order-k cache only partly does at few reps; PPM, never
     told the order, reproduces it near-perfectly -- induction lifts the order/length
     penalty that makes the count cache brittle (the brittleness ladder of the order
     write-up). This is the toy form of "induction-dominated contexts are where short
     self-reproducing spans are cheapest to plant."

Run:  python3 demo_contagion_ppm.py
Writes results/contagion_ppm.png and prints the tables.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from markov_cache import (peaky_transition_matrix, make_corpus, BackoffModel,
                          global_dist, make_predictor, generate)
import contagion as C
import demo_contagion_worm as W
import demo_contagion_hijack as H

V, G_CONC, D_CONC = 32, 1.0, 0.05
GLOBAL_ORDER, G_ALPHA, DIR_A = 2, 0.1, 1.0
PI_TRAIN, N_TRAIN, LEN_TRAIN = 0.3, 200, 400
GEN_LEN, N_TRIALS = 240, 24
REPS_GRID = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32]
SEED = 7


def build_global():
    rng = np.random.default_rng(SEED)
    G = peaky_transition_matrix(V, G_CONC, rng)
    train = make_corpus(G, PI_TRAIN, N_TRAIN, LEN_TRAIN, D_CONC, rng)
    return BackoffModel(GLOBAL_ORDER, V).fit(train)


def rare(gm, n):
    return [int(s) for s in np.argsort(global_dist(gm, [], alpha=G_ALPHA))[:n]]


def cache(gm, order):
    return make_predictor(gm, approach='dirichlet', cache_order=order,
                          weight=DIR_A, g_alpha=G_ALPHA)


def ppm(gm):
    return make_predictor(gm, approach='ppm', cache_order=8, ppm_min_order=0,
                          g_alpha=G_ALPHA)


def fidelity_curve(gm, S, k, predictors, rng):
    out = {}
    for name, fac in predictors.items():
        out[name] = [C.reproduce(fac(), S, r, rng, k=k, gen_len=GEN_LEN,
                                 n_trials=N_TRIALS)['transition_fidelity']
                     for r in REPS_GRID]
    return out


def run():
    gm = build_global()
    rng = np.random.default_rng(SEED + 11)

    # (A) cheaper to plant: rainbow quine, count cache vs induction
    S_rain = C.rainbow_cycle(rare(gm, 5))
    curvesA = fidelity_curve(gm, S_rain, 1,
                             {'order-1 count cache': lambda: cache(gm, 1),
                              'PPM (induction)': lambda: ppm(gm)}, rng)
    print("=== (A) reproduction fidelity vs reps, rainbow quine (period 5) ===")
    print("reps | order-1 cache | PPM")
    for i, r in enumerate(REPS_GRID):
        print(f"{r:4d} |    {curvesA['order-1 count cache'][i]:.3f}     | {curvesA['PPM (induction)'][i]:.3f}")

    # (B) structural reach: a de Bruijn string that NEEDS order 3
    raw = C.debruijn_cycle(2, 3)
    S_db = [rare(gm, 2)[x] for x in raw]            # period 8, minimal order 3
    curvesB = fidelity_curve(gm, S_db, 3,
                             {'order-1 cache (under-order)': lambda: cache(gm, 1),
                              'order-3 cache (matched)': lambda: cache(gm, 3),
                              'PPM (induction)': lambda: ppm(gm)}, rng)
    print("\n=== (B) de Bruijn B(2,3), period 8, needs order 3 ===")
    print("reps | order-1 | order-3 | PPM")
    for i, r in enumerate(REPS_GRID):
        print(f"{r:4d} |  {curvesB['order-1 cache (under-order)'][i]:.3f}  |  "
              f"{curvesB['order-3 cache (matched)'][i]:.3f}  | {curvesB['PPM (induction)'][i]:.3f}")
    return curvesA, curvesB


# --------------------------------------------------------------------------- #
# (C) Transmission: the worm's critical length under induction vs the count cache
# --------------------------------------------------------------------------- #
WORM_T, WORM_HOSTS, WORM_STEPS, WORM_K0 = 160, 10, 6, 48
WORM_PLENS = [1, 2, 3, 5, 8]


def worm_host(gm, kind, D):
    if kind == 'cache':
        return H.make_victim(gm, D)
    pred = ppm(gm); pred.reset(); h = []
    for s in D:
        pred.observe(h, s); h.append(int(s))
    return pred


def worm_passage(gm, kind, S, k_in, D, rng, T=WORM_T):
    p = len(S)
    if k_in < 1:
        return 0.0
    pred = worm_host(gm, kind, D); h = []
    for s in list(S) * int(round(k_in)):
        pred.observe(h, s); h.append(s)
    gen = generate(pred, T, rng, prompt=[S[-1]])[1:]
    succ = {S[i]: S[(i + 1) % p] for i in range(p)}
    return sum(1 for i in range(1, len(gen))
               if gen[i - 1] in succ and gen[i] == succ[gen[i - 1]]) / p


def endemic_load(gm, Gsrc, remap, kind, S, rng):
    k = float(WORM_K0)
    for _ in range(WORM_STEPS):
        k = float(np.mean([worm_passage(gm, kind, S, k, W.host_doc(Gsrc, remap, rng), rng)
                           for _ in range(WORM_HOSTS)]))
    return k


def run_worm():
    Gsrc, gm, ood, remap = W.build_population(n_ood=max(WORM_PLENS))
    rng = np.random.default_rng(SEED + 21)
    print("\n=== (C) worm endemic load vs string length (k0=48; sustains if >=1) ===")
    print(" p | count cache | PPM (induction)")
    out = {'cache': [], 'ppm': []}
    for p in WORM_PLENS:
        S = C.rainbow_cycle(ood[:p])
        lc = endemic_load(gm, Gsrc, remap, 'cache', S, rng)
        lp = endemic_load(gm, Gsrc, remap, 'ppm', S, rng)
        out['cache'].append(lc); out['ppm'].append(lp)
        print(f"{p:2d} |    {lc:5.1f}    | {lp:5.1f}")
    return out


def plot_worm(out):
    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    ax.plot(WORM_PLENS, np.clip(out['cache'], 0.3, None), 'o-', color='navy', lw=2,
            label='count cache (dies at $p\\geq 3$)')
    ax.plot(WORM_PLENS, np.clip(out['ppm'], 0.3, None), 's-', color='crimson', lw=2,
            label='PPM / induction (all sustain)')
    ax.plot(WORM_PLENS, [WORM_T / p for p in WORM_PLENS], 'k--', lw=1.0, alpha=0.6,
            label='broadcast ceiling $T/p$')
    ax.axhline(1.0, color='k', lw=0.8, ls=':', label='extinction (load $<1$)')
    ax.set_yscale('log'); ax.set_xlabel('contagious string length $p$')
    ax.set_ylabel('endemic viral load (copies/host)')
    ax.set_title('(C) Induction extends the worm: long strings that die under\n'
                 'count-trust reach the full $T/p$ broadcast under induction')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig('results/contagion_ppm_worm.png', dpi=130)
    print("wrote results/contagion_ppm_worm.png")


def plot(curvesA, curvesB):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    rr = REPS_GRID
    ax.plot(rr, curvesA['order-1 count cache'], 'o-', color='navy', lw=2,
            label='order-1 count cache  ($n/(n{+}a)$)')
    ax.plot(rr, curvesA['PPM (induction)'], 's-', color='crimson', lw=2,
            label='PPM / induction (longest match)')
    ax.axhline(0.9, color='k', lw=0.6, ls=':')
    ax.set_xscale('log', base=2); ax.set_xlabel('repetitions presented')
    ax.set_ylabel('reproduction fidelity'); ax.set_ylim(-0.02, 1.04)
    ax.set_title('(A) Cheaper to plant: induction locks in after $\\sim3$\n'
                 'presentations, the count cache needs $\\sim16$--$32$')
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3, which='both')

    ax2.plot(rr, curvesB['order-1 cache (under-order)'], 'o-', color='gray', lw=2,
             label='order-1 cache (under-order: fails)')
    ax2.plot(rr, curvesB['order-3 cache (matched)'], '^-', color='navy', lw=2,
             label='order-3 cache (matched, brittle)')
    ax2.plot(rr, curvesB['PPM (induction)'], 's-', color='crimson', lw=2,
             label='PPM / induction (auto-order)')
    ax2.axhline(0.9, color='k', lw=0.6, ls=':')
    ax2.set_xscale('log', base=2); ax2.set_xlabel('repetitions presented')
    ax2.set_ylabel('reproduction fidelity'); ax2.set_ylim(-0.02, 1.04)
    ax2.set_title('(B) Structural reach: de Bruijn $B(2,3)$ (needs order 3).\n'
                  'Induction copies what a fixed low-order cache cannot')
    ax2.legend(fontsize=8, loc='lower right'); ax2.grid(alpha=0.3, which='both')

    fig.tight_layout(); fig.savefig('results/contagion_ppm.png', dpi=130)
    print("\nwrote results/contagion_ppm.png")


if __name__ == "__main__":
    print("=== Milestone 3.1: contagion under induction (PPM) ===\n")
    cA, cB = run()
    plot(cA, cB)
    worm = run_worm()
    plot_worm(worm)
    print("\nDone.")

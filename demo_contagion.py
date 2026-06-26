"""
Milestone 1: does an exact quine actually reproduce, and does it follow the theory?

An empty order-1 cache, a fixed global model, and one payload: a RAINBOW CYCLE
(A -> B -> C -> ... -> A) through the global model's *rarest* symbols -- the
white-box adversarial choice, a string the model would essentially never emit on
its own, so any reproduction is the cache's trust and not the global's taste.

We stream `reps` copies of the quine into the empty cache (driving reliance
n/(n+a)), then generate and measure how faithfully the cache regenerates the
cycle. We overlay the closed form `predicted_per_step`: a static lower bound that
generation's self-reinforcement is expected to beat. The result is the
reproduction analogue of the parent repo's condensation knee.

Run:  python3 demo_contagion.py
Writes results/contagion_fidelity.png and prints the reps -> reliance -> fidelity table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo_utils import result_path
from markov_cache import (peaky_transition_matrix, make_corpus, BackoffModel,
                          global_dist, make_predictor)
import contagion as C

# --------------------------------------------------------------------------- #
# Config (shares the parent repo's source so results are comparable)
# --------------------------------------------------------------------------- #
V            = 32
G_CONC       = 1.0
D_CONC       = 0.05
GLOBAL_ORDER = 2
G_ALPHA      = 0.1
DIR_A        = 1.0       # Dirichlet prior strength a (global-prior weight a/(n+a))
PI_TRAIN     = 0.3
N_TRAIN, LEN_TRAIN = 200, 400
CYCLE_LEN    = 5
GEN_LEN      = 300
N_TRIALS     = 24
REPS_GRID    = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
SEED         = 7


def build_global():
    rng = np.random.default_rng(SEED)
    G = peaky_transition_matrix(V, G_CONC, rng)
    train = make_corpus(G, PI_TRAIN, N_TRAIN, LEN_TRAIN, D_CONC, rng)
    gm = BackoffModel(GLOBAL_ORDER, V).fit(train)
    return gm


def rarest_cycle(gm, length):
    """The `length` globally-rarest symbols, as an order-1 rainbow quine. These are
    the cheap-to-hijack 'weak keys' an attacker targets; they also make pg_correct
    small, so reproduction must be carried by cache trust, not global taste."""
    base = global_dist(gm, [], alpha=G_ALPHA)
    rarest = list(np.argsort(base)[:length])
    return C.rainbow_cycle(rarest)


def mean_pg_correct(gm, S):
    """Average global prior on the quine's own (adversarial) transitions: for each
    s in the cycle, pg(next | s). Feeds the closed-form theory curve."""
    p = len(S)
    vals = []
    for i in range(p):
        s, nxt = S[i], S[(i + 1) % p]
        vals.append(global_dist(gm, [s], alpha=G_ALPHA)[nxt])
    return float(np.mean(vals))


def run():
    gm = build_global()
    S = rarest_cycle(gm, CYCLE_LEN)
    assert C.is_deterministic(S, 1), "rainbow cycle must be an exact order-1 quine"
    pg_corr = mean_pg_correct(gm, S)
    print(f"quine = {S}  (order-1 rainbow, rarest symbols)")
    print(f"mean pg(correct transition) = {pg_corr:.4f}   [global would rarely emit this]\n")

    rng = np.random.default_rng(SEED + 3)
    rows = []
    for reps in REPS_GRID:
        pred = make_predictor(gm, approach='dirichlet', cache_order=1,
                              weight=DIR_A, g_alpha=G_ALPHA)
        m = C.reproduce(pred, S, reps, rng, k=1, gen_len=GEN_LEN, n_trials=N_TRIALS)
        rel = m['reliance']
        theory = C.predicted_per_step(rel, pg_corr)
        rows.append((reps, rel, m['transition_fidelity'], m['occupancy'],
                     m['run_length'], theory))
        print(f"reps={reps:3d}  reliance={rel:.3f}  trans_fid={m['transition_fidelity']:.3f}  "
              f"occupancy={m['occupancy']:.3f}  run_len={m['run_length']:6.1f}  "
              f"theory={theory:.3f}")
    return S, pg_corr, np.array(rows, dtype=float)


def plot(S, pg_corr, rows):
    reps, rel, fid, occ, run, theory = rows.T
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax.plot(rel, fid, 'o-', lw=2, color='crimson',
            label='measured transition fidelity')
    ax.plot(rel, theory, 's--', lw=1.6, color='navy', alpha=0.85,
            label=r'static theory $\rho + (1-\rho)\,p_g$')
    ax.plot(rel, occ, '^:', lw=1.3, color='seagreen', alpha=0.7,
            label='quine-alphabet occupancy')
    ax.axhline(1.0, color='k', lw=0.5)
    ax.set_xlabel('cache reliance on payload context  $\\rho = n/(n+a)$')
    ax.set_ylabel('per-step reproduction')
    ax.set_title(f'Order-1 rainbow quine (len {len(S)}): transition fidelity tracks theory\n'
                 f'$p_g$(correct)$={pg_corr:.3f}$ — global would not emit this')
    ax.legend(fontsize=8, loc='center right'); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.04)

    # run length: measured run vs predicted geometric run length
    pred_run = np.array([C.predicted_run_length(r, pg_corr) for r in rel])
    ax2.semilogy(rel, np.clip(run, 0.5, None), 'o-', lw=2, color='crimson',
                 label='measured run length (with reinforcement)')
    ax2.semilogy(rel, np.clip(pred_run, 0.5, None), 's--', lw=1.6, color='navy',
                 alpha=0.85, label='static geometric run length $1/(1-p)$')
    ax2.set_xlabel('cache reliance on payload context  $\\rho$')
    ax2.set_ylabel('correct steps before first slip')
    ax2.set_title('Run length diverges at the knee\n(reinforcement beats the static bound)')
    ax2.legend(fontsize=8, loc='upper left'); ax2.grid(alpha=0.3, which='both')

    fig.tight_layout(); fig.savefig(result_path('contagion_fidelity.png'), dpi=130)
    print("\nwrote results/contagion_fidelity.png")


if __name__ == "__main__":
    print("=== Exact-quine reproduction vs. reliance (order-1 rainbow) ===\n")
    S, pg_corr, rows = run()
    plot(S, pg_corr, rows)
    print("\nDone.")

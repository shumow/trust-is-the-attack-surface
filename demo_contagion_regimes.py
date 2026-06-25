"""
Milestone 1.7: closing out Q1's two attacker regimes.

Empty cache, fixed global model. The attacker wants the MINIMAL payload (repetitions
r, hence injected counts) to make an order-1 rainbow quine reproduce with expected
on-quine run length >= T. Two knowledge regimes:

  Regime A (WHITE-BOX): the attacker knows the global prior pg at the payload contexts
    and the Dirichlet strength a, so it computes the optimum in closed form:
        r*  =  a * (T*(1 - pg) - 1).              [contagion.white_box_min_reps]

  Regime B (BLACK-BOX): the attacker knows neither pg nor a, only sampled generations.
    - Zero queries: it can still GUARANTEE contagion with the pg=0 worst-case bound
        r0  =  a * (T - 1)                          [contagion.zero_knowledge_min_reps]
      because reliance -> 1 dominates any global. No knowledge of G needed.
    - With queries: an adaptive doubling+bisection search on r, using only measured
      run length as feedback, converges to ~r*. The query cost is the price of the gap.

Headline: contagion is always achievable black-box (the r0 bound is model-agnostic),
and the value of knowing the model, r0 - r* = a*T*pg, VANISHES as pg -> 0 -- exactly
for the stealthiest payloads. Knowing G buys the attacker the most against LOUD
payloads the model already likes, and nothing against maximally rare ones.

Run:  python3 demo_contagion_regimes.py
Writes results/contagion_regimes.png and prints the per-payload table.
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
G_CONC       = 0.3         # PEAKIER than the parent's 1.0: gives a real loudness axis
                           # (some transitions strongly predictable) so the white-box
                           # gain is visible. The high-entropy case (G_CONC=1.0) is the
                           # all-stealthy corollary -- gap ~0 everywhere; see notes.
D_CONC       = 0.05
GLOBAL_ORDER = 2
G_ALPHA      = 0.1
DIR_A        = 1.0          # the deployed model's Dirichlet strength a (unknown to B)
PI_TRAIN     = 0.3
N_TRAIN, LEN_TRAIN = 200, 400
TARGET_Q     = 0.97        # target per-step poison probability (low-variance observable)
T_EQUIV      = 1.0 / (1.0 - TARGET_Q)   # equivalent run length, for the closed forms
GEN_LEN      = 200
SEED         = 7


def build_global():
    rng = np.random.default_rng(SEED)
    G = peaky_transition_matrix(V, G_CONC, rng)
    train = make_corpus(G, PI_TRAIN, N_TRAIN, LEN_TRAIN, D_CONC, rng)
    return BackoffModel(GLOBAL_ORDER, V).fit(train)


def self_loop_pg(gm, x):
    """Effective global self-transition prob pg(x | x..x) at the global's own order --
    the per-step prior the combiner sees for the self-loop payload x* -> x*."""
    return float(global_dist(gm, [x] * GLOBAL_ORDER, alpha=G_ALPHA)[x])


def make_payloads(gm, n=10):
    """Self-loop payloads x* -> x* (the condensation primitive), choosing x* to span
    the loudness axis: pg(x*|x*) ranges widely across the vocab, so a graded set of
    symbols gives a clean low->high pg sweep with a single, low-variance observable."""
    pgs = np.array([self_loop_pg(gm, x) for x in range(V)])
    order = np.argsort(pgs)                                  # quiet -> loud
    pick = np.linspace(0, V - 1, n).round().astype(int)
    return [[int(order[i])] for i in pick]                  # each payload = [x*]


def measure_pstep(gm, S, reps, rng, n_samples):
    """Black-box observable: the one-step poison probability p_step = P(next=x* | x*)
    after streaming `reps`, estimated by a CHOSEN-PREFIX oracle -- prime the model with
    the payload context and observe the next token, n_samples times. Unbiased and
    low-variance, and exactly the quantity the white-box closed form predicts. Query
    cost = n_samples. (The attacker reads nothing it couldn't get by prompting +
    watching one emitted token.)"""
    pred = make_predictor(gm, approach='dirichlet', cache_order=1,
                          weight=DIR_A, g_alpha=G_ALPHA)
    C.stream_into(pred, S, reps)
    d = pred.dist(list(S))                     # context = the payload's last symbol(s)
    draws = rng.choice(len(d), size=n_samples, p=d)
    return float(np.mean(draws == S[-1])), n_samples


def black_box_attack(gm, S, rng, target_q, *, n_samples, max_reps=8192):
    """Regime B algorithm. Doubling search then bisection on reps, using ONLY the
    sampled one-step poison probability (no pg, no a). Returns (r_found, queries)."""
    queries = 0
    def f(r):
        nonlocal queries
        val, q = measure_pstep(gm, S, r, rng, n_samples)
        queries += q
        return val
    lo, hi = 0, 1
    while f(hi) < target_q and hi < max_reps:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) >= target_q:
            hi = mid
        else:
            lo = mid
    return hi, queries


def run_theory(gm):
    """Exact regime curves across the loudness axis (no sampling): zero-knowledge r0,
    white-box optimum r*, and the gap r0-r* = a*T*pg = the value of knowing G."""
    rows = []
    print("--- Theory (exact): regime gap vs payload loudness ---")
    for S in make_payloads(gm, n=12):
        pg = self_loop_pg(gm, S[0])
        r_star = C.white_box_min_reps(pg, DIR_A, T_EQUIV)
        r_zero = C.zero_knowledge_min_reps(DIR_A, T_EQUIV)
        rows.append((pg, r_star, r_zero))
        print(f"  pg={pg:.4f}  r*={r_star:5.1f}  r0={r_zero:4.0f}  gap={r_zero-r_star:4.1f}")
    return np.array(rows, float)


def run_blackbox_convergence(gm, n_grid=(50, 100, 200, 500, 1000, 2000, 5000),
                             repeats=8, pg_target=0.08):
    """Regime B cost: for one mid-loudness payload, how the adaptive search's found
    reps r_bb converges to the white-box optimum r* as it spends queries."""
    # pick the payload whose pg is closest to pg_target
    cands = make_payloads(gm, n=12)
    S = min(cands, key=lambda s: abs(self_loop_pg(gm, s[0]) - pg_target))
    pg = self_loop_pg(gm, S[0])
    r_star = C.white_box_min_reps(pg, DIR_A, T_EQUIV)
    r_zero = C.zero_knowledge_min_reps(DIR_A, T_EQUIV)
    print(f"\n--- Black-box convergence (payload pg={pg:.4f}, "
          f"r*={r_star:.1f}, r0={r_zero:.0f}) ---")
    out = []
    for n in n_grid:
        rs, qs = [], []
        for t in range(repeats):
            r_bb, q = black_box_attack(gm, S, np.random.default_rng(SEED + 100 + t),
                                       TARGET_Q, n_samples=n)
            rs.append(r_bb); qs.append(q)
        out.append((float(np.mean(qs)), float(np.mean(rs)), float(np.std(rs))))
        print(f"  n_samples={n:5d}  mean_queries={np.mean(qs):7.0f}  "
              f"r_bb={np.mean(rs):5.1f} ± {np.std(rs):.1f}")
    return pg, r_star, r_zero, np.array(out, float)


def plot(theory, conv):
    pg, r_star, r_zero = theory.T
    o = np.argsort(pg)
    pg, r_star, r_zero = pg[o], r_star[o], r_zero[o]
    pgc, rc_star, rc_zero, cdata = conv
    q_ax, r_bb, r_sd = cdata.T

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))

    ax.plot(pg, r_zero, 'D-', color='gray', lw=1.8,
            label='$r_0$ zero-knowledge ($p_g{=}0$ bound)')
    ax.plot(pg, r_star, 's-', color='navy', lw=2.0, label='$r^*$ white-box optimum')
    ax.fill_between(pg, r_star, r_zero, color='gold', alpha=0.3,
                    label='value of knowing $G$  ($r_0-r^*=aTp_g$)')
    ax.set_xlabel('payload prior $p_g$  (loudness $\\rightarrow$)')
    ax.set_ylabel(f'repetitions for one-step poison prob $q={TARGET_Q}$')
    ax.set_title('Regime gap closes for stealthy payloads\n'
                 'white-box buys nothing as $p_g\\to0$ (the adversarial corner)')
    ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=0.3)

    ax2.axhline(rc_zero, color='gray', ls='--', lw=1.4, label='$r_0$ zero-query fallback')
    ax2.axhline(rc_star, color='navy', ls='-', lw=1.4, label='$r^*$ white-box optimum')
    ax2.errorbar(q_ax, r_bb, yerr=r_sd, fmt='o-', color='seagreen', lw=1.8, capsize=3,
                 label='$r_{bb}$ adaptive black-box search')
    ax2.set_xscale('log')
    ax2.set_xlabel('black-box queries spent (tokens observed)')
    ax2.set_ylabel('reps found')
    ax2.set_title(f'Black-box recovers the optimum, paying queries\n'
                  f'(payload $p_g={pgc:.3f}$): $r_{{bb}}\\to r^*$ as queries grow')
    ax2.legend(fontsize=8, loc='upper right'); ax2.grid(alpha=0.3, which='both')

    fig.tight_layout(); fig.savefig('results/contagion_regimes.png', dpi=130)
    print("\nwrote results/contagion_regimes.png")


if __name__ == "__main__":
    print(f"=== Q1 regimes: white-box vs black-box minimal payload "
          f"(q={TARGET_Q}, self-loop x*) ===\n")
    theory = run_theory(gm := build_global())
    conv = run_blackbox_convergence(gm)
    plot(theory, conv)
    print("\nDone.")

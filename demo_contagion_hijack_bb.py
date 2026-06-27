"""
Q2 / Milestone 2.2: BLACK-BOX entry-finding (carrying Q1's regime split into delivery).

The attacker hijacks a cache warmed on D (as in 2.1) but is black-box: it cannot read
the victim's visit frequencies freq(c) or its warmed counts n_c. The question: to find
the MIN-POISON bridge, must the attacker actively probe leverage, or does passive
observation suffice?

The min-poison objective selects argmax_c (freq(c)-r)/(n_c+a) with r the entry-rate
ceiling (NOT the marginal argmax freq/(n_c+a), which only governs infinitesimal poison;
the two differ near the ceiling). Three attackers, each scored by the TRUE analytic
minimal poison of the context it picks (smallest w with P_enter >= 1/2):

  white-box   : knows freq, n_c exactly -> the optimal bridge (the floor).
  BB-observe  : estimates freq by WATCHING M tokens only; picks argmax freq_hat (most
                visited). Cheap, blind to leverage.
  BB-probe    : also estimates n_c by an ACTIVE PROBE -- inject w_probe OOD counts at a
                candidate c, sample the shift in p(x*|c) -> n_hat = w_probe/p_hat -
                w_probe - a (the Q1 one-step oracle). Picks argmax (freq_hat-r)/(n_hat+a).

Finding (mirrors Q1's stealthy corner): for the min-poison objective, VISIT FREQUENCY
is the dominant signal -- the (freq-r) term swamps leverage -- so BB-observe recovers
the white-box optimum from observation ALONE (~2000 tokens), and active leverage-probing
is NOT worth its query cost (noisy n_hat even mildly hurts). Black-box ~ white-box for
entry-finding. (Leverage only matters in the marginal, budget-starved regime of 2.1,
where the attacker cannot afford to reach the target and max freq/(n+a) wins.)

Run:  python3 demo_contagion_hijack_bb.py
Writes results/contagion_hijack_bb.png and prints the table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo_utils import result_path
from markov_cache import generate
import demo_contagion_hijack as H   # reuse build/victim/leverage helpers and constants

A      = H.DIR_A
T      = H.T_ENTRY
TARGET = 0.5         # entry-probability target the attacker wants
K_CAND = 8           # candidates BB-probe is willing to probe (top-K by freq_hat)
W_PROBE = 4          # counts injected per leverage probe
N_SAMP  = 200        # chosen-prefix samples per probe
M_GRID  = [200, 500, 1000, 2000, 5000, 10000]
REPEATS = 10
SEED    = 7

R_CEIL = 1.0 - (1.0 - TARGET) ** (1.0 / T)   # entry-rate ceiling: need freq > R_CEIL


def minpoison_score(freq, n, a=A, r=R_CEIL):
    """Selection criterion for the MIN-POISON objective: maximizing (freq-r)/(n+a)
    minimizes the poison needed to reach the target (vs freq/(n+a), which only
    optimizes the marginal w->0 entry rate; the two differ near the ceiling)."""
    return (freq - r) / (n + a) if freq > r else -1.0


def analytic_min_poison(freq, n, a=A, T=T, target=TARGET):
    """Smallest bridge poison w with P_enter = 1-(1-freq*w/(n+w+a))^T >= target.
    Returns inf if freq is below the rate ceiling r = 1-(1-target)^(1/T)."""
    r = 1.0 - (1.0 - target) ** (1.0 / T)
    if freq <= r:
        return np.inf
    return float(np.ceil(r * (n + a) / (freq - r)))


def observe_freq(gm, D, prompt, rng, M, L=60):
    """Black-box observation: watch generated tokens until M collected, tally context
    visit frequencies. Re-warm the victim per generation (generate() mutates)."""
    freq = np.zeros(H.V); tot = 0
    while tot < M:
        pred = H.make_victim(gm, D)
        out = generate(pred, L, rng, prompt=list(prompt))[len(prompt):]
        for s in out:
            freq[int(s)] += 1; tot += 1
    return freq / freq.sum()


def probe_leverage(gm, D, c, x_star, rng):
    """Active leverage probe: inject W_PROBE OOD counts at c->x*, sample p(x*|c) by the
    chosen-prefix oracle, invert to n_hat. Query cost = N_SAMP."""
    pred = H.make_victim(gm, D)
    for _ in range(W_PROBE):
        pred.observe([c], x_star)
    d = pred.dist([c])
    p = float(np.mean(rng.choice(len(d), size=N_SAMP, p=d) == x_star))
    if p <= 0:
        return np.inf
    return max(W_PROBE / p - W_PROBE - A, 0.0)


def ground_truth(gm, D, prompt, rng, x_star):
    victim = H.make_victim(gm, D)
    n_true = np.array([H.local_count(victim, [c]) for c in range(H.V)])
    freq_true = observe_freq(gm, D, prompt, rng, M=60000)
    score = np.array([minpoison_score(freq_true[c], n_true[c]) for c in range(H.V)])
    score[x_star] = -1
    c_wb = int(np.argmax(score))
    return freq_true, n_true, c_wb


def run():
    H.LEN_D = 1200                          # longer victim doc: warming (n_c) varies more
    gm, D, x_star = H.build_global_and_doc()
    prompt = [int(D[0])]
    rng = np.random.default_rng(SEED + 4)
    freq_true, n_true, c_wb = ground_truth(gm, D, prompt, rng, x_star)
    mp_wb = analytic_min_poison(freq_true[c_wb], n_true[c_wb])
    print(f"x*={x_star}; white-box bridge ctx={c_wb} "
          f"(freq={freq_true[c_wb]:.4f}, n_c={n_true[c_wb]:.0f}), min poison={mp_wb:.0f}\n")

    rows = []
    for M in M_GRID:
        obs_mp, probe_mp, probe_q = [], [], []
        for t in range(REPEATS):
            r = np.random.default_rng(SEED + 200 + t)
            fhat = observe_freq(gm, D, prompt, r, M)
            fhat[x_star] = -1
            # BB-observe: most-visited
            c_obs = int(np.argmax(fhat))
            obs_mp.append(analytic_min_poison(freq_true[c_obs], n_true[c_obs]))
            # BB-probe: probe top-K candidates' leverage
            cand = [int(c) for c in np.argsort(fhat)[::-1][:K_CAND]]
            best, best_score = cand[0], -2.0
            for c in cand:
                n_hat = probe_leverage(gm, D, c, x_star, r)
                s = minpoison_score(fhat[c], n_hat)
                if s > best_score:
                    best_score, best = s, c
            probe_mp.append(analytic_min_poison(freq_true[best], n_true[best]))
            probe_q.append(M + K_CAND * N_SAMP)
        rows.append((M, np.mean(obs_mp), np.std(obs_mp),
                     np.mean(probe_mp), np.std(probe_mp), np.mean(probe_q)))
        print(f"M={M:6d}  BB-observe min_poison={np.mean(obs_mp):5.1f}±{np.std(obs_mp):4.1f}  "
              f"BB-probe={np.mean(probe_mp):5.1f}±{np.std(probe_mp):4.1f}  "
              f"(probe queries={M + K_CAND*N_SAMP})")
    return mp_wb, np.array(rows, float)


def plot(mp_wb, rows):
    M, obs_m, obs_s, pr_m, pr_s, pr_q = rows.T
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.axhline(mp_wb, color='navy', ls='-', lw=1.8, label=f'white-box optimum ({mp_wb:.0f} counts)')
    ax.errorbar(M, obs_m, yerr=obs_s, fmt='o-', color='seagreen', lw=1.8, capsize=3,
                label='BB-observe (watch freq only)')
    ax.errorbar(M, pr_m, yerr=pr_s, fmt='s--', color='crimson', lw=1.4, capsize=3,
                alpha=0.8, label='BB-probe (freq + active leverage probe)')
    ax.set_xscale('log')
    ax.set_xlabel('observation budget $M$ (tokens watched)')
    ax.set_ylabel('bridge min poison (counts) achieved')
    ax.set_title('Q2 black-box entry-finding: observation alone recovers the optimum.\n'
                 'Frequency dominates; active leverage-probing is not worth its queries')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(result_path('contagion_hijack_bb.png'), dpi=130)
    print("\nwrote results/contagion_hijack_bb.png")


if __name__ == "__main__":
    print("=== Q2.2: black-box entry-finding ===\n")
    mp_wb, rows = run()
    plot(mp_wb, rows)
    print("\nDone.")

"""
Q2 / Milestone 2.5: JOINT minimal poison -- optimize entry and lock-in together.

Every prior hijack pinned one lever and swept the other (2.1 fixed lock-in, swept the
bridge; 2.3 fixed the bridge, swept the payload). But the two are coupled through

    P(hijack) = P(enter in T | w_bridge) x P(lock-in | reps),

so the cheapest hijack lives on a 2-D frontier, not at either pinned extreme. We sweep
the full grid (w_bridge, reps), measure P(hijack), and read off the minimum total poison
  total = w_bridge + reps * p
on the P(hijack) >= target contour -- then compare to the wasteful "pin lock-in strong"
recipe of 2.1.

Coupling intuition: strong lock-in (high reps) makes a SINGLE entry stick, so the bridge
can be cheap; weak lock-in needs entry to fire repeatedly and may still not hold, so the
bridge must be expensive. The frontier is L-shaped and the joint minimum sits at the
condensation KNEE, not at lock-in saturation -- over-paying for lock-in is wasted poison.

Run:  python3 demo_contagion_joint.py
Writes results/contagion_joint.png and prints the grids + the joint vs pinned totals.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo_utils import result_path
from markov_cache import generate
import contagion as C
import demo_contagion_hijack as H
import demo_contagion_hijack_quine as QH

V = H.V
T_GEN    = 160
N_TRIALS = 26
W_GRID   = [0, 1, 2, 4, 8, 16, 32]      # bridge poison (entry)
R_GRID   = [2, 4, 8, 16, 32, 64, 128]   # payload reps (lock-in / reproduction)
P_LENS   = [1, 2, 3, 5]
TARGET   = 0.5
SEED     = 7


def inject(pred, S, reps, c, w_bridge):
    p = len(S)
    for _ in range(reps):
        for i in range(p):
            pred.observe([S[i]], S[(i + 1) % p])
    for _ in range(w_bridge):
        pred.observe([c], S[0])


def success_prob(gm, D, S, c, w_bridge, reps, prompt, rng, n=N_TRIALS):
    """P(hijack): fraction of trials whose post-entry tail is >= 50% on the quine."""
    alpha = set(S); hits = 0
    for _ in range(n):
        pred = H.make_victim(gm, D)
        inject(pred, S, reps, c, w_bridge)
        gen = generate(pred, T_GEN, rng, prompt=list(prompt))[len(prompt):]
        tail = gen[len(gen) // 2:]
        if np.mean([s in alpha for s in tail]) >= 0.5:
            hits += 1
    return hits / n


def grid_for(gm, D, S, c, prompt, rng):
    P = np.zeros((len(R_GRID), len(W_GRID)))
    for i, r in enumerate(R_GRID):
        for j, w in enumerate(W_GRID):
            P[i, j] = success_prob(gm, D, S, c, w, r, prompt, rng)
    return P


def joint_and_pinned(P, p):
    """From a P(hijack) grid: the joint-minimal total poison over feasible cells, and
    the 2.1-style 'pin lock-in strong' total (reps = max, min feasible bridge)."""
    feas = [(R_GRID[i] * p + W_GRID[j], R_GRID[i], W_GRID[j])
            for i in range(len(R_GRID)) for j in range(len(W_GRID)) if P[i, j] >= TARGET]
    if not feas:
        return None, None
    joint = min(feas)                                  # (total, reps, w_bridge)
    pin_row = P[-1]                                    # reps = max
    wj = next((W_GRID[j] for j in range(len(W_GRID)) if pin_row[j] >= TARGET), None)
    pinned = (R_GRID[-1] * p + wj, R_GRID[-1], wj) if wj is not None else None
    return joint, pinned


def run():
    gm, D, ood = QH.build_with_ood(max(P_LENS))
    prompt = [int(D[0])]
    rng = np.random.default_rng(SEED + 4)
    c, freq = QH.most_visited_bridge(gm, D, prompt, rng, ood)
    print(f"bridge ctx={c} (freq={freq[c]:.3f}); target P(hijack)>={TARGET}\n")

    grids, totals = {}, {}
    for p in P_LENS:
        S = C.rainbow_cycle(ood[:p])
        P = grid_for(gm, D, S, c, prompt, rng)
        grids[p] = P
        joint, pinned = joint_and_pinned(P, p)
        totals[p] = (joint, pinned)
        jt = f"total={joint[0]} (reps={joint[1]}, bridge={joint[2]})" if joint else "infeasible"
        pn = f"total={pinned[0]} (reps={pinned[1]}, bridge={pinned[2]})" if pinned else "n/a"
        print(f"p={p}:  JOINT min {jt}   |   PINNED-strong {pn}")
    return gm, ood, c, grids, totals


def plot(ood, grids, totals):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.9))
    P = grids[1]
    im = ax.imshow(P, origin='lower', aspect='auto', cmap='magma', vmin=0, vmax=1,
                   extent=[-0.5, len(W_GRID) - 0.5, -0.5, len(R_GRID) - 0.5])
    ax.set_xticks(range(len(W_GRID))); ax.set_xticklabels(W_GRID)
    ax.set_yticks(range(len(R_GRID))); ax.set_yticklabels(R_GRID)
    ax.contour(P, levels=[TARGET], colors='cyan', linewidths=2,
               extent=[0, len(W_GRID) - 1, 0, len(R_GRID) - 1])
    joint, pinned = totals[1]
    if joint:
        ax.plot(W_GRID.index(joint[2]), R_GRID.index(joint[1]), '*', color='lime',
                ms=20, mec='k', label=f'joint min (total {joint[0]})')
    if pinned:
        ax.plot(W_GRID.index(pinned[2]), R_GRID.index(pinned[1]), 'P', color='white',
                ms=13, mec='k', label=f'pin-strong (total {pinned[0]})')
    ax.set_xlabel('bridge poison $w_{bridge}$ (entry)')
    ax.set_ylabel('payload reps (lock-in)')
    ax.set_title('P(hijack) for the self-loop (p=1).\n'
                 'Joint min sits at the condensation knee, not saturation')
    ax.legend(fontsize=8, loc='lower right'); fig.colorbar(im, ax=ax, label='P(hijack)')

    ps = [p for p in P_LENS if totals[p][0]]
    jt = [totals[p][0][0] for p in ps]
    pn = [totals[p][1][0] if totals[p][1] else np.nan for p in ps]
    x = np.arange(len(ps)); w = 0.38
    ax2.bar(x - w / 2, jt, w, color='seagreen', label='joint minimum')
    ax2.bar(x + w / 2, pn, w, color='salmon', label='pin lock-in strong (2.1-style)')
    for xi, a, b in zip(x, jt, pn):
        ax2.text(xi - w / 2, a + 2, str(int(a)), ha='center', fontsize=8)
        if not np.isnan(b):
            ax2.text(xi + w / 2, b + 2, str(int(b)), ha='center', fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels([f'p={p}' for p in ps])
    ax2.set_ylabel('minimal total poison (counts)')
    ax2.set_title('Joint optimization vs over-paying for lock-in\n'
                  '(savings grow as the payload term reps$\\times$p shrinks the slack)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis='y')

    fig.tight_layout(); fig.savefig(result_path('contagion_joint.png'), dpi=130)
    print("\nwrote results/contagion_joint.png")


if __name__ == "__main__":
    print("=== Q2.5: joint minimal poison (entry + lock-in together) ===\n")
    gm, ood, c, grids, totals = run()
    plot(ood, grids, totals)
    print("\nDone.")

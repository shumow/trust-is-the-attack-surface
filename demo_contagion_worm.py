"""
Q2 / Milestone 2.4: cross-cache PROPAGATION -- the literal worm, and its R0.

The prior milestones infect ONE cache. This asks whether the contagion TRANSMITS
between hosts. The mechanism makes transmission a corollary of Q1 reproduction:

  An infected host generates output in which the string S appears k times (its viral
  load). A second host that READS that output streams those k copies into its own
  cache -- which is exactly the Q1 payload injection with reps = k -- and the context
  ENDS mid-string, so the receiver starts already inside the quine basin (entry is
  free; no bridge needed for secondary infection). The receiver then reproduces S and
  re-broadcasts k' copies in its own T-token output.

So one passage is a map  k_in -> k_out, and the epidemic is its iteration:
  * below a transmissibility threshold the receiver does not reproduce -> k_out ~ 0
    (the chain dies);
  * above it the receiver reproduces and broadcasts ~ occupancy * T / p copies -- a
    ceiling set by how much it generates (T) divided by the string length (p).

R0 ~ (broadcast ceiling) / (threshold) ~ T / (p * k_thresh), so SHORTER strings are
more contagious: there is a critical length beyond which R0 < 1 and the worm dies out
after the index host. Length trades payload richness against transmissibility.

Run:  python3 demo_contagion_worm.py
Writes results/contagion_worm.png and prints the passage table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from markov_cache import (peaky_transition_matrix, make_corpus, sample_document,
                          BackoffModel, global_dist, generate)
import contagion as C
import demo_contagion_hijack as H

V, G_ALPHA = H.V, H.G_ALPHA
GLOBAL_ORDER = 2
G_CONC, D_CONC, PI_CTX = 1.0, 0.05, 0.6
N_TRAIN, LEN_TRAIN = 200, 400
LEN_D    = 400
T_GEN    = 160          # how much each host generates (its broadcast length)
N_HOSTS  = 12           # receiver hosts averaged per passage (different legit docs)
SEED     = 7


def build_population(n_ood):
    rng = np.random.default_rng(SEED)
    G = peaky_transition_matrix(V, G_CONC, rng)
    train = make_corpus(G, PI_CTX, N_TRAIN, LEN_TRAIN, D_CONC, rng)
    gm0 = BackoffModel(GLOBAL_ORDER, V).fit(train)
    base = global_dist(gm0, [], alpha=G_ALPHA)
    ood = [int(s) for s in np.argsort(base)[:n_ood]]
    keep = int(np.argmax([0 if s in ood else base[s] for s in range(V)]))
    remap = {s: keep for s in ood}
    train = [[remap.get(s, s) for s in doc] for doc in train]
    gm = BackoffModel(GLOBAL_ORDER, V).fit(train)
    return G, gm, ood, remap


def host_doc(G, remap, rng):
    """A fresh legitimate host document (OOD payload symbols sanitized out)."""
    Dmat = peaky_transition_matrix(V, D_CONC, rng)
    D = sample_document(G, Dmat, PI_CTX, LEN_D, rng)
    return [remap.get(int(s), int(s)) for s in D]


def passage(gm, S, k_in, D_host, rng, T=T_GEN):
    """One transmission: a host warmed on D_host READS a context carrying S repeated
    k_in times (streams it), then generates T tokens seeded at the string's end. Return
    the outgoing viral load k_out = number of full S-cycles reproduced in the output."""
    p = len(S)
    if k_in < 1:
        return 0.0
    pred = H.make_victim(gm, D_host)
    hist = []
    for s in list(S) * int(round(k_in)):          # read the contaminated context
        pred.observe(hist, s); hist.append(s)
    gen = generate(pred, T, rng, prompt=[S[-1]])[1:]
    succ = {S[i]: S[(i + 1) % p] for i in range(p)}
    cyc = sum(1 for i in range(1, len(gen))
              if gen[i - 1] in succ and gen[i] == succ[gen[i - 1]])
    return cyc / p


def passage_map(gm, S, G, remap, rng, k_grid):
    return np.array([np.mean([passage(gm, S, k, host_doc(G, remap, rng), rng)
                              for _ in range(N_HOSTS)]) for k in k_grid])


def serial_passage(gm, S, G, remap, rng, k0, n_steps=7):
    traj = [float(k0)]
    for _ in range(n_steps):
        k = traj[-1]
        traj.append(float(np.mean([passage(gm, S, k, host_doc(G, remap, rng), rng)
                                    for _ in range(N_HOSTS)])))
    return traj


def run():
    G, gm, ood, remap = build_population(n_ood=8)
    rng = np.random.default_rng(SEED + 4)
    k_grid = [1, 2, 4, 8, 16, 32, 64, 128]

    print("=== Passage map  k_in -> k_out  (viral load per transmission) ===")
    maps = {}
    for p in (1, 3, 6):
        S = C.rainbow_cycle(ood[:p])
        maps[p] = passage_map(gm, S, G, remap, rng, k_grid)
        print(f"\np={p}: " + "  ".join(f"{k}->{o:.0f}" for k, o in zip(k_grid, maps[p])))

    print("\n=== Serial passage from a strong index host (k0=48) ===")
    serial = {}
    for p in (1, 2, 3, 5, 8):
        S = C.rainbow_cycle(ood[:p])
        serial[p] = serial_passage(gm, S, G, remap, rng, k0=48)
        verdict = "SUSTAINS" if serial[p][-1] >= 1 else "DIES OUT"
        print(f"p={p}: " + " -> ".join(f"{k:.0f}" for k in serial[p]) + f"   [{verdict}]")
    return k_grid, maps, serial


def plot(k_grid, maps, serial):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {1: 'crimson', 3: 'seagreen', 6: 'navy'}
    kk = np.array(k_grid, float)
    for p, ys in maps.items():
        ax.plot(kk, ys, 'o-', color=colors[p], lw=1.9, label=f'string length p={p}')
    ax.plot(kk, kk, 'k--', lw=1.0, alpha=0.6, label='$k_{out}=k_{in}$ (replacement)')
    ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
    ax.set_xlabel('incoming viral load $k_{in}$ (copies read)')
    ax.set_ylabel('outgoing viral load $k_{out}$ (copies broadcast)')
    ax.set_title('Passage map: above the diagonal $\\Rightarrow$ load grows.\n'
                 'Shorter strings broadcast more copies per host ($\\propto T/p$)')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3, which='both')

    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(serial)))
    for (p, traj), col in zip(serial.items(), cmap):
        ax2.plot(range(len(traj)), np.clip(traj, 0.3, None), 'o-', color=col, lw=1.9,
                 label=f'p={p}  ({"sustains" if traj[-1] >= 1 else "dies"})')
    ax2.axhline(1.0, color='k', lw=0.8, ls=':', label='extinction (load<1)')
    ax2.set_yscale('log'); ax2.set_xlabel('transmission passage number')
    ax2.set_ylabel('viral load (S-copies per host output)')
    ax2.set_title('Serial passage: short strings reach an endemic load,\n'
                  'long strings die out after the index host (R0<1)')
    ax2.legend(fontsize=8, loc='center right'); ax2.grid(alpha=0.3, which='both')

    fig.tight_layout(); fig.savefig('results/contagion_worm.png', dpi=130)
    print("\nwrote results/contagion_worm.png")


if __name__ == "__main__":
    print("=== Q2.4: cross-cache propagation (the worm) ===\n")
    k_grid, maps, serial = run()
    plot(k_grid, maps, serial)
    print("\nDone.")

"""
Q2 / Milestone 2.3: hijacking a warmed cache into a multi-symbol QUINE -- installing a
contagious STRING, not just a stuck token. Closes the loop between the two halves.

Victim = an order-1 cache warmed on a legitimate document D. Payload = a rainbow quine
S = x0->x1->...->x0 over a RESERVED out-of-distribution alphabet (sanitized out of corpus
and D, so the quine's contexts are fresh -- count = reps -- and reproduction is the clean
Q1 dynamic). The hijack is the sum of the two prior milestones:

  ENTRY  (Q2):  a bridge edge c -> x0 injected at a high-visit context c, so generation
                riding D's trajectory crosses into the quine at its entry point x0.
  PAYLOAD (Q1): the cycle S streamed `reps` times into the cache, so once entered each
                quine context carries reliance reps/(reps+a) -- and reproduction obeys the
                Q1 per-step law, brittleness ladder (longer p needs more reps), and all.

So the total poison decomposes: a length-independent ENTRY cost (the bridge) plus a
PAYLOAD cost that grows with string length p (Q1 brittleness). We sweep reps for several
quine lengths p and measure quine takeover (occupancy of the OOD cycle in generation),
then report the minimal total poison to install a contagious string of each length.

Run:  python3 demo_contagion_hijack_quine.py
Writes results/contagion_hijack_quine.png and prints the table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo_utils import result_path
from markov_cache import (peaky_transition_matrix, make_corpus, sample_document,
                          BackoffModel, global_dist, generate)
import contagion as C
import demo_contagion_hijack as H   # make_victim, local_count, constants

V, A, G_ALPHA = H.V, H.DIR_A, H.G_ALPHA
GLOBAL_ORDER  = 2
G_CONC, D_CONC, PI_CTX = 1.0, 0.05, 0.6
N_TRAIN, LEN_TRAIN = 200, 400
LEN_D     = 600
W_BRIDGE  = 16          # strong-enough entry (from 2.2) so reproduction is the variable
GEN_LEN   = 200
N_TRIALS  = 40
P_LENS    = [1, 2, 3, 5]
REPS_GRID = [4, 8, 16, 32, 64, 128]
SEED      = 7


def build_with_ood(n_ood):
    """Global + victim document D, with `n_ood` rarest symbols RESERVED as the payload
    alphabet and sanitized out of corpus and D (tokens the legitimate system never
    emits). Returns (gm, D, ood_symbols)."""
    rng = np.random.default_rng(SEED)
    G = peaky_transition_matrix(V, G_CONC, rng)
    train = make_corpus(G, PI_CTX, N_TRAIN, LEN_TRAIN, D_CONC, rng)
    Dmat = peaky_transition_matrix(V, D_CONC, rng)
    D = [int(s) for s in sample_document(G, Dmat, PI_CTX, LEN_D, rng)]
    gm0 = BackoffModel(GLOBAL_ORDER, V).fit(train)
    base = global_dist(gm0, [], alpha=G_ALPHA)
    ood = [int(s) for s in np.argsort(base)[:n_ood]]          # rarest -> reserved payload
    keep = int(np.argmax([0 if s in ood else base[s] for s in range(V)]))
    remap = {s: keep for s in ood}
    train = [[remap.get(s, s) for s in doc] for doc in train]
    D = [remap.get(s, s) for s in D]
    gm = BackoffModel(GLOBAL_ORDER, V).fit(train)
    return gm, D, ood


def most_visited_bridge(gm, D, prompt, rng, ood):
    """The cheap optimal entry context (2.2): most-visited non-payload context."""
    freq = np.zeros(V)
    for _ in range(30):
        pred = H.make_victim(gm, D)
        for s in generate(pred, GEN_LEN, rng, prompt=list(prompt))[len(prompt):]:
            freq[int(s)] += 1
    for s in ood:
        freq[s] = -1
    return int(np.argmax(freq)), freq / max(freq.sum(), 1)


def inject_quine_hijack(pred, S, reps, bridge_c, w_bridge):
    """Stamp the payload quine S (reps copies of the cycle) + the bridge c->x0 into the
    warmed victim's cache."""
    p = len(S)
    for _ in range(reps):
        for i in range(p):
            pred.observe([S[i]], S[(i + 1) % p])
    for _ in range(w_bridge):
        pred.observe([bridge_c], S[0])


def quine_takeover(gm, D, S, reps, bridge_c, prompt, rng, n=N_TRIALS):
    """Hijack the warmed victim and measure, on the POST-entry tail of generation:
      occupancy   -- fraction of tail in the quine alphabet (takeover), and
      cyc_fidelity-- fraction of tail transitions obeying S's successor rule (confirms
                     the STRING is reproduced in order, not just OOD symbols visited).
    Tail = second half, so entry latency does not dilute the reproduction signal."""
    alpha = set(S)
    succ = {S[i]: S[(i + 1) % len(S)] for i in range(len(S))}
    occ, fid = [], []
    for _ in range(n):
        pred = H.make_victim(gm, D)
        inject_quine_hijack(pred, S, reps, bridge_c, W_BRIDGE)
        gen = generate(pred, GEN_LEN, rng, prompt=list(prompt))[len(prompt):]
        h = len(gen) // 2
        tail = gen[h:]
        occ.append(float(np.mean([s in alpha for s in tail])))
        correct, scored = 0, 0
        for i in range(max(1, h), len(gen)):
            if gen[i - 1] in succ:
                scored += 1; correct += (gen[i] == succ[gen[i - 1]])
        fid.append(correct / scored if scored else 0.0)
    return float(np.mean(occ)), float(np.mean(fid))


def run():
    gm, D, ood = build_with_ood(max(P_LENS))
    prompt = [int(D[0])]
    rng = np.random.default_rng(SEED + 4)
    bridge_c, freq = most_visited_bridge(gm, D, prompt, rng, ood)
    print(f"OOD payload alphabet={ood};  bridge ctx={bridge_c} "
          f"(visit_freq={freq[bridge_c]:.4f});  W_BRIDGE={W_BRIDGE}\n")

    curves, min_total = {}, {}
    for p in P_LENS:
        S = C.rainbow_cycle(ood[:p])
        res = [quine_takeover(gm, D, S, r, bridge_c, prompt, rng) for r in REPS_GRID]
        ys = [o for o, f in res]
        curves[p] = ys
        rmin = next((r for r, (o, f) in zip(REPS_GRID, res) if o >= 0.5), None)
        min_total[p] = (rmin, (W_BRIDGE + rmin * p) if rmin else None)
        print(f"p={p}  quine={S}")
        for r, (o, f) in zip(REPS_GRID, res):
            print(f"   reps={r:3d}  takeover={o:.3f}  cycle_fidelity={f:.3f}")
        rt = min_total[p]
        print(f"   -> min reps for takeover>=0.5: {rt[0]};  "
              f"min total poison (bridge+payload) = {rt[1]}\n")
    return ood, bridge_c, freq, curves, min_total


def plot(curves, min_total):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(P_LENS)))
    for p, col in zip(P_LENS, colors):
        ax.plot(REPS_GRID, curves[p], 'o-', color=col, lw=1.9,
                label=f'string length p={p}')
    ax.axhline(0.5, color='k', lw=0.6, ls=':')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('payload strength (reps of the cycle injected)')
    ax.set_ylabel('quine takeover (OOD-alphabet occupancy, tail)')
    ax.set_title('Hijack installs a contagious STRING, not just a token\n'
                 '(tail cycle-fidelity ~0.9: the string reproduces in order)')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3, which='both')
    ax.set_ylim(-0.02, 1.02)

    ps = [p for p in P_LENS if min_total[p][1] is not None]
    tot = [min_total[p][1] for p in ps]
    pay = [min_total[p][0] * p for p in ps]
    ax2.bar(ps, pay, color='steelblue', label='payload counts (reps$\\times$p)')
    ax2.bar(ps, [W_BRIDGE] * len(ps), bottom=pay, color='salmon',
            label=f'entry counts (bridge={W_BRIDGE})')
    for p, t in zip(ps, tot):
        ax2.text(p, t + 1, str(int(t)), ha='center', fontsize=9)
    ax2.set_xlabel('contagious string length p'); ax2.set_ylabel('minimal total poison (counts)')
    ax2.set_title('Total poison = length-independent ENTRY\n+ length-growing PAYLOAD')
    ax2.set_xticks(ps); ax2.legend(fontsize=8, loc='upper left'); ax2.grid(alpha=0.3, axis='y')

    fig.tight_layout(); fig.savefig(result_path('contagion_hijack_quine.png'), dpi=130)
    print("wrote results/contagion_hijack_quine.png")


if __name__ == "__main__":
    print("=== Q2.3: hijack a warmed cache into a contagious string ===\n")
    ood, bridge_c, freq, curves, min_total = run()
    plot(curves, min_total)
    print("\nDone.")

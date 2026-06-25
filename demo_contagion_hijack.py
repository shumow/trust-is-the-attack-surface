"""
Q2 / Milestone 2.1: infecting a PRE-TRAINED cache (the delivery half).

Setup: a victim cache already warmed on a legitimate document D (over a fixed global
model). Generation rides D's local structure. The attacker wants to hijack it into
emitting the contagious payload x* (a self-loop x*->x*, the condensation primitive).

The hijack factorizes into ENTRY x LOCK-IN:
  * LOCK-IN: inject the self-loop x*->x* strongly enough that once x* is emitted, the
    Polya feedback captures generation (the condensation knee from the parent repo).
    Fixed here at high strength so the spotlight is on entry.
  * ENTRY: generation only reaches x* if some context it actually VISITS routes there.
    The attacker injects a 'bridge' edge c->x* at one context c. The right c maximizes
        entry_score(c) = visit_freq(c) * leverage(c),   leverage(c) = 1/(n_c + a),
    i.e. a context both ON D's trajectory and CHEAP to poison. These trade off: common
    contexts are visited often but have high warmed count n_c (low leverage); the sweet
    spot is a 'bridge' -- visited during generation yet under-warmed by this particular
    D. This is the weak-key-on-the-path.

Predicted hijack probability:
    P(hijack) ~ [1 - (1 - freq(c)*p(x*|c))^T] * P(lock-in),
with p(x*|c) = (w_bridge + a*pg) / (n_c + w_bridge + a) after injecting w_bridge counts.

Experiments:
  A  minimal poison: at w_bridge = 1 (a single injected bridge count), which entry
     strategy hijacks? best-bridge vs most-visited vs lowest-count vs random.
  B  takeover vs bridge budget, per strategy, against the closed-form entry curve.

Run:  python3 demo_contagion_hijack.py
Writes results/contagion_hijack.png and prints the per-strategy table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from markov_cache import (peaky_transition_matrix, make_corpus, sample_document,
                          BackoffModel, global_dist, make_predictor)

# --------------------------------------------------------------------------- #
V            = 32
G_CONC       = 1.0
D_CONC       = 0.05
PI_CTX       = 0.6         # D carries real local structure for the cache to warm on
GLOBAL_ORDER = 2
G_ALPHA      = 0.1
DIR_A        = 1.0
N_TRAIN, LEN_TRAIN = 200, 400
LEN_D        = 400         # the victim's legitimate document
W_LOOP       = 300         # self-loop strength (reliance ~0.997): full lock-in once entered
T_ENTRY      = 60          # window over which we score entry (short, so the leak doesn't saturate)
N_TRIALS     = 80
W_BRIDGE_GRID = [0, 1, 2, 3, 5, 8, 13, 21, 34]
SEED         = 7


def build_global_and_doc():
    """Build the global + a legitimate victim document, then SANITIZE the payload
    token x* out of both -- x* is an out-of-distribution token the legitimate system
    never emits (a trigger/marker), so spontaneous leak is just the smoothing floor and
    every real x* count is the attacker's. Returns (gm, D, x_star)."""
    rng = np.random.default_rng(SEED)
    G = peaky_transition_matrix(V, G_CONC, rng)
    train = make_corpus(G, PI_CTX, N_TRAIN, LEN_TRAIN, D_CONC, rng)
    Dmat = peaky_transition_matrix(V, D_CONC, rng)
    D = list(int(s) for s in sample_document(G, Dmat, PI_CTX, LEN_D, rng))
    gm0 = BackoffModel(GLOBAL_ORDER, V).fit(train)
    x_star = int(np.argmin(global_dist(gm0, [], alpha=G_ALPHA)))   # globally rarest
    repl = (x_star + 1) % V
    train = [[repl if s == x_star else int(s) for s in doc] for doc in train]
    D = [repl if s == x_star else s for s in D]
    gm = BackoffModel(GLOBAL_ORDER, V).fit(train)
    return gm, D, x_star


def make_victim(gm, D):
    """A fresh predictor warmed on D -- the pre-trained victim cache."""
    pred = make_predictor(gm, approach='dirichlet', cache_order=1,
                          weight=DIR_A, g_alpha=G_ALPHA)
    pred.reset()
    hist = []
    for s in D:
        pred.observe(hist, s)
        hist.append(int(s))
    return pred


def local_count(pred, ctx):
    """Warmed cache count n_c at context ctx, via reliance n/(n+a)."""
    rel = pred.local_reliance(list(ctx))
    return DIR_A * rel / (1.0 - rel) if rel < 1.0 else np.inf


def visit_frequencies(gm, D, prompt, rng, n=24):
    """Generation-time visit distribution over contexts on the CLEAN victim (the
    trajectory entry must exploit). Re-warm each trial since generate() mutates."""
    freq = np.zeros(V)
    from markov_cache import generate
    for _ in range(n):
        pred = make_victim(gm, D)
        out = generate(pred, T_ENTRY, rng, prompt=list(prompt))
        for s in out:
            freq[int(s)] += 1
    return freq / freq.sum()


def entry_prob(gm, D, bridge_ctx, x_star, w_bridge, prompt, rng, n=N_TRIALS):
    """P(generation first emits x* within T_ENTRY steps), with ONLY the bridge edge
    c->x* injected (no self-loop) -- the pure ENTRY event, decoupled from lock-in.
    Rebuild+reinject per trial so trials are independent."""
    from markov_cache import generate
    hits = 0
    for _ in range(n):
        pred = make_victim(gm, D)
        for _ in range(w_bridge):
            pred.observe(list(bridge_ctx), x_star)
        out = generate(pred, T_ENTRY, rng, prompt=list(prompt))
        if x_star in out[len(prompt):]:
            hits += 1
    return hits / n


def takeover(gm, D, bridge_ctx, x_star, w_bridge, prompt, rng, n=N_TRIALS):
    """Full hijack: bridge (entry) + strong self-loop (lock-in). Mean x* occupancy --
    confirms entry -> takeover once lock-in is past the condensation knee."""
    from markov_cache import generate
    occ = []
    for _ in range(n):
        pred = make_victim(gm, D)
        for _ in range(W_LOOP):
            pred.observe([x_star], x_star)
        for _ in range(w_bridge):
            pred.observe(list(bridge_ctx), x_star)
        out = generate(pred, T_ENTRY, rng, prompt=list(prompt))
        gen = out[len(prompt):]
        occ.append(float(np.mean(np.array(gen) == x_star)))
    return float(np.mean(occ))


def choose_bridges(gm, D, x_star, freq):
    """White-box entry-strategy selection from the victim's warmed counts + the
    generation visit frequencies."""
    victim = make_victim(gm, D)
    n_c = np.array([local_count(victim, [s]) for s in range(V)])
    lev = 1.0 / (n_c + DIR_A)
    score = freq * lev
    visited = (freq > 1.0 / (V * T_ENTRY))      # plausibly on-trajectory
    score[x_star] = -1; freq_m = freq.copy(); freq_m[x_star] = -1
    nc_m = n_c.copy(); nc_m[~visited] = np.inf; nc_m[x_star] = np.inf
    rng = np.random.default_rng(SEED + 99)
    cand = [s for s in range(V) if visited[s] and s != x_star]
    strategies = {
        'best bridge (freq/(n+a))': int(np.argmax(score)),
        'most visited':             int(np.argmax(freq_m)),
        'lowest count (visited)':   int(np.argmin(nc_m)),
        'random visited':           int(rng.choice(cand)),
    }
    return strategies, n_c, freq


def run():
    gm, D, x_star = build_global_and_doc()
    base = global_dist(gm, [], alpha=G_ALPHA)
    prompt = [int(D[0])]
    rng = np.random.default_rng(SEED + 4)

    freq = visit_frequencies(gm, D, prompt, rng)
    strategies, n_c, freq = choose_bridges(gm, D, x_star, freq)
    print(f"x* = {x_star} (sanitized out; global floor p={base[x_star]:.4f});  "
          f"W_LOOP={W_LOOP}\n")
    for name, c in strategies.items():
        print(f"  {name:28s}: ctx={c:2d}  visit_freq={freq[c]:.4f}  "
              f"n_c={n_c[c]:6.1f}  leverage={1/(n_c[c]+DIR_A):.4f}")

    entry = {}
    for name, c in strategies.items():
        ps = [entry_prob(gm, D, [c], x_star, w, prompt, rng) for w in W_BRIDGE_GRID]
        entry[name] = ps
        # minimal poison to make entry likely (P_enter >= 0.5)
        wmin = next((w for w, p in zip(W_BRIDGE_GRID, ps) if p >= 0.5), None)
        print(f"\n{name} (ctx {c}, freq={freq[c]:.4f}, n_c={n_c[c]:.0f}):  "
              f"min poison for P_enter>=0.5: {wmin}")
        for w, p in zip(W_BRIDGE_GRID, ps):
            print(f"  w_bridge={w:3d}  P_enter={p:.3f}")
    # confirm entry -> takeover with lock-in, for the most-visited bridge at w=8
    cmv = strategies['most visited']
    tk = takeover(gm, D, [cmv], x_star, 8, prompt, rng)
    print(f"\nlock-in check: most-visited bridge w=8 -> takeover occupancy {tk:.3f}")
    return gm, D, x_star, strategies, n_c, freq, entry


def entry_rate(p, T):
    p = min(p, 1 - 1e-6)
    return -np.log(1 - p) / T


def plot(strategies, n_c, freq, entry):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    colors = {'best bridge (freq/(n+a))': 'crimson', 'most visited': 'navy',
              'lowest count (visited)': 'seagreen', 'random visited': 'gray'}
    w = np.array(W_BRIDGE_GRID, float)
    leak = entry_rate(entry['most visited'][0], T_ENTRY)   # w=0 baseline ~ global leak
    for name, ps in entry.items():
        c = strategies[name]
        rate = np.array([entry_rate(p, T_ENTRY) for p in ps])
        ax.plot(w, rate, 'o-', color=colors[name], lw=1.9,
                label=f'{name}: ctx {c}, freq={freq[c]:.3f}, $n_c$={n_c[c]:.0f}')
        # theory: leak + freq(c) * w/(n_c+w+a)
        th = leak + freq[c] * w / (n_c[c] + w + DIR_A)
        ax.plot(w, th, '--', color=colors[name], lw=1.0, alpha=0.55)
    ax.axhline(leak, color='k', lw=0.8, ls=':', label='spontaneous leak (w=0)')
    ax.set_xlabel('bridge poison $w_{bridge}$ (counts injected at $c\\to x^*$)')
    ax.set_ylabel('entry rate  $-\\ln(1-P_{enter})/T$  (per step)')
    ax.set_title('Q2 entry rate $=\\mathrm{freq}(c)\\cdot p(x^*|c,w)$ (dashed = theory).\n'
                 'Low poison: best $=\\max\\,\\mathrm{freq}/(n_c{+}a)$ (leverage);  '
                 'high poison: best $=\\max\\,\\mathrm{freq}$ (the rate ceiling)')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('results/contagion_hijack.png', dpi=130)
    print("\nwrote results/contagion_hijack.png")


if __name__ == "__main__":
    print("=== Q2: infecting a pre-trained cache (entry x lock-in) ===\n")
    gm, D, x_star, strategies, n_c, freq, entry = run()
    plot(strategies, n_c, freq, entry)
    print("\nDone.")

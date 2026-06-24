"""
Stage-4 driver: the *conservation law* between context usefulness and context
exploitability.

Thesis (the "neural chytrid", made measurable on this architecture)
-------------------------------------------------------------------
The cache helps a prediction exactly insofar as the output depends on the cache's
contents; a poison injected into the cache hurts the output by the *same*
dependence. So in-context usefulness and in-context poisonability are two readings
of one quantity -- how much realized behaviour leans on the cache -- and the knob
`pi_ctx` (fraction of the source driven by per-document structure the global model
provably cannot have) dials that quantity directly.

What this script measures, swept over pi_ctx
---------------------------------------------
  benefit            usefulness: bits/symbol saved by the cache vs global-only.
  reliance           raw Dirichlet weight on the cache, n/(n+a), warmed up.
  static_leverage    CONTROL: prob-mass an attacker buys with ONE injected count
                     at the live context, single query. Predicted ~flat in pi_ctx
                     (a thin context is always pollutable; the point is that a
                     one-shot poison that does not propagate is cheap everywhere).
  propagated_lev     the real exploitability: a single injected count at one
                     context, then the cache keeps streaming the true document;
                     we accumulate the extra probability the model places on the
                     poison target at every later step. This is the lattice
                     Green's-function influence -- it grows when the poisoned
                     context RECURS and is TRUSTED, i.e. with pi_ctx.

Headline (what the run actually shows)
--------------------------------------
The naive law "usefulness == exploitability" is FALSE as stated, and the script
is built to expose that honestly. Exploitability tracks cache TRUST (reliance),
not usefulness -- and trust and usefulness only coincide when contexts are SPARSE
enough that a context recurring is itself evidence of real structure:

  * order-1 cache (dense, contexts always fill): trust saturates (~0.9) at every
    pi_ctx, so propagated leverage stays high even where the cache HURTS by >1 bit.
    corr(benefit, propagated) ~ 0.5. The model is fully poisonable precisely where
    its context is useless -- the alarming, decoupled regime.

  * order-3 cache (sparse, the induction/PPM regime): trust is EARNED from genuine
    recurrence, so reliance, usefulness, and propagated leverage rise together.
    corr(benefit, propagated) ~ 0.97. Here the conservation law holds.

So the law is conditional: it holds in the earned-trust regime that real high-order
/ natural-language contexts live in (and that the repo already saw as reliance ~0.01
on real text), and breaks in the saturated-trust regime. The static 1-shot control
is ANTI-correlated with usefulness in both (~ -0.95): raw per-cell pollutability
actually falls as the cache fills, confirming it is a red herring -- only PROPAGATED
leverage is the right exploitability object.

Figure 2: condensation. At fixed high pi_ctx, sweep the cache weight (-> reliance)
and seed a single self-looping poison; generation either shrugs it off or locks
onto it (rich-get-richer). The takeover curve is the Polya-urn phase transition --
the model "sporulating" on the zoospore once it trusts context enough.

Run:  python3 demo_conservation.py
Writes results/conservation_law.png and results/condensation.png and prints a table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from markov_cache import (
    peaky_transition_matrix, make_corpus, sample_document,
    BackoffModel, MarkovCounts, global_dist, combine_dirichlet,
    make_predictor, evaluate, evaluate_predictor, generate,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
V             = 32        # vocab
G_CONC        = 1.0       # HIGH-entropy global: at pi=0 documents are unstructured,
                          # so a context recurring is NOT automatic -> trust can vary
D_CONC        = 0.05      # peaky per-document law: real, copyable local structure
GLOBAL_ORDER  = 2         # BackoffModel max order (the "weights")
CACHE_ORDERS  = [1, 3]    # dense (saturated-trust) vs sparse (earned-trust) cache
DIR_A         = 1.0       # Dirichlet prior strength (global-as-prior weight)
G_ALPHA       = 0.1
N_TRAIN, LEN_TRAIN = 200, 400
N_TEST,  LEN_TEST  = 30,  400
PI_GRID       = np.round(np.linspace(0.0, 0.9, 10), 3)
SEED          = 7

# --------------------------------------------------------------------------- #
# Exploitability measures
# --------------------------------------------------------------------------- #
def static_leverage(test_docs, global_model, *, cache_order, a, g_alpha):
    """CONTROL. Mean prob-mass an attacker gains on the (globally) rarest symbol by
    injecting ONE count at the current live context -- a single, non-propagating
    poison. Prequential: score-then-observe, exactly like evaluate()."""
    V = global_model.V
    swings = []
    for seq in test_docs:
        cache = MarkovCounts(cache_order, V)
        for t, sym in enumerate(seq):
            hist = seq[:t]
            pg = global_dist(global_model, hist, alpha=g_alpha)
            p = combine_dirichlet(pg, cache, hist, a)
            x_star = int(np.argmin(pg))            # target an attacker forces
            ctx = cache._ctx(hist)
            cache.counts[ctx][x_star] += 1.0       # inject one zoospore
            p_pois = combine_dirichlet(pg, cache, hist, a)
            cache.counts[ctx][x_star] -= 1.0       # undo
            swings.append(float(p_pois[x_star] - p[x_star]))
            cache.observe(hist, sym)
    return float(np.mean(swings))

def propagated_leverage(test_docs, global_model, *, cache_order, a, g_alpha):
    """The real thing. Inject ONE count at one context a quarter of the way in,
    then let the cache keep streaming the TRUE document. Accumulate the extra
    probability placed on the poison target at every later step. Equals the
    Green's-function influence of a unit perturbation through the (history-
    dependent) cache operator: large when the poisoned context recurs and is
    trusted. Returns mean over docs of total extra target-mass."""
    V = global_model.V
    influences = []
    for seq in test_docs:
        L = len(seq)
        t0 = L // 4
        clean = MarkovCounts(cache_order, V)
        for t in range(t0):
            clean.observe(seq[:t], seq[t])
        hist0 = seq[:t0]
        s0 = clean._ctx(hist0)
        pg0 = global_dist(global_model, hist0, alpha=g_alpha)
        x_star = int(np.argmin(pg0))
        pois = MarkovCounts(cache_order, V)
        for k, val in clean.counts.items():
            pois.counts[k] = val.copy()
        pois.counts[s0][x_star] += 1.0             # the single injected count
        infl = 0.0
        for t in range(t0, L):
            hist = seq[:t]
            pg = global_dist(global_model, hist, alpha=g_alpha)
            p_clean = combine_dirichlet(pg, clean, hist, a)
            p_pois  = combine_dirichlet(pg, pois,  hist, a)
            infl += float(p_pois[x_star] - p_clean[x_star])
            clean.observe(hist, seq[t]); pois.observe(hist, seq[t])
        influences.append(infl)
    return float(np.mean(influences))

# --------------------------------------------------------------------------- #
# Main sweep: usefulness vs exploitability over pi_ctx
# --------------------------------------------------------------------------- #
def run_sweep():
    """For each cache order, sweep pi_ctx and record usefulness + exploitability.
    A high-entropy global (G_CONC) means that at pi=0 documents are unstructured,
    so cache trust is *earned* from genuine recurrence rather than handed out by
    the small context space -- this is what lets trust (and the law) vary."""
    rng = np.random.default_rng(SEED)
    G = peaky_transition_matrix(V, G_CONC, rng)          # high-entropy shared law
    results = {}
    for order in CACHE_ORDERS:
        rows = []
        for pi in PI_GRID:
            train = make_corpus(G, pi, N_TRAIN, LEN_TRAIN, D_CONC, rng)
            test  = make_corpus(G, pi, N_TEST,  LEN_TEST,  D_CONC, rng)
            gm = BackoffModel(GLOBAL_ORDER, V).fit(train)

            bits_glob, _ = evaluate(test, gm, cache_order=order,
                                    method='global', weight=0.0, g_alpha=G_ALPHA)
            bits_cache, _ = evaluate(test, gm, cache_order=order,
                                     method='dirichlet', weight=DIR_A, g_alpha=G_ALPHA)
            benefit = bits_glob - bits_cache

            _, _, rel = evaluate_predictor(
                test, make_predictor(gm, approach='dirichlet',
                                     cache_order=order, weight=DIR_A, g_alpha=G_ALPHA),
                reliance=True)
            reliance = float(rel[-1])

            stat = static_leverage(test, gm, cache_order=order, a=DIR_A, g_alpha=G_ALPHA)
            prop = propagated_leverage(test, gm, cache_order=order, a=DIR_A, g_alpha=G_ALPHA)
            rows.append((pi, benefit, reliance, stat, prop))
            print(f"order={order}  pi={pi:4.2f}  benefit={benefit:+.4f} b  "
                  f"reliance={reliance:.3f}  static_lev={stat:.4f}  prop_lev={prop:.3f}")
        results[order] = np.array(rows)
    return results

# --------------------------------------------------------------------------- #
# Condensation: the Polya-urn phase transition in generation
# --------------------------------------------------------------------------- #
def run_condensation(pi=0.8, weights=(8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.12, 0.06),
                     gen_len=300, seed_k=3, n_trials=12):
    """At fixed pi, seed a single self-looping poison (x* -> x*) and generate.
    Sweep the Dirichlet weight a (small a => cache dominates => high reliance).
    Report the generated frequency of x* (poisoned) minus baseline (unseeded)."""
    rng = np.random.default_rng(SEED + 1)
    G = peaky_transition_matrix(V, G_CONC, rng)
    train = make_corpus(G, pi, N_TRAIN, LEN_TRAIN, D_CONC, rng)
    gm = BackoffModel(GLOBAL_ORDER, V).fit(train)
    # poison target = a symbol the global almost never emits, so any takeover is the poison
    base_unigram = global_dist(gm, [], alpha=G_ALPHA)
    x_star = int(np.argmin(base_unigram))

    rel_axis, takeover = [], []
    for a in weights:
        seeded_freqs, base_freqs, rels = [], [], []
        for _ in range(n_trials):
            # baseline: no seed
            pred = make_predictor(gm, approach='dirichlet', cache_order=1,
                                  weight=a, g_alpha=G_ALPHA)
            gen = generate(pred, gen_len, rng, prompt=[x_star])
            base_freqs.append(np.mean(np.array(gen) == x_star))
            # poisoned: seed a self-loop x* -> x* a few times
            pred.reset()
            for _ in range(seed_k):
                pred.observe([x_star], x_star)
            gen = generate(pred, gen_len, rng, prompt=[x_star])
            arr = np.array(gen)
            seeded_freqs.append(np.mean(arr == x_star))
            rels.append(pred.local_reliance([x_star]))
        rel_axis.append(float(np.mean(rels)))
        takeover.append(float(np.mean(seeded_freqs) - np.mean(base_freqs)))
        print(f"a={a:5.2f}  reliance(x*)={rel_axis[-1]:.3f}  "
              f"poison_takeover={takeover[-1]:+.3f}")
    return np.array(rel_axis), np.array(takeover)

# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_sweep(results):
    def nz(x):
        x = x - x.min()
        return x / x.max() if x.max() > 0 else x
    orders = sorted(results)
    fig, axes = plt.subplots(1, len(orders) + 1, figsize=(6 * (len(orders)) + 4, 4.6))
    corrs = {}
    for ax, order in zip(axes[:-1], orders):
        pi, benefit, reliance, stat, prop = results[order].T
        r_bp = np.corrcoef(benefit, prop)[0, 1]
        r_br = np.corrcoef(benefit, reliance)[0, 1]
        r_bs = np.corrcoef(benefit, stat)[0, 1]
        corrs[order] = (r_bp, r_br, r_bs)
        ax.plot(pi, nz(benefit),  'o-',  label='usefulness (bits saved)', lw=2)
        ax.plot(pi, nz(prop),     's-',  label='propagated poison leverage', lw=2)
        ax.plot(pi, nz(reliance), 'x:',  label='cache reliance n/(n+a)', lw=1.3, alpha=0.7)
        ax.plot(pi, nz(stat),     '^--', label='static 1-shot (control)', lw=1.3, alpha=0.6)
        regime = "dense / trust saturated" if order == 1 else "sparse / trust earned"
        ax.set_title(f'cache order {order}  ({regime})\n'
                     f'corr(benefit, propagated) = {r_bp:.2f}')
        ax.set_xlabel(r'$\pi_{\mathrm{ctx}}$'); ax.set_ylabel('normalized [0,1]')
        ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    # summary bar panel
    axb = axes[-1]
    x = np.arange(len(orders)); w = 0.26
    axb.bar(x - w, [corrs[o][0] for o in orders], w, label='corr(benefit, propagated)')
    axb.bar(x,     [corrs[o][1] for o in orders], w, label='corr(benefit, reliance)')
    axb.bar(x + w, [corrs[o][2] for o in orders], w, label='corr(benefit, static control)')
    axb.set_xticks(x); axb.set_xticklabels([f'order {o}' for o in orders])
    axb.axhline(0, color='k', lw=0.6); axb.set_ylim(-1, 1.05)
    axb.set_ylabel('Pearson r across the $\\pi_{ctx}$ sweep')
    axb.set_title('Where the conservation law holds')
    axb.legend(fontsize=8); axb.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig('results/conservation_law.png', dpi=130)
    print("\nwrote results/conservation_law.png")
    for o in orders:
        print(f"  order {o}: corr(benefit,propagated)={corrs[o][0]:+.3f}  "
              f"corr(benefit,reliance)={corrs[o][1]:+.3f}  "
              f"corr(benefit,static)={corrs[o][2]:+.3f}")
    return corrs

def plot_condensation(rel_axis, takeover):
    order = np.argsort(rel_axis)
    rel_axis, takeover = rel_axis[order], takeover[order]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(rel_axis, takeover, 'o-', lw=2, color='crimson')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('cache reliance on the poisoned context  (warmed up)')
    ax.set_ylabel('poison takeover of generation\n(seeded freq − baseline freq)')
    ax.set_title('Condensation: one seeded self-loop locks in\n'
                 'once context is trusted (Pólya-urn transition)')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig('results/condensation.png', dpi=130)
    print("wrote results/condensation.png")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== Conservation-law sweep over pi_ctx (per cache order) ===")
    results = run_sweep()
    corrs = plot_sweep(results)
    print("\n=== Condensation sweep (fixed pi, varying trust) ===")
    rel_axis, takeover = run_condensation()
    plot_condensation(rel_axis, takeover)
    print("\nDone.")

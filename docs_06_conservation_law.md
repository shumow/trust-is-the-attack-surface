# Context as a poisonable membrane: a cryptanalytic reading of the cache, with measurements

*A write-up of where the "parametric vs. context memory" split meets differential
cryptanalysis, worked on this repo's global+cache Markov architecture, and tested
with `demo_conservation.py`.*

This note does two things. First it lays out a dictionary that reads the
global+cache predictor as a cipher and the act of context-poisoning as a
differential attack. Then it states a **conservation law** — context usefulness and
context exploitability are two readings of one quantity — and reports that the law,
once actually measured, is **conditional**: it holds in the sparse / earned-trust
regime (the induction/PPM regime that real high-order contexts live in) and breaks
in the dense / saturated-trust regime, where the model stays fully poisonable even
where its context is useless.

---

## 1. The dictionary

The same forward computation can be read three ways. The point of the toy model is
that the rightmost column is **exact and computable**, where the middle column is
only ever approximate.

| Differential cryptanalysis | Transformer | This repo (global + cache) |
|---|---|---|
| Plaintext difference (finite DDT) | token edit → embedding difference | a single injected symbol/count |
| Round function | attention + MLP block | one mixture step `M = λ·cache + (1−λ)·global` |
| S-box (fixed DDT) | MLP nonlinearity | the count→probability normalization |
| Self-rewiring S-box (no clean DDT) | softmax attention | the cache operator — but here **explicit, linear in counts** |
| Markov-cipher assumption | *approximation* (no round keys) | **literally true** — it is a Markov chain |
| Max differential probability / round | unbounded, no branch number | **Dobrushin coefficient** `τ(M)`, closed form |
| Differential hull (thick, uncomputable) | sum over trail cluster | **resolvent** `(I−M)⁻¹` on the mean-zero subspace |
| Weak keys / encystment site | dead ReLUs, `μ`-null tokens | **low-count contexts**, leverage `1/(n+a)` |
| Induction head (copy + attack vector) | QK match-and-copy | **PPM longest-match** — same dual role, provable |
| Self-propagation / sporulation | autoregressive feedback | **nonlinear Pólya urn**, condensation transition |
| Light cone of a perturbation | causal mask (position × layer) | **cache order** (fixed) or longest repeat (PPM) |
| Behavioral-fever cure | high sampling temperature | escape ↑ / `λ` ↓ / `α` ↑ — the repo's "safe-by-default" tuning |

What makes the toy model worth the trouble: every row that is *approximate* for a
transformer becomes a *closed form* here, because the architecture is an affine map
on the probability simplex with no softmax to hide the difference behavior inside.

---

## 2. Four objects that become exact

**Per-step difference decay = the Dobrushin coefficient.** A difference is a
signed measure `δ` with `Σδ = 0`; it propagates by `δ ↦ δᵀM`. Its worst-case
one-step contraction is exactly
`τ(M) = ½ · maxₛ,ₛ′ ‖M(s,·) − M(s′,·)‖₁`,
the literal "maximum differential probability per round." AES forces this small by
design (branch number); the mixture chain simply *has* a value, computable from its
rows. The wide-trail bound that does not exist for transformers exists here.

**The differential hull = the resolvent.** The sum over all trails sharing
endpoints is the iterated operator; over a stationary stretch it is the Neumann
series `(I − M)⁻¹` restricted to the mean-zero subspace — the chain's fundamental
matrix. The "thick hull" we could not compute for a transformer is a Green's
function you can invert. The one genuinely path-dependent piece — the cache updates
as the sequence streams — is still explicit (linear in counts), so its Jacobian is
writable.

**Encystment = low-count contexts, leverage = `1/(n+a)`.** Injecting one count of
target `x*` into a context seen `n` times moves the Dirichlet-combined prediction by
`Δ ≈ (1 − p)/(n + a) ≈ 1/(n+a)`. The weak key is the rare n-gram, and "rare" is a
number in the count table. This is textbook estimator variance — the honest core of
the whole picture.

**Sporulation = a nonlinear Pólya urn.** In *generation*, sampling a symbol
increments its own count, so under high context-reliance the process is positively
reinforced. Nonlinear reinforced urns exhibit condensation / lock-in; the
"sporangium spawning zoospores" image is literally a rich-get-richer phase
transition in the generation dynamics, with the order parameter being cache trust.

---

## 3. What `demo_conservation.py` measures

Source: the repo's synthetic two-layer model. A high-entropy global law `G`
(`G_CONC = 1.0`) is shared by all documents; each document also has a peaky private
law `D` (`D_CONC = 0.05`) mixed in with probability `pi_ctx`. The global model,
trained across documents, can only learn `G` (each `D` washes out), so the cache's
test-time gain is exactly the in-context information the global *provably cannot
have* — and `pi_ctx` dials how much of it exists.

Swept over `pi_ctx ∈ [0, 0.9]`, for cache orders 1 and 3:

- **benefit** — bits/symbol saved by the Dirichlet cache vs. global-only (usefulness).
- **reliance** — warmed-up cache weight `n/(n+a)` (trust).
- **static leverage** — *control*: prob-mass an attacker buys with one injected
  count at the live context, single query, non-propagating.
- **propagated leverage** — one injected count at one context, then the cache keeps
  streaming the true document; accumulate the extra probability placed on the poison
  target at every later step. This is the Green's-function influence of a unit
  perturbation through the (history-dependent) cache operator.

A second sweep fixes `pi_ctx` high and varies the Dirichlet weight (hence trust),
seeding one self-looping poison `x*→x*` and measuring its takeover of generation.

---

## 4. Results

**The conservation law is conditional on context sparsity** (`results/conservation_law.png`).

*Order 3 — sparse, trust earned.* As `pi_ctx` rises 0 → 0.9, reliance climbs
0.02 → 0.39, benefit climbs −0.02 → +0.47 bits, and propagated leverage climbs
0.52 → 0.77. They move together:

| quantity vs. benefit | Pearson r |
|---|---|
| propagated leverage | **+0.97** |
| reliance (trust) | +0.93 |
| static 1-shot (control) | −0.94 |

Here usefulness, trust, and propagated exploitability are one quantity. This is the
regime that matters for real LLMs — high-order / natural-language contexts are
mostly novel, so a context recurring *is* evidence of real structure (the same
reason the repo sees reliance ≈ 0.01 on real prose).

*Order 1 — dense, trust saturated.* Only 32 order-1 contexts exist, so the cache
fills regardless and reliance pins at ~0.91 across **all** `pi_ctx`. Propagated
leverage stays flat (~1.1–1.3) even at `pi_ctx = 0`, where the cache *hurts* by
−1.29 bits. corr(benefit, propagated) drops to +0.52. **The model is fully
poisonable precisely where its context is useless.** This decoupling is the
alarming finding, and it is the one the clean "usefulness = exploitability" slogan
would have hidden.

*The control behaves as a control should.* Static 1-shot leverage is **anti**-
correlated with usefulness in both regimes (≈ −0.95): raw per-cell pollutability
actually *falls* as the cache fills (`1/(n+a)` shrinks). One-shot leverage is a red
herring; only **propagated** leverage is the right exploitability object — which is
the cryptographic point that a differential's value is the hull, not a single trail.

**Condensation is a clean phase transition** (`results/condensation.png`). With one
seeded self-loop, poison takeover of generation stays near zero up to reliance ≈ 0.9,
then turns sharply upward: +0.01 at reliance 0.65, +0.15 at 0.96, +0.65 above 0.99.
A single injected count locks in once context is trusted enough — the Pólya-urn
knee, the sporulation event made quantitative.

---

## 5. The cure, as a design constraint

Behavioral fever is escape ↑ / `λ` ↓ / `α` ↑ — each caps the single-count leverage
`1/(n+a)` and lowers trust, pulling the generation dynamics back below the
condensation knee. The repo's held-out PPM-escape self-tuning (`demo_ppm_tuned.py`,
"safe-by-default across regimes") is exactly this immune response; the lens just
gives it a target: **keep worst-case single-count leverage below the condensation
threshold.** The cost is the conservation law's other face — leverage you deny the
parasite is retrieval precision you deny the model. The two sweeps here are the two
ends of that trade.

The sharper, measured statement: the dangerous quantity is **trust, not
usefulness**, and the order-1 result shows the two can be driven apart — so a
safe-by-default escape should be tuned against *trust saturation in dense contexts*,
not against measured usefulness, because a context can be maximally trusted (hence
maximally poisonable) while contributing nothing.

---

## 6. Honesty flags

- The exact pieces — Dobrushin coefficient as per-step contraction, fundamental
  matrix as hull, leverage `1/(n+a)`, cache-order as light cone, reinforced-urn
  condensation — are standard Markov-chain / reinforced-process results. The
  *assembly* lining them up one-to-one with differential cryptanalysis and the
  chytrid is a constructed correspondence, not a citable published result.
- The naive conservation law was **refuted** by its own experiment and replaced by
  the conditional version above. The figures are the evidence; the slogan was wrong.
- Everything is on the synthetic two-layer source. The claim that the order-3
  (earned-trust) regime is the one real high-order/natural-language contexts inhabit
  is motivated by the repo's own real-text reliance (~0.01) but is an analogy across
  architectures, not a measurement on text. Re-running these curves through
  `text_adapter.py` on a real corpus is the obvious next step.

*Reproduce:* `python3 demo_conservation.py` → `results/conservation_law.png`,
`results/condensation.png`, and the printed per-order correlation table.

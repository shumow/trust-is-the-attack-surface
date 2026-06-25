# Context contagion: self-reproducing token strings (running notes)

Branch `context-contagion`. Extends *Trust Is the Attack Surface*: from "a single
self-loop locks in" (the parent repo's condensation result) to **a whole string that
regenerates itself** when an empty cache, over a fixed global model, has observed it.

The experiment↔theory loop accumulates here; figures land in `results/`; the eventual
LaTeX write-up is downstream of this file.

## Setup and threat model

Empty order-`k` cache, fixed global model `G` (the parent repo's two-layer synthetic
source). Payload = a string `S` streamed `reps` times into the cache, then generation
is seeded with `S`'s last `k` symbols. Two regimes for the attacker, mapping onto
white-box vs. black-box differential cryptanalysis:

- **Regime A (white-box):** attacker knows `G`, the prior `pg` at every context, and
  `a`. Can compute the *minimal / stealthy* quine.
- **Regime B (black-box):** only sampled-generation access. Conjecture: contagion is
  always achievable black-box (repetition is model-agnostic), but minimality is
  white-box-only; the gap is a query-complexity term (cost of finding `G`'s rarest
  symbols / lowest-count contexts).

## The exact-quine ladder

`S` is self-reproducing at order `k` iff every length-`k` window in `S` has a unique
successor (de Bruijn / Eulerian determinism; branch points = divergence).
- order 1: **rainbow cycle** of distinct symbols `A→B→…→A` (constant run = degenerate
  length-1 case).
- order `k`: **de Bruijn cycle** `B(V,k)`.

Determinism is necessary but not sufficient *in generation*: the cache counts must
out-weigh the global prior `a·pg`. That is the noisy-robust half, governed by reliance.

## Theory (closed form)

At a payload context with all `r` counts on the unique successor `s`,
`combine_dirichlet` gives per-step on-quine probability

    p_step(ρ) = ρ + (1−ρ)·pg(s),     ρ = r/(r+a)   (reliance).

Static expected run before first slip is geometric: `1/(1−p_step)`, which diverges as
ρ→1 — the contagion knee, quantitative. This is a **lower bound**: in generation each
correct step increments its own count (Pólya), nudging ρ up, so realized run length
should *exceed* the static bound.

## Result 1 — order-1 rainbow quine (`demo_contagion.py`, `results/contagion_fidelity.png`)

Payload = the 5 globally-rarest symbols as a rainbow cycle, so `pg(correct)=0.015`
(the global would essentially never emit it; all reproduction is cache trust).

| reps | reliance ρ | transition fidelity | static theory | run length | static `1/(1−p)` |
|---|---|---|---|---|---|
| 5  | 0.800 | 0.814 | 0.803 | 5.7  | 5.1  |
| 13 | 0.923 | 0.932 | 0.924 | 18.7 | 13.2 |
| 34 | 0.971 | 0.974 | 0.971 | 39.0 | 34.5 |
| 55 | 0.982 | 0.984 | 0.982 | 78.7 | 55.5 |
| 89 | 0.989 | 0.993 | 0.989 | 159.3| 90.9 |

Two findings:
1. **Transition fidelity matches the closed form to ~3 digits**, sitting slightly
   *above* it at high ρ — the reinforcement bonus.
2. **Run length beats the static geometric bound** once ρ crosses ~0.65 and diverges
   at the knee (159 vs. 91 at ρ=0.989): the quine **self-heals**. Occupancy of the
   quine alphabet → 0.96, i.e. even after a slip the walk re-enters the cycle.

**Methodological note:** positional fidelity (vs. a fixed phase) is the wrong metric —
one slip phase-shifts the cycle and tanks all later positions even while the cache is
faithfully reproducing. Score at the **transition level** (did the next symbol obey
the successor rule given the actual previous window). This is what matches theory and
is the honest measure of reproduction.

## Results 1.2–1.5 — the order ladder, orders 2..5 (`demo_contagion_orders.py`)

Canonical order-`k` payload = a **binary de Bruijn cycle** `B(2,k)`, period `2^k`,
embedded on the two globally-rarest symbols (alphabet fixed at 2 to isolate order).
Figures `results/contagion_orders.png`, `results/contagion_collapse.png`.

**(i) Per-step contagion is order-blind.** In the contagious regime (ρ ≳ 0.9) the
per-step transition fidelity of every order 2..5 collapses onto the *same* closed form
`ρ + (1−ρ)·pg` from Result 1. Order does not change the per-step law; it only sets how
many independent steps must all succeed.

**(ii) Brittleness ladder (doubly exponential in order).** Surviving one whole period
needs `p_step^(2^k)`, so reproduced-periods-before-slip falls sharply with order. At
ρ=0.996: order 2 → 63 periods, order 3 → 30, order 4 → 15, order 5 → 9 (≈ halving per
order). Higher-order quines are exponentially harder to keep contagious — the order is
a brittleness dial, not a per-step one.

**(iii) The divisibility question, resolved by a theorem — and relocated.** The
collapse spectrum shows **every settled run lands on the full period `2^k`, with zero
mass on any proper divisor** (no harmonics), at every order. Reason (verified on 2157
random exact quines, orders 1–4, periods 2–12, alphabets 2–4, zero counterexamples):

> **Window-distinctness theorem.** A string is an exact order-`k` quine of genuine
> period `p` **iff** its `p` length-`k` windows are all distinct. Hence its de Bruijn
> subgraph is a single `p`-cycle of out-degree-1 nodes, which has **no proper
> sub-cycle**. (If a `k`-window repeated, determinism would force the same
> continuation, making the true period a proper divisor — contradiction.)

So for **exact** quines, period factorization / order primality is **irrelevant**:
divisor sub-cycles do not exist to collapse into, prime or composite. The divisibility
/ prime-period intuition is well-posed **only for approximate (branch-point) quines**,
where a divisor sub-cycle can become a competing attractor and a prime period would
forbid it. That is exactly the **noisy-robust** regime — so this milestone hands the
divisibility question directly to milestone 2 with a sharp prediction:

> **Prediction (to test in 1.6 / milestone 2):** for *approximate* quines of period
> `p`, sub-cycle lock-in occurs on proper divisors of `p`; **prime-period** approximate
> quines are protected from harmonic collapse (only full-reproduction or break-to-noise),
> while composite-period ones collapse to their largest proper divisor.

## Result 1.6 / 2 — approximate quines: divisibility gates collapse (`demo_contagion_divis.py`)

Approximate payload = a **quasi-periodic quine**: `m` blocks sharing a period-`d`
motif, distinguished only by a leading tag symbol. Genuine period `p = m·d`; the
order-1 structure is just the period-`d` motif; resolving the `m` blocks (hence the
full period) needs cache order ≥ `d`, because the disambiguating tag sits `d`
positions back. `d` is taken as `p`'s **largest proper divisor** (`p / smallest prime
factor`); for a **prime** `p` there is no nontrivial `d`, so the construction
degenerates to an exact rainbow `p`-cycle — no sub-motif to collapse onto.

**Experiment B — divisibility sweep, order-1 cache** (`results/contagion_divis_sweep.png`).
The prediction holds exactly:

| p | type | modal emergent period | interpretation |
|---|---|---|---|
| 7, 11, 13 | prime | 7, 11, 13 (frac_full ≈ 1.0) | reproduces full period; no collapse |
| 8 | comp (lpd 4) | 4 (65%) | collapses to largest proper divisor |
| 9 | comp (lpd 3) | 3 (42%) | " |
| 12 | comp (lpd 6) | 6 (68%) | " |
| 15 | comp (lpd 5) | 5 (47%) | " |
| 16 | comp (lpd 8) | 8 (70%) | " |
| 25 | comp (lpd 5) | 5 (57%) | " |

**Prime period = harmonic protection.** A prime-period approximate quine has no proper
sub-cycle to fall into, so it either reproduces fully or breaks to noise; a composite
one collapses onto its largest proper divisor. This is the user's prime-order /
divisibility intuition, confirmed — and it lives precisely where the
window-distinctness theorem said it must: the *approximate* regime.

**Experiment A — order threshold** (`results/contagion_divis_order.png`, `p=15=5·3`).
Sweeping cache order, the **divisor-collapse fraction falls to exactly 0 at `k = d`**
(the minimal resolving order) while full-reproduction jumps up there. Two order effects
fight: raising order is needed to *resolve* the full period (escape the harmonic), but
raising order also raises *brittleness* (the 1.2–1.5 ladder), so full reproduction
needs jointly `k ≥ d` **and** high reliance — at low reps the high-order full
reproduction is starved and nothing settles. Escaping a harmonic by raising order is
not free; it is paid for in required trust.

**Synthesis so far.** Three knobs, cleanly separated: **reliance ρ** sets per-step
fidelity (`ρ+(1−ρ)pg`, order-blind); **order k** sets brittleness (`p_step^{period}`)
and the resolving threshold `k≥d`; **period factorization** sets the available
sub-cycle attractors (none for exact / prime; the divisor lattice for composite
approximate). A defender's takeaway falls out: a payload that is *robust* (low order,
short period) is easy to make contagious but offers only coarse/­harmonic-prone
control; a *precise* long high-order payload needs near-saturated trust to survive.

## Result 1.7 — closing Q1's two attacker regimes (`demo_contagion_regimes.py`)

Empty cache, fixed global. Attacker wants the minimal payload (reps `r`, hence injected
counts) to reach a one-step poison probability `q` (equivalently expected run length
`T = 1/(1−q)`; `q` composes — it bounds run length and feeds condensation). Payload =
the self-loop `x*→x*` (the condensation primitive); `x*` swept across the vocab gives a
clean loudness axis `pg = pg(x*|x*)`.

**Regime A — white-box (knows `pg`, `a`).** Closed form, exact on this model since
`p_step = (r + a·pg)/(r + a)`:

> `r* = a·(T(1 − pg) − 1) = a·(q − pg)/(1 − q)`.

The attack *is* evaluating this — zero queries.

**Regime B — black-box (knows neither `pg` nor `a`, only sampled generations).**
- **Zero queries:** the `pg=0` worst case `r0 = a·(T − 1)` **guarantees** the target
  against *any* global (reliance → 1 dominates the prior). Contagion is always
  achievable black-box, model-agnostically. This answers "the most that can happen
  unknown": full contagion, at the `r0` rep cost.
- **With queries:** an adaptive **doubling + bisection** search on `r`, using only a
  *chosen-prefix one-step oracle* (prime with `x*`, observe the next token, repeat),
  recovers `r*`. Convergence: `r_bb → r* ≈ 29.7` with std shrinking 16.4 → 1.3 as
  queries grow 525 → 51k (≈ `1/ε²` tokens to resolve to ±ε reps near the threshold).

**Headline — the regime gap is `r0 − r* = a·T·pg`, the value of knowing `G`, and it
VANISHES as `pg → 0`.** Knowing the model buys the attacker the most against *loud*
payloads it already likes, and *nothing* against maximally rare ones — so black-box ≡
white-box exactly in the **adversarial corner** (stealthy, rare-symbol payloads). The
"what does the attack algorithm look like" answer: white-box = evaluate `r*`; black-box
= zero-query `r0` fallback, or doubling+bisection on the one-step oracle to approach
`r*`.

**Methodology flags (both load-bearing).** (1) The observable must be the **one-step**
poison probability via chosen-prefix sampling; the autoregressive run-fidelity is
biased low (a self-loop that slips early contributes `run/(run+1)`, over-penalizing
short runs) and inflates apparent `r` ~2×. (2) `pg` must be conditioned at the global's
**own order** (`GLOBAL_ORDER`), not order-1, or loudness is misread. (3) This demo uses
a peakier `G_CONC=0.3` to make a visible loudness axis; the parent's high-entropy
`G_CONC=1.0` is the **all-stealthy corollary** — no transition is predictable, so the
gap ≈ 0 everywhere and black-box ≈ white-box *always*. That is consistent with the
project thesis: real sparse high-order contexts are stealthy, so the attacker gains
little from knowing the model.

## Result 2.1 — Q2: infecting a pre-trained cache (`demo_contagion_hijack.py`)

Victim = an order-1 cache **warmed on a legitimate document `D`** over a fixed global.
Payload `x*` = an **out-of-distribution token** sanitized out of corpus and `D` (a
trigger/marker the legitimate system never emits). Hijack factorizes as **entry ×
lock-in**: inject a self-loop `x*→x*` (lock-in, condensation, Q1) plus a *bridge* edge
`c→x*` (entry). Lock-in fixed strong (`W_LOOP=300`, reliance ≈0.997); spotlight on entry.

**Entry is the new physics, measured as `P(first x* within T)` with no lock-in.**
Spontaneous leak is **exactly 0** at `w=0` — a trained cache will not emit a novel
payload token on its own (validates the premise). A bridge adds entry rate

> `freq(c) · p(x*|c, w)`,  `p(x*|c,w) = (w + a·pg)/(n_c + w + a)`.

Measured rates match this closed form (`results/contagion_hijack.png`). Minimal poison
to make entry likely (`P_enter ≥ 0.5`, `T=60`):

| strategy | ctx | freq(c) | n_c | min poison |
|---|---|---|---|---|
| **best bridge** = max `freq/(n+a)` | 11 | 0.043 | 10 | **3 counts** |
| most visited = max `freq` | 14 | 0.090 | 33 | 8 |
| random visited | 18 | 0.015 | 6 | 13 |
| lowest count = min `n` | 7 | 0.0075 | 2 | never |

**The answer to "what makes the best entry" is regime-dependent, and there is a hard
ceiling:**
- **Low poison (the minimal-poison regime):** best `= max freq(c)/(n_c+a)` — leverage
  matters, so a moderately-visited *under-warmed* context beats the most-visited one
  (which has high `n_c`, low leverage). Best-bridge hijacks in **3 counts** vs 8 for
  most-visited.
- **High poison:** `p(x*|c)→1`, so entry rate → `freq(c)` and best `= max freq`.
- **Frequency ceiling:** entry rate saturates at `freq(c)`, so a context with
  `freq(c) < −ln(1−target)/T` can **never** reach the target however much poison is
  injected (ctx 7, freq 0.0075, never crosses 0.5). A viable bridge must be visited
  enough; leverage then sets the cost *among* viable contexts.

**Probabilistic hijack.** `P(hijack) = P(enter in T) × P(lock-in)`, with
`P(enter) = 1 − (1 − freq(c)·p(x*|c,w))^T` (entry a geometric first-passage on the
trajectory) and `P(lock-in)` the condensation curve. Lock-in check: most-visited bridge
at `w=8` → takeover occupancy 0.30 (enters mid-window, then locks for the remainder),
confirming the factorization. Information the attacker needs = the victim's **visit
frequencies** (trajectory) and **warmed counts `n_c`** (leverage) — white-box reads
both; black-box estimates `freq` from observed generations (the Q1 regime split carries
over to entry-finding).

## Open / next

- **Order-`k` de Bruijn quines:** does the same `p_step(ρ)` hold with `pg` averaged
  over the de Bruijn contexts? Brittleness should rise with `k` (more contexts to keep
  deterministic under noise).
- **White-box minimal solve (Regime A):** given `G,k,a`, closed-form minimum `reps`
  (hence min injected counts) for target run length / takeover. Pick cycle symbols to
  minimize it.
- **Black-box attacker (Regime B):** adaptive loop that estimates rare symbols / `a`
  from sampled generations; measure query complexity vs. the white-box optimum — the
  regime gap.
- **Q2 — infecting a *trained* cache:** hitting-then-locking. Best entry = bridge
  context maximizing `P(visited under D) × leverage 1/(n+a)`; min poison; hijack
  probability = `P(enter basin in T) × P(lock-in)`.

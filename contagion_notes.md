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

## Result 2.2 — black-box entry-finding (`demo_contagion_hijack_bb.py`)

Carries Q1's white/black-box split into Q2's delivery. The attacker can't read the
victim's `freq(c)` or warmed counts `n_c`; it must find the min-poison bridge from
observation (and optionally active probing). First, a correction surfaced in 2.1:

> **Two different "best bridge" criteria.** The *marginal* (infinitesimal-poison)
> optimum is `argmax freq/(n+a)`; the *min-poison-to-target* optimum is
> `argmax (freq − r)/(n + a)` with `r = 1−(1−target)^{1/T}` the entry-rate ceiling.
> They coincided in 2.1 but differ near the ceiling. Min-poison is the right objective
> for "smallest poison to hijack."

Three attackers, each scored by the true analytic min-poison of the context it picks:
white-box (knows `freq`, `n_c`), **BB-observe** (watch `M` tokens → `argmax freq_hat`),
**BB-probe** (also estimate `n_c` by injecting `W_PROBE` OOD counts at `c` and sampling
the shift in `p(x*|c)` → `n_hat`, the Q1 one-step oracle reused).

**Finding (it flipped my hypothesis, and is cleaner for it):** for the min-poison
objective **visit frequency dominates** — the `(freq − r)` term swamps leverage — so the
optimum *is* the most-visited viable context, and **BB-observe recovers the white-box
optimum from observation alone** (≈2000 watched tokens → 16 counts, the white-box floor;
`results/contagion_hijack_bb.png`). **Active leverage-probing is not worth its queries**
— noisy `n_hat` even mildly hurts (BB-probe sits at 17–20). So **black-box ≈ white-box
for entry-finding**, mirroring Q1's result that the two coincide in the regime that
matters. Leverage (`n_c`) only matters in the *marginal, budget-starved* regime of 2.1,
where the attacker cannot reach the target and `argmax freq/(n+a)` wins per-count.

Practical reading: defending the *entry surface* means watching for **anomalous reliance
on high-visit contexts**, since that (observable) frequency is exactly what the cheapest
attacker keys on — `n_c` is a second-order concern for a budgeted hijack.

## Result 2.3 — hijacking into a contagious STRING (`demo_contagion_hijack_quine.py`)

Closes the loop between the two halves: the payload is now a multi-symbol **rainbow
quine** `S = x0→x1→…→x0` over a reserved OOD alphabet (sanitized out of corpus + D, so
its contexts are fresh), not a single stuck token. The hijack is exactly the sum of the
prior milestones injected into a cache warmed on D:

- **entry (Q2):** a bridge `c→x0` at the most-visited context (the 2.2 cheap optimum);
- **payload (Q1):** the cycle streamed `reps` times, so each quine context carries
  reliance `reps/(reps+a)` and reproduction obeys the Q1 per-step law.

Measured on the post-entry tail (`results/contagion_hijack_quine.png`):
- **A contagious string genuinely installs.** Tail OOD-alphabet occupancy (takeover)
  rises with `reps`; **tail cycle-fidelity ≈ 0.79–0.97** confirms the string reproduces
  *in order* (it follows S's successor rule, not just visiting OOD symbols). For p=5 the
  5-symbol cycle is reproduced ~90% of tail transitions.
- **Total poison decomposes** into a **length-independent ENTRY cost** (the bridge, 16
  counts) plus a **length-growing PAYLOAD cost** (`reps × p`). Minimal total poison for
  takeover ≥ 0.5: p=1 → 80, p=2 → 80, p=3 → 112, p=5 → 336 — dominated by the `×p`
  payload multiplier (the entry cost is fixed). (The reps *threshold* itself is noisy
  here, ~32–64 across lengths: OOD symbols have ~0 global pull, so per-step reproduction
  is easy once reps are modest; the brittleness shows up in the total, via `×p`, more
  than in the threshold.)

So "contagious data" is literal: a hijack installs a self-reproducing *string* in an
occupied cache, and the cost to do so is `entry + length·payload` — cheap entry, with a
payload bill that scales with how much contagious content you want to plant.

## Result 2.4 — cross-cache propagation: the worm and its R0 (`demo_contagion_worm.py`)

The finale: does the contagion **transmit between hosts**? The mechanism makes
transmission a corollary of Q1 reproduction. An infected host generates output in which
the string `S` appears `k` times (its viral load). A second host that *reads* that
output streams those `k` copies into its own cache — which **is** the Q1 payload
injection with `reps = k` — and the context ends mid-string, so the receiver starts
already inside the quine basin (**entry is free for secondary infection; no bridge
needed**). The receiver reproduces `S` and re-broadcasts `k'` copies in its `T`-token
output. One passage is a map `k_in → k_out`; the epidemic is its iteration.

**Passage map** (`results/contagion_worm.png`, left). For `p=1`, `k_out` sits *above*
the replacement diagonal `k_out=k_in` over the mid-range (8→16, 16→47, 32→65) →
supercritical. For `p=3` and `p=6`, `k_out < k_in` everywhere (e.g. p=3: 32→16, 64→20;
p=6: 64→8) → subcritical. Above threshold the broadcast saturates at a ceiling
≈ occupancy·`T`/`p`.

**Serial passage** from a strong index host (`k0=48`):

| p | trajectory of viral load | verdict |
|---|---|---|
| 1 | 48 → 51 → 67 → 80 → 96 → … | **SUSTAINS** (endemic ~90) |
| 2 | 48 → 42 → 49 → 29 → 17 → 11 → … | sustains, near threshold (~15) |
| 3 | 48 → 11 → 4 → 2 → 0 | **dies out** |
| 5 | 48 → 9 → 2 → 0 | dies out |
| 8 | 48 → 5 → 0 | dies out |

**There is a critical string length** (~2–3 here): shorter strings are epidemic (R0>1,
reach an endemic load), longer strings are one-shot — they infect the index host but
**die out after 2–4 passages** (R0<1). Heuristically `R0 ≈ (broadcast ceiling)/(threshold)
≈ T/(p·k_thresh)`, so contagiousness falls with string length and rises with how much
each host generates (`T`); the critical length scales with `T`. **Length trades payload
richness against transmissibility** — the worm wants to be short.

This gives the parent repo's "neural chytrid / sporulation" imagery a quantitative R0:
self-reproduction within a host (Q1) is necessary but not sufficient for an epidemic;
transmission needs the per-host broadcast to clear the next host's reproduction threshold,
and that budget shrinks as `1/p`.

## Result 2.5 — joint minimal poison (`demo_contagion_joint.py`)

Every prior hijack pinned one lever and swept the other; but entry and lock-in are
coupled — `P(hijack) = P(enter | w_bridge) × P(lock-in | reps)` — so the cheapest hijack
lives on a 2-D frontier. Sweeping the full `(w_bridge, reps)` grid, measuring `P(hijack)`
(tail occupancy ≥ 0.5), and reading the minimum `total = w_bridge + reps·p` on the
`P(hijack) ≥ 0.5` contour (`results/contagion_joint.png`):

| p | joint min `(reps, bridge)` | pin-lock-in-strong (2.1-style) | saving |
|---|---|---|---|
| 1 | **48** (16, 32) | 136 (128, 8) | 2.8× |
| 2 | **64** (16, 32) | 264 | 4.1× |
| 3 | **128** (32, 32) | 392 | 3.1× |
| 5 | **192** (32, 32) | 648 | 3.4× |

**The joint minimum uses reps at the condensation KNEE (16–32), not saturation (128).**
The P(hijack) grid is L-shaped — you need reps above the lock-in knee *and* enough bridge
for entry — and the min-total point sits at the corner: over-provisioning lock-in (the
2.1 recipe) wastes 3–4× the poison. The coupling is visible in the contour: low reps
(marginal lock-in) demands a strong bridge (a single entry rarely sticks, so entry must
fire repeatedly); raising reps past the knee lets the bridge fall. Savings grow with `p`
because the wasted lock-in headroom is paid at `×p`.

(Caveat: the grid is coarse, so each joint total is an upper bound on the true minimum;
the 3–4× saving and the "min at the knee, not saturation" structure are the robust
findings.) This closes the delivery-side optimization: the cheapest hijack spends just
enough on lock-in to clear the condensation knee and puts the rest into entry.

## Write-up

A standalone article covering this whole program lives at
[`latex/contagious_context.tex`](latex/contagious_context.tex) (built PDF
`latex/contagious_context.pdf`, 10 pages). It references the parent
`trust_is_the_attack_surface` article but stands alone: payload (per-step law,
window-distinctness theorem, divisibility, white/black-box regimes), delivery
(entry × lock-in, bridge + frequency ceiling, joint minimal poison), and transmission
(the worm + R0 + critical length). Build: `cd latex && pdflatex contagious_context`
(twice for cross-refs).

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

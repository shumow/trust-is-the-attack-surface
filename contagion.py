"""
Context contagion: self-reproducing token strings on the global+cache substrate.

This module is the payload layer for the "echolalia" question -- strings S such
that an *empty* cache, after observing S, regenerates S when run in generative
mode. It sits on top of the markov-cache substrate exactly like demo_conservation,
and keeps the same discipline: every object here is exact on the toy.

The exact-quine ladder
----------------------
A string S is "self-reproducing at cache order k" iff, after the cache observes S,
seeding generation with S's last k symbols regenerates S. The condition is purely
combinatorial: every length-k window in S must be followed by a UNIQUE symbol
(branch points = divergence). That is the de Bruijn / Eulerian condition:

  * order 1: a RAINBOW CYCLE of distinct symbols  A -> B -> C -> ... -> A.
             The constant run x* x* x* is the degenerate length-1 cycle.
  * order k: a de BRUIJN CYCLE B(V,k) -- every k-window occurs once.

Exact determinism is necessary but not sufficient in *generation*: the cache must
also out-weigh the global prior a*pg. That is the noisy-robust half of the story,
and it is governed by the same condensation/reliance knee as the parent repo --
see `predicted_run_length` for the closed form this module then measures.
"""
import numpy as np

from markov_cache import make_predictor, generate


# --------------------------------------------------------------------------- #
# Quine constructors
# --------------------------------------------------------------------------- #
def rainbow_cycle(symbols):
    """Order-1 exact quine: a cycle through DISTINCT symbols. Each symbol has a
    unique successor, so the order-1 context->symbol map is deterministic.
    `symbols` is a list of distinct ints; returns it unchanged (one period)."""
    if len(set(symbols)) != len(symbols):
        raise ValueError("rainbow_cycle needs distinct symbols (order-1 determinism)")
    return list(symbols)


def debruijn_cycle(V, k):
    """Order-k exact quine: a de Bruijn sequence over an alphabet of size V, in
    which every length-k word appears exactly once -- hence every k-window has a
    unique successor. Returns one period (length V**k) as a list of ints in
    [0, V). Standard greedy/prefer-largest construction."""
    a = [0] * (k * V)
    seq = []

    def db(t, p):
        if t > k:
            if k % p == 0:
                seq.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, V):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return seq


def is_deterministic(S, k):
    """True iff S (treated cyclically) is an exact order-k quine: no k-window is
    followed by two different symbols."""
    succ = {}
    n = len(S)
    for i in range(n):
        ctx = tuple(S[(i + j) % n] for j in range(k))
        nxt = S[(i + k) % n]
        if succ.setdefault(ctx, nxt) != nxt:
            return False
    return True


def true_period(S):
    """Smallest d | len(S) such that S is d-periodic (its genuine period). A prime
    length forces this to be len(S) unless S is constant -- the number-theoretic
    reason prime-period quines cannot collapse to a proper sub-cycle."""
    n = len(S)
    for d in range(1, n + 1):
        if n % d == 0 and all(S[i] == S[i % d] for i in range(n)):
            return d
    return n


def minimal_order(S, max_k=None):
    """Smallest k for which S is an exact order-k quine (deterministic). The order
    of the payload, independent of its period."""
    max_k = max_k or len(S)
    for k in range(1, max_k + 1):
        if is_deterministic(S, k):
            return k
    return None


def find_quine(period, order, n_symbols, rng, tries=40000):
    """Random search for a cyclic sequence of genuine period `period` over
    `n_symbols` symbols whose MINIMAL order is exactly `order`. Lets us vary period
    (prime vs composite) and order independently -- the lever for the divisibility
    question. Returns a list, or None if the search fails."""
    for _ in range(tries):
        S = list(int(x) for x in rng.integers(0, n_symbols, size=period))
        if true_period(S) == period and minimal_order(S, order) == order:
            return S
    return None


def quasiperiodic_quine(tags, motif_tail):
    """Approximate quine with a dominant period-d sub-motif, used to test whether
    period DIVISIBILITY gates sub-cycle collapse (the prime-period question).

    m = len(tags) blocks, each block = [tag_i] + motif_tail, so blocks share the
    motif and differ only by their leading tag. d = len(motif_tail)+1 is the motif
    period; the genuine period is m*d (tags are distinct). At order 1 the structure
    is just the period-d motif (...A_{d-1} -> some tag -> A_1...), a divisor of the
    period; resolving the m blocks (hence the full period) needs cache order >= d,
    because the tag that disambiguates a block sits d positions back. For a PRIME
    period the only factorization is 1*p, so d=1 and this degenerates to an exact
    rainbow p-cycle -- there is no proper sub-motif to collapse onto.

    Returns the payload string (list of ints).
    """
    S = []
    for t in tags:
        S.append(t)
        S.extend(motif_tail)
    return S


def smallest_prime_factor(p):
    d = 2
    while d * d <= p:
        if p % d == 0:
            return d
        d += 1
    return p


def dominant_period(seq, max_p):
    """The period the generated sequence has actually settled into: the lag d in
    1..max_p maximizing P(seq[i] == seq[i-d]). Returns (d, score). A faithful quine
    gives d = its full period with score ~1; a collapse to a proper sub-cycle gives
    a divisor d; noise gives a low score. This is where divisor harmonics show up."""
    seq = np.asarray(seq)
    n = len(seq)
    best_d, best_s = 1, -1.0
    for d in range(1, min(max_p, n - 1) + 1):
        s = float(np.mean(seq[d:] == seq[:-d]))
        if s > best_s:
            best_s, best_d = s, d
    return best_d, best_s


# --------------------------------------------------------------------------- #
# Train-then-generate: stream the payload into an empty cache, then reproduce
# --------------------------------------------------------------------------- #
def stream_into(pred, S, reps):
    """Reset `pred` and stream `reps` whole copies of S into its cache (the empty
    cache 'observing' the payload). Returns the streamed history list."""
    pred.reset()
    hist = []
    for _ in range(reps):
        for sym in S:
            pred.observe(hist, sym)
            hist.append(sym)
    return hist


def reproduce(pred, S, reps, rng, *, k, gen_len, n_trials=20):
    """Stream `reps` copies of S into the cache, then generate `gen_len` symbols
    seeded with S's last k symbols. generate() keeps observing as it samples
    (prequential), so the cache self-reinforces during reproduction -- the
    noisy-robust dynamic itself.

    Reproduction is scored at the TRANSITION level, not by position: a single slip
    only phase-shifts a positional comparison, so we instead ask, at each step,
    whether the generated symbol followed the quine's successor rule given the
    actual previous symbol (k-window). This is phase-robust and is exactly the
    quantity `predicted_per_step` models.

    Returns a dict with
      transition_fidelity = fraction of steps that obeyed the successor rule
                            (over steps whose k-window is a known quine context),
      occupancy           = fraction of generated symbols in the quine alphabet,
      run_length          = correct steps before the first slip (>= the static
                            geometric bound when reinforcement self-heals),
      reliance            = warmed-up cache weight on a payload context n/(n+a).
    """
    p = len(S)
    succ = {tuple(S[(i + j) % p] for j in range(k)): S[(i + k) % p]
            for i in range(p)}
    alphabet = set(S)
    fids, occs, runs, periods = [], [], [], []
    for _ in range(n_trials):
        stream_into(pred, S, reps)
        prompt = list(S[-k:])
        out = generate(pred, gen_len, rng, prompt=prompt)
        gen = out[len(prompt):]
        window = list(prompt)
        correct, scored, run, run_open = 0, 0, 0, True
        for sym in gen:
            ctx = tuple(window[-k:])
            if ctx in succ:
                scored += 1
                ok = (sym == succ[ctx])
                correct += ok
                if run_open and ok:
                    run += 1
                elif run_open:
                    run_open = False
            window.append(sym)
        fids.append(correct / scored if scored else 0.0)
        occs.append(float(np.mean([s in alphabet for s in gen])))
        runs.append(run)
        # emergent period of the settled tail (collapse analysis)
        tail = gen[len(gen) // 2:]
        d, score = dominant_period(tail, max_p=2 * p)
        periods.append((d, score))
    stream_into(pred, S, reps)
    rel = float(pred.local_reliance(list(S[-k:])))
    return {
        'transition_fidelity': float(np.mean(fids)),
        'occupancy': float(np.mean(occs)),
        'run_length': float(np.mean(runs)),
        'run_periods': float(np.mean(runs)) / p,
        'reliance': rel,
        'periods': periods,                       # list of (dominant_d, score)
    }


# --------------------------------------------------------------------------- #
# Theory: the closed form the experiment is supposed to match
# --------------------------------------------------------------------------- #
def white_box_min_reps(pg_correct, a, target_run, counts_per_rep=1.0):
    """Regime A (white-box) closed form. Minimal repetitions to reach expected
    on-quine run length `target_run = T`, given the KNOWN global prior pg at the
    payload contexts and Dirichlet strength a. From the static bound
    1/(1-p_step) >= T with p_step = (r + a*pg)/(r + a):

        r >= a * (T*(1 - pg) - 1).

    counts_per_rep c rescales when a context gets c counts per repetition. This is a
    conservative (static) bound: generation's self-reinforcement lets the real attack
    reach T with somewhat fewer reps."""
    r = a * (target_run * (1.0 - pg_correct) - 1.0)
    return max(0.0, r) / counts_per_rep


def zero_knowledge_min_reps(a, target_run, counts_per_rep=1.0):
    """Regime B with ZERO queries. The pg=0 worst case of the white-box bound,
    r >= a*(T - 1): it guarantees run length T against ANY global model (reliance
    dominates the prior), so it needs no knowledge of G at all. The gap to the
    white-box optimum is a*T*pg / counts_per_rep -- the value of knowing the model --
    which VANISHES as pg -> 0, i.e. exactly for the stealthiest (rarest) payloads."""
    return a * (target_run - 1.0) / counts_per_rep


def predicted_per_step(reliance, pg_correct):
    """Static (pre-reinforcement) probability of staying on the quine for one step
    at a payload context: p = reliance + (1-reliance)*pg_correct, from
    combine_dirichlet p(s) = (r + a*pg_s)/(r + a) with r counts all on the unique
    successor s. This is a LOWER bound on realized fidelity, because generation
    reinforces correct steps (Polya), nudging reliance upward as it goes."""
    return reliance + (1.0 - reliance) * pg_correct


def predicted_run_length(reliance, pg_correct):
    """Expected number of correct steps before the first slip, geometric with the
    static per-step on-quine probability: 1 / (1 - p). Diverges as reliance -> 1,
    which is the contagion knee made quantitative."""
    p = predicted_per_step(reliance, pg_correct)
    return np.inf if p >= 1.0 else 1.0 / (1.0 - p)

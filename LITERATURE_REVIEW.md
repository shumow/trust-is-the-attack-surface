# Literature review and citation audit

This note separates the background literature from this repository's own claims. It
also records whether the two paper drafts cite those sources faithfully.

## Summary judgment

The existing citations are mostly faithful. The strongest sources support the
following background claims:

- Transformer work does distinguish parametric memory from in-context adaptation,
  and Bietti et al. use a synthetic global/context-specific bigram setup close in
  spirit to this repo's toy substrate.
- Induction heads are described as attention heads that implement the pattern
  `[A][B] ... [A] -> [B]`, with strong causal evidence in small attention-only
  models and weaker/correlational evidence in larger models.
- Indirect prompt injection and RAG-mediated AI worms are real, documented threat
  models.
- Witten-Bell smoothing, de Bruijn cycles, reinforced urns, and epidemic
  reproduction-number language are legitimate mathematical background.

The main citation fix is wording, not substance: real transformer induction heads
should not be described as "precisely" longest-match caches. The faithful phrasing
is that the repo's PPM predictor is a toy longest-match/count surrogate for the
copy behavior that induction-head papers study.

## Related work

### Parametric memory versus in-context memory

Bietti, Cabannes, Bouchacourt, Jegou, and Bottou's *Birth of a Transformer: A
Memory Viewpoint* studies how transformers balance knowledge from weights with
information supplied in context. Their synthetic setup has tokens generated from
global and context-specific bigram distributions, and their simplified transformer
learns global bigrams before developing induction-head behavior for in-context
bigrams.

Citation fidelity: good. The repo's "global Markov model plus online cache" is not
their model, but it is a fair from-scratch surrogate for the same parametric vs.
in-context distinction.

Source: https://arxiv.org/abs/2306.00802

### Induction heads and copy mechanisms

Olsson et al.'s *In-context Learning and Induction Heads* defines induction heads as
attention heads that complete sequences like `[A][B] ... [A] -> [B]`. The paper
presents multiple lines of evidence that induction heads are involved in
in-context learning, with strong causal evidence for small attention-only models
and more correlational evidence for larger models with MLPs.

Citation fidelity: mostly good, with one overstatement corrected in this branch.
It is faithful to use Olsson et al. for the copy/previous-token mechanism and for
the hypothesis that induction is important to in-context learning. It is too strong
to say real induction heads are "precisely" longest-match caches.

Source: https://arxiv.org/abs/2209.11895

### Classical cache language models and smoothing

Goodman's *A Bit of Progress in Language Modeling* reviews several count-based
language-modeling techniques, including caching, higher-order n-grams, smoothing,
clustering, and combinations of those methods. The paper is a good citation for
classical language-model caching/smoothing background, including Witten-Bell-style
interpolated count modeling.

Citation fidelity: good for the paper's role as language-modeling background. The
repo's exact Dirichlet cache mixture is its own toy construction, not something
Goodman claims as a context-poisoning model.

Source: https://arxiv.org/abs/cs/0108005

### Indirect prompt injection and context poisoning

Greshake et al.'s *Not What You've Signed Up For* argues that LLM-integrated
applications blur data and instructions, introduces indirect prompt injection via
retrieved external data, and demonstrates attacks against real and synthetic
applications. It explicitly includes risks such as data theft, tool/API control,
worming, and information ecosystem contamination.

Citation fidelity: good. The repo's "context poisoning through retrieved documents,
tool output, or prior turns" is well aligned with this source.

Source: https://arxiv.org/abs/2302.12173

### Self-replicating prompts and AI worms

Cohen, Bitton, and Nassi's *Here Comes The AI Worm* demonstrates Morris-II, a
zero-click worm-like chain reaction in GenAI-powered applications using
RAG-mediated self-replicating prompts. The paper evaluates propagation across hops
and studies factors such as context size, prompt used, embedding algorithm, and
number of hops; it also proposes a guardrail.

Citation fidelity: good. The repo's contagion paper should be read as a toy
mechanistic skeleton for the type of self-replicating-prompt behavior that Cohen et
al. demonstrate in real RAG-style applications, not as another empirical
demonstration on production systems.

Source: https://arxiv.org/abs/2403.02817

### Differential cryptanalysis and Markov-chain language

Lai, Massey, and Murphy's "Markov ciphers and differential cryptanalysis" is the
right historical citation for the Markov-cipher assumption in differential
cryptanalysis. Dobrushin's ergodicity coefficient is standard Markov-chain
background for total-variation contraction.

Citation fidelity: acceptable, with an important boundary. The paper's
"cryptanalytic reading" is an analogy plus an exact toy calculation, not a claim
that transformer internals satisfy the assumptions of Markov-cipher analysis.

### De Bruijn cycles and quines

De Bruijn's 1946 combinatorial work is the right citation for de Bruijn cycles:
cyclic sequences in which each length-`k` word over an alphabet appears exactly
once. That property is exactly what the toy quine construction needs: every
length-`k` window has one successor.

Citation fidelity: good. The additional window-distinctness theorem for exact
quines is the repo's own simple graph observation, not something attributed to de
Bruijn.

### Reinforced urns and condensation

Pemantle's survey covers generalized Polya urns, reinforced random walks,
interacting urns, and continuous reinforced processes. Chung, Handjani, and
Jungreis are an appropriate citation for generalized Polya-urn behavior and
lock-in/condensation-style intuition.

Citation fidelity: mostly good. The repo's measured "condensation knee" is not a
theorem quoted from those papers; it is a toy-model phenomenon interpreted through
standard reinforced-process language.

Source: https://arxiv.org/abs/math/0610076

### Epidemic threshold language

Anderson and May's *Infectious Diseases of Humans* is an appropriate background
reference for epidemic modeling and reproduction-number language.

Citation fidelity: good if kept metaphorical. The repo's `R0` is an organizing
quantity for cross-cache propagation in the toy, not a claim that prompt contagion
follows biological epidemic dynamics.

## Omissions and competing explanations

The audit above checks whether the *cited* sources support the claims they are
attached to. It is not a survey, and it does not by itself license any novelty
claim. Two boundaries worth stating explicitly:

- **Adjacent literature this note did not engage.** The trust-vs-usefulness framing
  sits next to a large body of work on *training-time* data poisoning and backdoors
  in language models, on gradient/optimization-based prompt-injection and adversarial
  suffixes, and on planted-trigger ("sleeper agent") behavior. None of that is cited
  or surveyed here. So any statement that "no published work addresses this" should
  be read narrowly: the repo's specific contribution is the *exact, closed-form*
  decoupling of context **trust** from context **usefulness** on a Markov substrate,
  and the claim of novelty should be scoped to that exact object, not to context
  poisoning in general, until a real related-work search is done.

- **The toy result is now shown robust to the source parameters.** The order-1 vs
  order-3 split was originally demonstrated at one synthetic configuration (`V`,
  `G_CONC`, `D_CONC`, document length, Dirichlet strength) and a single seed. It is
  now guarded across seeds at a reduced configuration
  (`tests/test_core.py::TrustSaturationDichotomyTests`), swept across those source
  parameters one at a time over 3 seeds in `demo_sensitivity.py` (REVISION_PLAN item
  6, reliance dichotomy holds in all 11 configs, decoupling survives), and swept
  *jointly* over `V` × cache order in `demo_joint_sweep.py` (orders 1–4), where
  reliance-when-useless collapses onto the single density axis `V^k` — so `V` and
  order matter only through their product, not separately. The real gap that remains
  is any measurement *off* the synthetic source: "trust saturates because contexts
  are dense" is now robust to the substrate's free parameters, but it is still not a
  claim about transformers.

## Recommendations

1. Keep the current citations, but make the evidence boundary visible near the
   claims they support.
2. Prefer "modeled by a longest-match surrogate" over "precisely a longest-match
   cache" when discussing real transformer induction heads.
3. State that the cryptographic and epidemiological frames are organizing
   analogies plus exact toy calculations, not imported theorems about transformers.
4. Add future citations from the transformer-validation sibling once those results
   exist; until then, transformer claims remain conjectures.
5. Before any external "novel" framing, run a real related-work search over the
   training-time-poisoning / backdoor / adversarial-prompt literature named in
   "Omissions and competing explanations," and either cite it or scope the novelty
   claim to the exact trust-vs-usefulness object.

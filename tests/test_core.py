import os
import tempfile
import unittest

import numpy as np

import contagion as C
from demo_utils import result_path
from markov_cache import (
    BackoffModel,
    evaluate,
    evaluate_predictor,
    make_corpus,
    make_predictor,
    peaky_transition_matrix,
)


class ContagionCoreTests(unittest.TestCase):
    def test_debruijn_cycle_is_exact_minimal_order_quine(self):
        seq = C.debruijn_cycle(2, 3)
        self.assertEqual(len(seq), 8)
        self.assertTrue(C.is_deterministic(seq, 3))
        self.assertEqual(C.minimal_order(seq), 3)
        self.assertEqual(C.true_period(seq), 8)

    def test_rainbow_cycle_requires_distinct_symbols(self):
        self.assertEqual(C.rainbow_cycle([1, 2, 3]), [1, 2, 3])
        with self.assertRaises(ValueError):
            C.rainbow_cycle([1, 2, 1])

    def test_closed_form_reproduction_probabilities(self):
        self.assertAlmostEqual(C.predicted_per_step(0.8, 0.015), 0.803)
        self.assertAlmostEqual(C.predicted_run_length(0.8, 0.0), 5.0)
        self.assertEqual(C.predicted_run_length(1.0, 0.0), np.inf)

    def test_min_repetition_formulas(self):
        self.assertAlmostEqual(C.white_box_min_reps(0.25, 1.0, 10.0), 6.5)
        self.assertAlmostEqual(C.zero_knowledge_min_reps(1.0, 10.0), 9.0)


class ReproducibilityUtilityTests(unittest.TestCase):
    def test_result_path_creates_results_directory(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            try:
                path = result_path("figure.png")
                self.assertTrue(os.path.isdir("results"))
                self.assertEqual(str(path), os.path.join("results", "figure.png"))
            finally:
                os.chdir(cwd)


class CacheSubstrateTests(unittest.TestCase):
    def test_cache_reliance_is_zero_for_unseen_high_order_contexts(self):
        docs = [[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0]]
        gm = BackoffModel(2, 3).fit(docs)
        test = [[0, 2, 1, 0]]
        bits_global, _ = evaluate(
            test, gm, cache_order=3, method="global", weight=0.0,
        )
        bits_cache, _ = evaluate(
            test, gm, cache_order=3, method="dirichlet", weight=1.0,
        )
        _, _, reliance = evaluate_predictor(
            test,
            make_predictor(gm, approach="dirichlet", cache_order=3, weight=1.0),
            reliance=True,
        )
        self.assertGreaterEqual(bits_global, 0.0)
        self.assertGreaterEqual(bits_cache, 0.0)
        self.assertLess(reliance[-1], 0.5)


class TrustSaturationDichotomyTests(unittest.TestCase):
    """Fast, multi-seed guard on the *headline mechanism* of the trust article.

    The full ``demo_conservation`` sweep (and its single-seed numeric regression
    in ``tests/test_slow_regression.py``) is expensive and is skipped by default.
    This test runs a small, cheap version across several seeds so that the
    default suite exercises the paper's central claim rather than only plumbing:

      * order-1 cache (few contexts, always fills): trust *saturates* high and is
        nearly flat in ``pi_ctx`` -- handed out for quantity, not earned.
      * order-3 cache (many rare contexts): trust is *earned* -- low when there is
        no per-document structure (``pi_ctx = 0``) and rising with ``pi_ctx``.

    Asserting the qualitative dichotomy per seed addresses the single-seed
    fragility of the headline figures without pinning seed-specific magnitudes.
    """

    # Reduced config: V small enough that order-1's V contexts saturate at this
    # document length, while order-3's V**3 contexts stay mostly novel.
    V, GLOBAL_ORDER = 16, 2
    N_TRAIN, N_TEST, LEN = 40, 8, 200
    D_CONC, DIR_A, G_ALPHA = 0.05, 1.0, 0.1
    SEEDS = (1, 2, 3)

    def _final_reliance(self, *, order, pi, seed):
        rng = np.random.default_rng(seed)
        G = peaky_transition_matrix(self.V, 1.0, rng)
        train = make_corpus(G, pi, self.N_TRAIN, self.LEN, self.D_CONC, rng)
        test = make_corpus(G, pi, self.N_TEST, self.LEN, self.D_CONC, rng)
        gm = BackoffModel(self.GLOBAL_ORDER, self.V).fit(train)
        _, _, reliance = evaluate_predictor(
            test,
            make_predictor(
                gm, approach="dirichlet", cache_order=order,
                weight=self.DIR_A, g_alpha=self.G_ALPHA,
            ),
            reliance=True,
        )
        return float(reliance[-1])

    def test_order1_trust_saturates_independent_of_usefulness(self):
        for seed in self.SEEDS:
            lo = self._final_reliance(order=1, pi=0.0, seed=seed)
            hi = self._final_reliance(order=1, pi=0.9, seed=seed)
            with self.subTest(seed=seed):
                # high even with no per-document structure to learn ...
                self.assertGreater(lo, 0.85)
                # ... and barely moves as that structure appears: trust is free.
                self.assertLess(abs(hi - lo), 0.08)

    def test_order3_trust_is_earned_from_recurrence(self):
        for seed in self.SEEDS:
            lo = self._final_reliance(order=3, pi=0.0, seed=seed)
            hi = self._final_reliance(order=3, pi=0.9, seed=seed)
            with self.subTest(seed=seed):
                # near zero when nothing genuinely recurs ...
                self.assertLess(lo, 0.25)
                # ... and rises substantially once per-document structure exists.
                self.assertGreater(hi - lo, 0.2)

    def test_dense_and_sparse_caches_are_decoupled_at_pi_zero(self):
        # At pi_ctx = 0 the context is useless to both, yet the dense cache is
        # fully trusted and the sparse cache is not -- the decoupling that the
        # naive "usefulness == exploitability" law would have hidden.
        for seed in self.SEEDS:
            dense = self._final_reliance(order=1, pi=0.0, seed=seed)
            sparse = self._final_reliance(order=3, pi=0.0, seed=seed)
            with self.subTest(seed=seed):
                self.assertGreater(dense - sparse, 0.6)


if __name__ == "__main__":
    unittest.main()

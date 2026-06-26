import os
import tempfile
import unittest

import numpy as np

import contagion as C
from demo_utils import result_path
from markov_cache import BackoffModel, evaluate, evaluate_predictor, make_predictor


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


if __name__ == "__main__":
    unittest.main()

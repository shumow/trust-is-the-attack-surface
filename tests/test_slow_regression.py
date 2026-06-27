import os
import unittest


@unittest.skipUnless(
    os.environ.get("RUN_SLOW_TESTS") == "1",
    "set RUN_SLOW_TESTS=1 to run the full conservation regression",
)
class SlowConservationRegressionTests(unittest.TestCase):
    """Pins the exact headline correlation magnitudes from the full sweep.

    These numbers are specific to ``demo_conservation.SEED`` (= 7) and that
    module's configuration; they are a reproducibility lock, not a claim that the
    magnitudes are seed-invariant. The *qualitative* claim (order-1 trust
    saturates, order-3 trust is earned) is checked cheaply and across several
    seeds by ``TrustSaturationDichotomyTests`` in ``tests/test_core.py``.
    """

    def test_headline_correlations_stay_in_expected_ranges(self):
        import demo_conservation as D

        results = D.run_sweep()
        corrs = D.plot_sweep(results)
        self.assertAlmostEqual(corrs[1][0], 0.52, delta=0.08)
        self.assertAlmostEqual(corrs[3][0], 0.97, delta=0.04)
        self.assertLess(corrs[1][2], -0.9)
        self.assertLess(corrs[3][2], -0.85)


if __name__ == "__main__":
    unittest.main()

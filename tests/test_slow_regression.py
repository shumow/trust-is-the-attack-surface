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

    def test_reliance_dichotomy_is_robust_across_source_parameters(self):
        """The cheap, robust half of the sensitivity sweep: across every one-at-a-time
        source-parameter config, the dense cache is trusted when context is useless
        and the sparse cache is not. Coarse grid + 1 seed to stay tractable."""
        import numpy as np
        import demo_sensitivity as S

        pi_grid, seeds = np.array([0.0, 0.45, 0.9]), (1,)
        for label, cfg in S.build_configs():
            curves = S.measure_config(cfg, pi_grid, seeds)
            summary = S.summarize(curves)
            with self.subTest(config=label):
                self.assertGreater(summary["rel1_0"], 0.60)  # dense saturates
                self.assertLess(summary["rel3_0"], 0.30)     # sparse earns

    def test_trust_when_useless_collapses_onto_context_count(self):
        """The joint V x order sweep's headline: reliance at pi=0 is governed by the
        single density axis V**k. Decreasing in V**k, and pairs of equal V**k agree --
        the within-iso-density spread is small against the overall spread. Coarse grid
        + 1 seed to stay tractable; uses the V**k=4096 trio {8**4, 16**3, 64**2}."""
        import numpy as np
        import demo_joint_sweep as J

        v_grid, k_grid = (8, 16, 64), (2, 3, 4)
        pi_grid, seeds = np.array([0.0, 0.45, 0.9]), (1,)
        rows = J.run(v_grid, k_grid, pi_grid, seeds)
        by_nctx = {(r["V"], r["k"]): r["rel0"] for r in rows}
        # monotone: a denser cache (smaller V**k) is at least as trusted-when-useless.
        ordered = sorted(rows, key=lambda r: r["n_contexts"])
        for a, b in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(a["rel0"] + 0.08, b["rel0"])
        # collapse: the three V**k = 4096 configs agree.
        trio = [by_nctx[(8, 4)], by_nctx[(16, 3)], by_nctx[(64, 2)]]
        self.assertLess(np.std(trio), 0.10)
        within, overall = J._collapse_quality(rows)
        self.assertLess(within, 0.12)
        self.assertGreater(overall, 0.20)


if __name__ == "__main__":
    unittest.main()

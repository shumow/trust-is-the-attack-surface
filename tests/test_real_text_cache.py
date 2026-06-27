import unittest

import demo_real_text_cache as R


class RealTextCacheDemoTests(unittest.TestCase):
    def test_measure_returns_all_token_levels_and_orders(self):
        rows = R.measure()
        keys = {(r["token_level"], r["cache_order"]) for r in rows}
        expected = {
            (level, order)
            for level in R.TOKEN_LEVELS
            for order in R.CACHE_ORDERS
        }
        self.assertEqual(keys, expected)
        for row in rows:
            self.assertGreater(row["vocab"], 0)
            self.assertGreater(row["test_docs"], 0)
            self.assertGreaterEqual(row["final_reliance"], 0.0)
            self.assertLessEqual(row["final_reliance"], 1.0)


if __name__ == "__main__":
    unittest.main()

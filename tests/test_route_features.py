import unittest

from mxmoe_adapt.route_features import extract_route_features


class RouteFeatureTests(unittest.TestCase):
    def test_balanced_nested_route(self):
        features = extract_route_features([[0, 1], [2, 3]], num_experts=4)
        self.assertEqual(features.tokens, 2)
        self.assertEqual(features.top_k, 2)
        self.assertEqual(features.counts, (1, 1, 1, 1))
        self.assertAlmostEqual(features.normalized_entropy, 1.0)
        self.assertEqual(features.route_class(), "decode_sparse")

    def test_skew_and_padding(self):
        features = extract_route_features([0, 0, 0, 0, 1, 1, 2, 3], 4, top_k=2)
        self.assertEqual(features.max_tokens_per_expert, 4)
        self.assertAlmostEqual(features.padding_ratio(4), 1.0)

    def test_invalid_expert_id(self):
        with self.assertRaises(ValueError):
            extract_route_features([[0, 4]], num_experts=4)


if __name__ == "__main__":
    unittest.main()

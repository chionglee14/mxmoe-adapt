import unittest

from mxmoe_adapt.config_schema import SearchConstraints
from mxmoe_adapt.search_space import generate_search_space


class SearchSpaceTests(unittest.TestCase):
    def test_generated_candidates_respect_proxy_constraints(self):
        constraints = SearchConstraints()
        candidates = generate_search_space(constraints)
        self.assertGreater(len(candidates), 100)
        self.assertTrue(all(constraints.accepts(candidate) for candidate in candidates))
        self.assertGreater(len({candidate.align_block_size for candidate in candidates}), 1)


if __name__ == "__main__":
    unittest.main()

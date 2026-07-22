import unittest

from mxmoe_adapt.dispatch import ConfigDatabase, Workload


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "schema_version": 1,
            "entries": [
                {
                    "workload": {
                        "tokens": 4,
                        "experts": 64,
                        "hidden_size": 4096,
                        "intermediate_size": 14336,
                        "top_k": 2,
                        "dtype": "fp16",
                        "route_class": "decode_sparse",
                    },
                    "config": {
                        "BLOCK_SIZE_M": 16,
                        "BLOCK_SIZE_N": 64,
                        "BLOCK_SIZE_K": 32,
                        "GROUP_SIZE_M": 1,
                        "num_warps": 4,
                        "num_stages": 2,
                        "ALIGN_BLOCK_SIZE": 8,
                    },
                    "latency_us": 9.5,
                    "verified": True,
                    "environment_id": "c500-test",
                }
            ],
        }

    def test_selects_nearest_verified_entry(self):
        database = ConfigDatabase.from_mapping(self.payload)
        query = Workload(8, 64, 4096, 14336, 2, "fp16", "decode_sparse")
        selected = database.select(query, environment_id="c500-test")
        self.assertIsNotNone(selected)
        self.assertEqual(selected.config.block_size_m, 16)

    def test_dtype_mismatch_falls_back(self):
        database = ConfigDatabase.from_mapping(self.payload)
        query = Workload(8, 64, 4096, 14336, 2, "bf16", "decode_sparse")
        self.assertIsNone(database.select(query))

    def test_environment_mismatch_falls_back(self):
        database = ConfigDatabase.from_mapping(self.payload)
        query = Workload(4, 64, 4096, 14336, 2, "fp16", "decode_sparse")
        self.assertIsNone(database.select(query, environment_id="different-stack"))


if __name__ == "__main__":
    unittest.main()

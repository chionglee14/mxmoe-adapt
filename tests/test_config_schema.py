import unittest
from unittest.mock import patch

from mxmoe_adapt.adapters.flaggems_tuned import _config_from_environment
from mxmoe_adapt.config_schema import KernelConfig, SearchConstraints


class KernelConfigTests(unittest.TestCase):
    def test_tuned_adapter_requires_coupled_alignment_for_m0(self):
        raw = (
            '{"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, '
            '"GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, '
            '"ALIGN_BLOCK_SIZE": 8}'
        )
        with patch.dict("os.environ", {"MXMOE_KERNEL_CONFIG": raw}):
            _config_from_environment.cache_clear()
            with self.assertRaises(ValueError):
                _config_from_environment()
            _config_from_environment.cache_clear()

    def test_uppercase_mapping_and_flaggems_roundtrip(self):
        config = KernelConfig.from_mapping(
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 2,
                "ALIGN_BLOCK_SIZE": 8,
            }
        )
        self.assertEqual(config.align_block_size, 8)
        self.assertEqual(config.to_flaggems()["BLOCK_SIZE_N"], 64)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            KernelConfig(0, 64, 32, 1, 4, 2)

    def test_constraints_filter_estimated_smem(self):
        limits = SearchConstraints(shared_memory_budget_bytes=65_536)
        small = KernelConfig(16, 64, 32, 1, 4, 2)
        large = KernelConfig(128, 256, 128, 1, 4, 3)
        self.assertTrue(limits.accepts(small))
        self.assertFalse(limits.accepts(large))


if __name__ == "__main__":
    unittest.main()

import unittest

from mxmoe_adapt.config_schema import KernelConfig
from mxmoe_adapt.route_features import extract_route_features
from mxmoe_adapt.tuning import Measurement, select_best


class TuningTests(unittest.TestCase):
    def test_joint_objective_can_prefer_less_padding(self):
        route = extract_route_features([[0, 1], [0, 1], [0, 2]], num_experts=4)
        low_latency_high_padding = Measurement(
            KernelConfig(16, 64, 32, 1, 4, 2, align_block_size=16),
            latency_us=10.0,
            compile_ms=100.0,
            max_abs_error=0.0,
            verified=True,
        )
        higher_latency_low_padding = Measurement(
            KernelConfig(16, 64, 32, 1, 4, 2, align_block_size=2),
            latency_us=10.5,
            compile_ms=100.0,
            max_abs_error=0.0,
            verified=True,
        )
        selected = select_best(
            [low_latency_high_padding, higher_latency_low_padding],
            route,
            padding_penalty_us=1.0,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.measurement.config.align_block_size, 2)


if __name__ == "__main__":
    unittest.main()

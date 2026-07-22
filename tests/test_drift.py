import unittest

from mxmoe_adapt.drift import detect_drift


class DriftTests(unittest.TestCase):
    def test_environment_change_requires_retune(self):
        report = detect_drift({"environment_id": "old"}, {"environment_id": "new"})
        self.assertTrue(report.retune_required)

    def test_latency_threshold(self):
        report = detect_drift(
            {"environment_id": "same"},
            {"environment_id": "same"},
            baseline_latency_us=100.0,
            current_latency_us=107.0,
        )
        self.assertTrue(report.performance_regression)


if __name__ == "__main__":
    unittest.main()

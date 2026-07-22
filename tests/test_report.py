import json
import tempfile
import unittest
from pathlib import Path

from mxmoe_adapt.report import summarize


class ReportTests(unittest.TestCase):
    def test_geomean_excludes_failed_correctness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pass.json").write_text(
                json.dumps(
                    {
                        "status": "measured",
                        "speedup": 1.21,
                        "correctness": {"passed": True},
                        "environment": {"environment_id": "c500"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "fail.json").write_text(
                json.dumps(
                    {
                        "status": "measured",
                        "speedup": 5.0,
                        "correctness": {"passed": False},
                        "environment": {"environment_id": "c500"},
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize(root.glob("*.json"))
            self.assertEqual(summary["correctness_passed"], 1)
            self.assertAlmostEqual(summary["geomean_speedup_passing_only"], 1.21)


if __name__ == "__main__":
    unittest.main()

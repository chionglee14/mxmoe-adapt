import json
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


class ProjectIntegrityTests(unittest.TestCase):
    def test_pyproject_and_required_documents(self):
        with (ROOT / "pyproject.toml").open("rb") as handle:
            metadata = tomllib.load(handle)
        self.assertEqual(metadata["project"]["name"], "mxmoe-adapt")
        for relative in (
            "README.md",
            "LICENSE",
            "docs/项目申报书.md",
            "docs/技术方案.md",
            "docs/C500实测手册.md",
            "docs/填表短文本.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_example_json_files_are_valid_and_unverified(self):
        for path in (ROOT / "configs").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
        database = json.loads(
            (ROOT / "configs/c500-config-database.example.json").read_text(encoding="utf-8")
        )
        self.assertTrue(all(not entry["verified"] for entry in database["entries"]))


if __name__ == "__main__":
    unittest.main()

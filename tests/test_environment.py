import unittest

from mxmoe_adapt.environment import _mx_smi_identity


class EnvironmentTests(unittest.TestCase):
    def test_mx_smi_identity_excludes_volatile_metrics(self):
        first = {
            "available": True,
            "returncode": 0,
            "stdout": """mx-smi  version: 2.2.12
Timestamp: now
| MX-SMI 2.2.12 Kernel Mode Driver Version: 3.8.30 |
| MACA Version: 3.5.3.20 BIOS Version: 1.0 |
| 0 MetaX C500 | 0 Off | GPU-Util 10% | Memory 100 MiB |
""",
        }
        second = dict(first)
        second["stdout"] = first["stdout"].replace("10%", "90%").replace(
            "100 MiB", "50000 MiB"
        )
        self.assertEqual(_mx_smi_identity(first), _mx_smi_identity(second))
        self.assertEqual(_mx_smi_identity(first)["maca_version"], "3.5.3.20")


if __name__ == "__main__":
    unittest.main()

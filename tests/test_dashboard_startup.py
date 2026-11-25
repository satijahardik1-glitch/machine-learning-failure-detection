import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class DashboardStartupTests(unittest.TestCase):
    def test_overview_entrypoint_loads_without_exception(self):
        app = AppTest.from_file(str(ROOT / "app" / "main.py")).run(timeout=30)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()

import importlib.util, os, unittest
from datetime import datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sc", os.path.join(ROOT, "app", "ai", "session_context.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)


class SessionTests(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(sc.classify_session(datetime(2026, 11, 11, 3, 0))["label"], "Asian session")
        self.assertTrue(sc.classify_session(datetime(2026, 11, 11, 8, 30))["kill_zone"])
        self.assertEqual(sc.classify_session(datetime(2026, 11, 11, 13, 0))["score"], 0.9)
        self.assertEqual(sc.classify_session(datetime(2026, 11, 11, 23, 0))["score"], 0.2)

    def test_weekend_is_closed(self):
        self.assertEqual(sc.classify_session(datetime(2026, 11, 14, 10, 0))["score"], 0.0)   # Saturday

    def test_offset_converts_journal_clock_to_utc(self):
        r = sc.classify_session(datetime(2026, 11, 11, 11, 0), utc_offset_hours=3)   # 11:00 Nairobi = 08:00 UTC
        self.assertTrue(r["kill_zone"]); self.assertIn("08:00", r["utc_time"])

    def test_string_inputs_and_garbage(self):
        self.assertIsNotNone(sc.classify_session("2026-11-11T08:15"))
        self.assertIsNotNone(sc.classify_session("2026-11-11 08:15:00"))
        self.assertIsNone(sc.classify_session("not a time"))
        self.assertIsNone(sc.classify_session(None))


if __name__ == "__main__":
    unittest.main()

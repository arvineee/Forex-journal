#!/usr/bin/env python3
"""Smoke tests. Uses in-memory SQLite and never touches live user data."""
import os
import tempfile
import unittest
from datetime import datetime, timedelta

os.environ.setdefault("ENV", "testing")
os.environ.setdefault("SECRET_KEY", "test-only-secret-key")
os.environ.setdefault("COOKIE_SECURE", "false")

from app import create_app, db
from app.models import User
from app.utils.analysis import TradeAnalyzer


class TestConfig:
    TESTING = True
    ENV = "development"
    SECRET_KEY = "test-only-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    WTF_CSRF_ENABLED = False
    UPLOAD_FOLDER = tempfile.gettempdir()
    VIDEO_FOLDER = tempfile.gettempdir()
    REMEMBER_COOKIE_DURATION = timedelta(days=30)
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_SECURE = False
    SESSION_COOKIE_SECURE = False
    AI_MODEL_DIR = tempfile.gettempdir()


class ForexJournalSmokeTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(TestConfig)
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def test_database_and_password_hashing(self):
        user = User(username="tester", email="tester@example.test")
        user.set_password("A-strong-test-password")
        db.session.add(user)
        db.session.commit()
        self.assertTrue(user.check_password("A-strong-test-password"))
        self.assertEqual(User.query.count(), 1)

    def test_remember_me_sets_persistent_cookie(self):
        user = User(username="remember", email="remember@example.test")
        user.set_password("A-strong-test-password")
        db.session.add(user)
        db.session.commit()
        response = self.client.post("/auth/login", data={
            "username": "remember", "password": "A-strong-test-password",
            "remember_me": "y",
        }, follow_redirects=False)
        self.assertIn(response.status_code, (302, 303))
        cookies = response.headers.getlist("Set-Cookie")
        self.assertTrue(any("remember_token=" in c and "Expires=" in c for c in cookies))

    def test_unchecked_remember_me_does_not_issue_persistent_cookie(self):
        user = User(username="session", email="session@example.test")
        user.set_password("A-strong-test-password")
        db.session.add(user)
        db.session.commit()
        response = self.client.post("/auth/login", data={
            "username": "session", "password": "A-strong-test-password",
        }, follow_redirects=False)
        self.assertIn(response.status_code, (302, 303))
        cookies = response.headers.getlist("Set-Cookie")
        self.assertFalse(any("remember_token=" in c for c in cookies))

    def test_external_next_redirect_is_rejected(self):
        user = User(username="redirect", email="redirect@example.test")
        user.set_password("A-strong-test-password")
        db.session.add(user)
        db.session.commit()
        response = self.client.post("/auth/login?next=https://attacker.invalid/",
                                    data={"username": "redirect",
                                          "password": "A-strong-test-password"},
                                    follow_redirects=False)
        self.assertEqual(response.headers.get("Location", "").split("?")[0],
                         "/dashboard/")

    def test_timeframe_aggregation_counts_all_trades(self):
        class SampleTrade:
            def __init__(self, pnl, hour):
                self.pnl = pnl
                self.entry_time = datetime(2026, 1, 5, hour, 0)
        stats = TradeAnalyzer.analyze_by_timeframe([
            SampleTrade(10, 9), SampleTrade(-5, 9), SampleTrade(3, 10)
        ])
        self.assertEqual(stats["by_hour"][9]["trades"], 2)
        self.assertEqual(stats["by_day"]["Monday"]["trades"], 3)
        self.assertEqual(stats["by_month"]["2026-01"]["trades"], 3)

    def test_empty_metrics(self):
        metrics = TradeAnalyzer.calculate_metrics([])
        self.assertEqual(metrics["total_trades"], 0)
        self.assertEqual(metrics["win_rate"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

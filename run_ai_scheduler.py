#!/usr/bin/env python3
"""Dedicated single-instance worker for scheduled AI learning."""
import logging
from threading import Event
from app.ai.scheduler import setup_scheduler

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    scheduler = setup_scheduler()
    logging.info("AI learning worker is running; press Ctrl+C to stop.")
    try:
        Event().wait()
    except KeyboardInterrupt:
        scheduler.shutdown(wait=True)

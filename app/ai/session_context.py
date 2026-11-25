"""Classify an entry time into a forex trading session (pure Python, no Flask).

Screenshots rarely prove which session a trade was taken in, so the vision model had to leave
``session_quality`` at 0.5. When the entry time is known we can score it directly.

The scores are rules of thumb (liquidity is highest around the London open and the London/New York
overlap), NOT evidence that a session wins for you - the playbook still excludes session features
until you have enough of your own trades to test them.
"""
from datetime import datetime, timedelta

# (start_hour_utc, end_hour_utc, label, score, kill_zone)
WINDOWS = (
    (0, 7, "Asian session", 0.35, False),
    (7, 10, "London open (kill zone)", 0.90, True),
    (10, 12, "London mid-session", 0.60, False),
    (12, 15, "London/New York overlap (kill zone)", 0.90, True),
    (15, 17, "New York afternoon", 0.65, False),
    (17, 22, "Late New York", 0.35, False),
    (22, 24, "Rollover / thin liquidity", 0.20, False),
)


def parse_entry_time(value):
    """Accept a datetime or an ISO / datetime-local string; return a datetime or None."""
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value).strip().replace(" ", "T")[:19]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def classify_session(entry_time, utc_offset_hours=0):
    """Return {"label","score","kill_zone","utc_time"} for the entry time, or None if it cannot be read.

    ``utc_offset_hours`` is the offset of the times you record in your journal (Nairobi = 3).
    """
    dt = parse_entry_time(entry_time)
    if dt is None:
        return None
    try:
        utc = dt - timedelta(hours=float(utc_offset_hours or 0))
    except (TypeError, ValueError):
        utc = dt
    wd, hour = utc.weekday(), utc.hour
    closed = wd == 5 or (wd == 4 and hour >= 22) or (wd == 6 and hour < 22)
    stamp = utc.strftime("%Y-%m-%d %H:%M UTC")
    if closed:
        return {"label": "Market closed (weekend)", "score": 0.0, "kill_zone": False, "utc_time": stamp}
    for start, end, label, score, kill in WINDOWS:
        if start <= hour < end:
            return {"label": label, "score": score, "kill_zone": kill, "utc_time": stamp}
    return None

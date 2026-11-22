"""Timezone helpers. Database timestamps are treated as UTC when legacy rows are naive."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TIMEZONE = 'Africa/Nairobi'
COMMON_TIMEZONES = [
    ('Africa/Nairobi', 'Nairobi (EAT, UTC+3)'),
    ('UTC', 'UTC'),
    ('Europe/London', 'London'),
    ('Europe/Paris', 'Paris'),
    ('Europe/Berlin', 'Berlin'),
    ('America/New_York', 'New York'),
    ('America/Chicago', 'Chicago'),
    ('America/Denver', 'Denver'),
    ('America/Los_Angeles', 'Los Angeles'),
    ('America/Toronto', 'Toronto'),
    ('Asia/Dubai', 'Dubai'),
    ('Asia/Kolkata', 'India'),
    ('Asia/Singapore', 'Singapore'),
    ('Asia/Tokyo', 'Tokyo'),
    ('Australia/Sydney', 'Sydney'),
]

def valid_timezone(name):
    if not isinstance(name, str) or len(name) > 64:
        return False
    try:
        ZoneInfo(name)
        return True
    except (ZoneInfoNotFoundError, ValueError):
        return False

def as_utc(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

def to_user_timezone(value, name=None):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return value
    if not isinstance(value, datetime):
        return value
    zone = name or DEFAULT_TIMEZONE
    if not valid_timezone(zone):
        zone = DEFAULT_TIMEZONE
    return as_utc(value).astimezone(ZoneInfo(zone))

def local_to_utc_naive(value, name=None):
    """Convert a timezone-naive form datetime (interpreted as user-local) to naive UTC for DB storage."""
    if value is None:
        return None
    from datetime import timezone
    zone = name or DEFAULT_TIMEZONE
    if not valid_timezone(zone):
        zone = DEFAULT_TIMEZONE
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo(zone))
    return value.astimezone(timezone.utc).replace(tzinfo=None)

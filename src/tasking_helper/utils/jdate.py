"""
jdate — Julian Date utilities.

Conversions between calendar representations and Julian Dates, plus
epoch string parsing compatible with both ISO 8601 and NORAD TLE format.

Note on tle.py compatibility
-----------------------------
tle.py's internal _epoch_to_jd returns a Julian Date that is exactly
1.0 day too high (it adds day_of_year to midnight-of-Jan-1 instead of
subtracting 1 to account for the 1-based day count).  The functions here
are correct.  Callers that use tle.epoch_jd directly must subtract 1.0
before computing time differences; see mk_covariance.propagate().
"""

import math
from datetime import datetime, timezone

# J2000.0 epoch: noon 1 January 2000 UTC
J2000 = 2_451_545.0
_J2000_DT = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


def epoch_to_jd(year: int, day_of_year: float) -> float:
    """Year + fractional day-of-year → Julian Date.

    Uses the Fliegel & Van Flandern algorithm for the Julian Day Number
    of January 1 of *year*, then offsets by (day_of_year − 1) days from
    midnight of January 1.

    TLE convention: day_of_year = 1.0 → Jan 1 00:00:00 UTC.
    """
    a   = (14 - 1) // 12          # = 1 for month = January
    y   = year + 4800 - a
    m   = 1 + 12 * a - 3
    jdn = (1 + (153 * m + 2) // 5 + 365 * y
           + y // 4 - y // 100 + y // 400 - 32045)
    # jdn is the noon Julian Day Number for Jan 1 of year.
    # Midnight of Jan 1 = jdn − 0.5.  Day 1.0 = that midnight, so:
    return float(jdn) - 1.5 + day_of_year


def datetime_to_jd(dt: datetime) -> float:
    """UTC datetime → Julian Date."""
    return J2000 + (dt - _J2000_DT).total_seconds() / 86400.0


def jd_to_datetime(jd: float) -> datetime:
    """Julian Date → UTC datetime.

    Uses the standard calendar algorithm (Meeus, *Astronomical Algorithms*,
    Chapter 7).  Accurate to the nearest second.
    """
    z    = int(jd + 0.5)
    frac = jd + 0.5 - z
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)

    day   = b - d - int(30.6001 * e)
    month = e - 1 if e < 14 else e - 13
    year  = c - 4716 if month > 2 else c - 4715

    total_sec = frac * 86400.0
    hour, rem = divmod(int(total_sec), 3600)
    minute, second = divmod(rem, 60)
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def fmt_epoch(jd: float, fmt: str = '%Y-%m-%dT%H:%M:%S') -> str:
    """Format Julian Date as a UTC string (default: ISO 8601)."""
    return jd_to_datetime(jd).strftime(fmt)


def parse_epoch(s: str) -> float:
    """Parse an epoch string to Julian Date.

    Accepts:
      ISO 8601:   YYYY-mm-ddTHH:MM:SS[.ffffff]  or  YYYY-mm-dd
      TLE format: yyddd.ddddddd
                  (2-digit year; day 1.0 = Jan 1 00:00:00 UTC)

    Raises ValueError for unrecognised input.
    """
    # ISO 8601 variants
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',    '%Y-%m-%d'):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return datetime_to_jd(dt)
        except ValueError:
            pass

    # TLE format: yyddd.ddddddd
    try:
        dot      = s.index('.')
        yy       = int(s[:2])
        year     = 1900 + yy if yy >= 57 else 2000 + yy
        day_int  = int(s[2:dot])
        day_frac = float('0' + s[dot:])
        return epoch_to_jd(year, day_int + day_frac)
    except (ValueError, IndexError):
        pass

    raise ValueError(
        f"Cannot parse epoch {s!r}.  "
        "Expected ISO 8601 ('YYYY-mm-ddTHH:MM:SS') or "
        "TLE format ('yyddd.ddddddd')."
    )

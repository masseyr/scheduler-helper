"""Unit tests for tasking_helper.utils.jdate."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from tasking_helper.utils.jdate import (
    J2000,
    JulianDate,
    datetime_to_jd,
    epoch_to_jd,
    fmt_epoch,
    jd_to_datetime,
    parse_epoch,
)


class TestConstants:
    def test_j2000_value(self):
        assert J2000 == 2_451_545.0


class TestEpochToJd:
    def test_j2000_noon(self):
        # Jan 1 2000 noon (day 1.5) = J2000
        jd = epoch_to_jd(2000, 1.5)
        assert math.isclose(jd, J2000, abs_tol=1e-6)

    def test_jan1_midnight(self):
        # TLE day 1.0 = Jan 1 00:00 UTC; JD for that = 2451544.5
        jd = epoch_to_jd(2000, 1.0)
        assert math.isclose(jd, 2_451_544.5, abs_tol=1e-6)

    def test_fractional_day(self):
        # day 1.5 = Jan 1 12:00 UTC = J2000 reference noon
        jd_ep = epoch_to_jd(2000, 1.5)
        jd_dt = datetime_to_jd(datetime(2000, 1, 1, 12, tzinfo=timezone.utc))
        assert math.isclose(jd_ep, jd_dt, abs_tol=1e-6)

    def test_march1_non_leap(self):
        # March 1 of non-leap year 2001 = day 60
        jd = epoch_to_jd(2001, 60.0)
        expected = datetime_to_jd(datetime(2001, 3, 1, tzinfo=timezone.utc))
        assert math.isclose(jd, expected, abs_tol=1e-6)

    def test_leap_year_day366(self):
        # 2000 is a leap year; day 366 = Dec 31
        jd = epoch_to_jd(2000, 366.0)
        expected = datetime_to_jd(datetime(2000, 12, 31, tzinfo=timezone.utc))
        assert math.isclose(jd, expected, abs_tol=1e-6)

    def test_agrees_with_datetime_to_jd(self):
        # Cross-check several dates using datetime_to_jd as the reference.
        cases = [
            (2024, 1, 1),
            (2024, 7, 4),
            (2001, 9, 11),
        ]
        for year, month, day in cases:
            dt = datetime(year, month, day, tzinfo=timezone.utc)
            # day_of_year for Jan 1 = 1, so subtract 1-Jan from the date.
            day_of_year = (dt - datetime(year, 1, 1, tzinfo=timezone.utc)).days + 1
            jd_ep = epoch_to_jd(year, float(day_of_year))
            jd_dt = datetime_to_jd(dt)
            assert math.isclose(jd_ep, jd_dt, abs_tol=1e-6), (
                f"Mismatch for {year}-{month:02d}-{day:02d}: "
                f"epoch_to_jd={jd_ep}, datetime_to_jd={jd_dt}"
            )


class TestDatetimeToJd:
    def test_j2000_reference(self):
        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert math.isclose(datetime_to_jd(dt), J2000, abs_tol=1e-10)

    def test_one_day_before_j2000(self):
        dt = datetime(1999, 12, 31, 12, 0, 0, tzinfo=timezone.utc)
        assert math.isclose(datetime_to_jd(dt), J2000 - 1.0, abs_tol=1e-10)

    def test_one_day_after_j2000(self):
        dt = datetime(2000, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        assert math.isclose(datetime_to_jd(dt), J2000 + 1.0, abs_tol=1e-10)

    def test_midnight_is_half_day_before_noon(self):
        dt = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert math.isclose(datetime_to_jd(dt), J2000 - 0.5, abs_tol=1e-10)

    def test_six_hours(self):
        dt_noon = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt_18 = datetime(2000, 1, 1, 18, 0, 0, tzinfo=timezone.utc)
        assert math.isclose(
            datetime_to_jd(dt_18) - datetime_to_jd(dt_noon), 0.25, abs_tol=1e-10
        )


class TestJdToDatetime:
    def test_j2000_reference(self):
        dt = jd_to_datetime(J2000)
        assert dt == datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_midnight(self):
        dt = jd_to_datetime(J2000 - 0.5)
        assert dt == datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_timezone_is_utc(self):
        dt = jd_to_datetime(J2000)
        assert dt.tzinfo == timezone.utc

    def test_roundtrip(self):
        # jd_to_datetime is documented as accurate to the nearest second.
        original = datetime(2024, 6, 15, 8, 30, 45, tzinfo=timezone.utc)
        recovered = jd_to_datetime(datetime_to_jd(original))
        assert abs((recovered - original).total_seconds()) <= 1

    def test_roundtrip_various_dates(self):
        cases = [
            datetime(1957, 10, 4, 19, 28, 34, tzinfo=timezone.utc),  # Sputnik
            datetime(1969, 7, 20, 20, 17, 40, tzinfo=timezone.utc),  # Moon landing
            datetime(2038, 1, 19, 3, 14, 7, tzinfo=timezone.utc),    # Y2K38
        ]
        for dt in cases:
            recovered = jd_to_datetime(datetime_to_jd(dt))
            assert abs((recovered - dt).total_seconds()) <= 1

    def test_returns_datetime_instance(self):
        result = jd_to_datetime(J2000)
        assert isinstance(result, datetime)


class TestFmtEpoch:
    def test_default_iso8601(self):
        assert fmt_epoch(J2000) == "2000-01-01T12:00:00"

    def test_custom_format(self):
        assert fmt_epoch(J2000, fmt="%Y/%m/%d %H:%M") == "2000/01/01 12:00"

    def test_date_only_format(self):
        assert fmt_epoch(J2000, fmt="%Y-%m-%d") == "2000-01-01"

    def test_returns_string(self):
        assert isinstance(fmt_epoch(J2000), str)


class TestParseEpoch:
    # --- ISO 8601 variants ---

    def test_iso_datetime(self):
        jd = parse_epoch("2000-01-01T12:00:00")
        assert math.isclose(jd, J2000, abs_tol=1e-6)

    def test_iso_datetime_with_microseconds(self):
        jd = parse_epoch("2024-03-15T10:30:00.500000")
        expected = datetime_to_jd(
            datetime(2024, 3, 15, 10, 30, 0, 500_000, tzinfo=timezone.utc)
        )
        assert math.isclose(jd, expected, abs_tol=1e-6)

    def test_iso_space_separator(self):
        jd = parse_epoch("2000-01-01 12:00:00")
        assert math.isclose(jd, J2000, abs_tol=1e-6)

    def test_iso_date_only(self):
        jd = parse_epoch("2000-01-01")
        expected = datetime_to_jd(datetime(2000, 1, 1, tzinfo=timezone.utc))
        assert math.isclose(jd, expected, abs_tol=1e-6)

    # --- TLE format ---

    def test_tle_post2000(self):
        # yy=00 → year 2000; day 1.5 = noon Jan 1 = J2000
        jd = parse_epoch("00001.50000000")
        assert math.isclose(jd, J2000, abs_tol=1e-6)

    def test_tle_pre2000(self):
        # yy=99 (>=57) → year 1999; day 1.0 = midnight Jan 1 1999
        jd = parse_epoch("99001.00000000")
        expected = datetime_to_jd(datetime(1999, 1, 1, tzinfo=timezone.utc))
        assert math.isclose(jd, expected, abs_tol=1e-6)

    def test_tle_cutoff_57_maps_to_1957(self):
        jd = parse_epoch("57001.00000000")
        assert math.isclose(jd, epoch_to_jd(1957, 1.0), abs_tol=1e-6)

    def test_tle_cutoff_56_maps_to_2056(self):
        jd = parse_epoch("56001.00000000")
        assert math.isclose(jd, epoch_to_jd(2056, 1.0), abs_tol=1e-6)

    # --- Error cases ---

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Cannot parse epoch"):
            parse_epoch("not-a-date")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_epoch("")

    def test_partial_iso_raises(self):
        with pytest.raises(ValueError):
            parse_epoch("2024-13")  # invalid month

    # --- Round-trip ---

    def test_roundtrip_iso(self):
        # parse_epoch → fmt_epoch is accurate to the nearest second.
        epoch_str = "2024-06-15T08:30:45"
        result = fmt_epoch(parse_epoch(epoch_str))
        from datetime import datetime, timezone
        t1 = datetime.strptime(epoch_str, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        t2 = datetime.strptime(result, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        assert abs((t2 - t1).total_seconds()) <= 1


# ── JulianDate class ─────────────────────────────────────────────────────────

class TestJulianDateConstruction:
    def test_from_float(self):
        jd = JulianDate(J2000)
        assert jd.jd == J2000

    def test_from_datetime(self):
        dt = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
        jd = JulianDate.from_datetime(dt)
        assert math.isclose(jd.jd, J2000, abs_tol=1e-10)

    def test_from_epoch(self):
        jd = JulianDate.from_epoch(2000, 1.5)
        assert math.isclose(jd.jd, J2000, abs_tol=1e-6)

    def test_from_string_iso(self):
        jd = JulianDate.from_string("2000-01-01T12:00:00")
        assert math.isclose(jd.jd, J2000, abs_tol=1e-6)

    def test_from_string_tle(self):
        jd = JulianDate.from_string("00001.50000000")
        assert math.isclose(jd.jd, J2000, abs_tol=1e-6)


class TestJulianDateConversions:
    def test_to_datetime(self):
        jd = JulianDate(J2000)
        dt = jd.to_datetime()
        assert dt == datetime(2000, 1, 1, 12, tzinfo=timezone.utc)

    def test_to_string_default(self):
        jd = JulianDate(J2000)
        assert jd.to_string() == "2000-01-01T12:00:00"

    def test_to_string_custom_fmt(self):
        jd = JulianDate(J2000)
        assert jd.to_string("%Y/%m/%d") == "2000/01/01"

    def test_float(self):
        jd = JulianDate(J2000)
        assert float(jd) == J2000

    def test_str(self):
        jd = JulianDate(J2000)
        assert str(jd) == "2000-01-01T12:00:00"

    def test_repr(self):
        jd = JulianDate(J2000)
        assert repr(jd) == f"JulianDate({J2000})"


class TestJulianDateArithmetic:
    def setup_method(self):
        self.jd = JulianDate(J2000)

    def test_add_float(self):
        result = self.jd + 1.0
        assert isinstance(result, JulianDate)
        assert result.jd == J2000 + 1.0

    def test_add_int(self):
        result = self.jd + 10
        assert result.jd == J2000 + 10

    def test_add_timedelta(self):
        result = self.jd + timedelta(hours=12)
        assert math.isclose(result.jd, J2000 + 0.5, abs_tol=1e-10)

    def test_radd_float(self):
        result = 1.0 + self.jd
        assert isinstance(result, JulianDate)
        assert result.jd == J2000 + 1.0

    def test_sub_float(self):
        result = self.jd - 1.0
        assert isinstance(result, JulianDate)
        assert result.jd == J2000 - 1.0

    def test_sub_timedelta(self):
        result = self.jd - timedelta(days=1)
        assert math.isclose(result.jd, J2000 - 1.0, abs_tol=1e-10)

    def test_sub_julian_date_returns_float(self):
        other = JulianDate(J2000 - 1.0)
        diff = self.jd - other
        assert isinstance(diff, float)
        assert math.isclose(diff, 1.0, abs_tol=1e-10)

    def test_rsub_float(self):
        result = (J2000 + 1.0) - self.jd
        assert isinstance(result, JulianDate)
        assert math.isclose(result.jd, 1.0, abs_tol=1e-10)

    def test_add_unsupported_type_returns_not_implemented(self):
        assert self.jd.__add__("oops") is NotImplemented

    def test_sub_unsupported_type_returns_not_implemented(self):
        assert self.jd.__sub__("oops") is NotImplemented


class TestJulianDateComparison:
    def setup_method(self):
        self.a = JulianDate(J2000)
        self.b = JulianDate(J2000 + 1.0)
        self.same = JulianDate(J2000)

    def test_eq_same_value(self):
        assert self.a == self.same

    def test_eq_different_value(self):
        assert not (self.a == self.b)

    def test_ne(self):
        assert self.a != self.b

    def test_lt(self):
        assert self.a < self.b
        assert not (self.b < self.a)

    def test_le_less(self):
        assert self.a <= self.b

    def test_le_equal(self):
        assert self.a <= self.same

    def test_gt(self):
        assert self.b > self.a
        assert not (self.a > self.b)

    def test_ge_greater(self):
        assert self.b >= self.a

    def test_ge_equal(self):
        assert self.a >= self.same

    def test_eq_float(self):
        assert self.a == J2000

    def test_lt_float(self):
        assert self.a < J2000 + 1.0

    def test_gt_float(self):
        assert self.b > J2000

    def test_eq_unsupported_type(self):
        assert self.a.__eq__("nope") is NotImplemented


class TestJulianDateHashing:
    def test_hashable(self):
        jd = JulianDate(J2000)
        assert isinstance(hash(jd), int)

    def test_usable_as_dict_key(self):
        jd = JulianDate(J2000)
        d = {jd: "epoch"}
        assert d[jd] == "epoch"

    def test_usable_in_set(self):
        s = {JulianDate(J2000), JulianDate(J2000 + 1), JulianDate(J2000)}
        assert len(s) == 2

    def test_equal_objects_same_hash(self):
        a = JulianDate(J2000)
        b = JulianDate(J2000)
        assert hash(a) == hash(b)


class TestJulianDateHelpers:
    def setup_method(self):
        self.earlier = JulianDate(J2000)
        self.later = JulianDate(J2000 + 1.0)

    def test_approx_eq_within_tolerance(self):
        almost = JulianDate(J2000 + 0.5 / 86400.0)  # 0.5 s later
        assert self.earlier.approx_eq(almost, tol_seconds=1.0)

    def test_approx_eq_outside_tolerance(self):
        far = JulianDate(J2000 + 2.0 / 86400.0)  # 2 s later
        assert not self.earlier.approx_eq(far, tol_seconds=1.0)

    def test_days_since(self):
        assert math.isclose(self.later.days_since(self.earlier), 1.0, abs_tol=1e-10)

    def test_seconds_since(self):
        assert math.isclose(self.later.seconds_since(self.earlier), 86400.0, abs_tol=1e-4)

    def test_days_since_negative(self):
        assert math.isclose(self.earlier.days_since(self.later), -1.0, abs_tol=1e-10)

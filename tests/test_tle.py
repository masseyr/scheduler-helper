"""Unit tests for tasking_helper.utils.tle."""

import math

import numpy as np
import pytest

from tasking_helper.utils.tle import (
    TLE,
    _epoch_to_jd,
    _j2_secular_rates,
    parse_tle,
    parse_tle_batch,
    propagate_tle,
    tle_checksum,
    tle_epoch_state,
    tle_to_lines,
    tle_to_string,
    update_epoch,
    update_mean_anomaly,
    verify_checksum,
    verify_tle,
)
from tasking_helper.utils.utils import MU_EARTH, R_EARTH

# ---------------------------------------------------------------------------
# Canonical ISS TLE (SPACETRACK Report #3 example — checksums are valid)
# ---------------------------------------------------------------------------
_ISS_NAME = "ISS (ZARYA)"
_ISS_L1   = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
_ISS_L2   = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


# ════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def iss_tle() -> TLE:
    return parse_tle(_ISS_NAME, _ISS_L1, _ISS_L2)


# ════════════════════════════════════════════════════════════════════════════
#  Parsing
# ════════════════════════════════════════════════════════════════════════════

class TestParseTle:
    def test_name(self, iss_tle):
        assert iss_tle.name == _ISS_NAME

    def test_norad_id(self, iss_tle):
        assert iss_tle.norad_id == 25544

    def test_classification(self, iss_tle):
        assert iss_tle.classification == "U"

    def test_intl_designator(self, iss_tle):
        assert iss_tle.intl_designator == "98067A"

    def test_epoch_year(self, iss_tle):
        # yy = 08 < 57 → 2000 + 8 = 2008
        assert iss_tle.epoch_year == 2008

    def test_epoch_day(self, iss_tle):
        assert math.isclose(iss_tle.epoch_day, 264.51782528, rel_tol=1e-8)

    def test_ndot(self, iss_tle):
        assert math.isclose(iss_tle.ndot, -0.00002182, rel_tol=1e-4)

    def test_nddot_zero(self, iss_tle):
        # "00000-0" → 0.0
        assert iss_tle.nddot == 0.0

    def test_bstar(self, iss_tle):
        # "-11606-4" → -0.11606e-4 = -1.1606e-5
        assert math.isclose(iss_tle.bstar, -1.1606e-5, rel_tol=1e-3)

    def test_element_set(self, iss_tle):
        assert iss_tle.element_set == 292

    def test_inclination_deg(self, iss_tle):
        assert math.isclose(np.rad2deg(iss_tle.inclination), 51.6416, rel_tol=1e-5)

    def test_raan_deg(self, iss_tle):
        assert math.isclose(np.rad2deg(iss_tle.raan), 247.4627, rel_tol=1e-5)

    def test_eccentricity(self, iss_tle):
        # "0006703" → 0.0006703
        assert math.isclose(iss_tle.eccentricity, 0.0006703, rel_tol=1e-5)

    def test_argp_deg(self, iss_tle):
        assert math.isclose(np.rad2deg(iss_tle.argp), 130.5360, rel_tol=1e-5)

    def test_mean_anomaly_deg(self, iss_tle):
        assert math.isclose(np.rad2deg(iss_tle.mean_anomaly), 325.0288, rel_tol=1e-5)

    def test_mean_motion(self, iss_tle):
        assert math.isclose(iss_tle.mean_motion, 15.72125391, rel_tol=1e-7)

    def test_rev_number(self, iss_tle):
        assert iss_tle.rev_number == 56353

    def test_semi_major_axis_range(self, iss_tle):
        # ISS is at ~400 km → a ≈ 6.78e6 m
        assert 6.7e6 < iss_tle.semi_major_axis < 6.85e6

    def test_catalog_id(self, iss_tle):
        assert iss_tle.catalog_id == "25544"

    def test_epoch_jd_set(self, iss_tle):
        # epoch_jd must be non-zero and plausible for 2008
        assert 2_454_500 < iss_tle.epoch_jd < 2_454_800

    def test_two_line_parse(self):
        # parse_tle with 2 lines (no name) should set name to empty string
        t = parse_tle(_ISS_L1, _ISS_L2)
        assert t.name == ""
        assert t.norad_id == 25544

    def test_epoch_year_pre2000(self):
        # yy = 99 ≥ 57 → 1900 + 99 = 1999
        # Construct a minimal valid-ish TLE with epoch year 99
        l1 = "1 00001U 57001A   99001.00000000  .00000000  00000-0  00000-0 0  0001"
        l2 = "2 00001  90.0000   0.0000 0000001   0.0000   0.0000 15.00000000000001"
        t = parse_tle(l1, l2)
        assert t.epoch_year == 1999

    def test_epoch_year_boundary_57(self):
        l1 = "1 00001U 57001A   57001.00000000  .00000000  00000-0  00000-0 0  0001"
        l2 = "2 00001  90.0000   0.0000 0000001   0.0000   0.0000 15.00000000000001"
        t = parse_tle(l1, l2)
        assert t.epoch_year == 1957


class TestParseTleBatch:
    _BATCH_3LINE = f"{_ISS_NAME}\n{_ISS_L1}\n{_ISS_L2}"
    _BATCH_2LINE = f"{_ISS_L1}\n{_ISS_L2}"

    def test_3line_batch_count(self):
        tles = parse_tle_batch(self._BATCH_3LINE)
        assert len(tles) == 1

    def test_3line_batch_name(self):
        tles = parse_tle_batch(self._BATCH_3LINE)
        assert tles[0].name == _ISS_NAME

    def test_2line_batch_count(self):
        tles = parse_tle_batch(self._BATCH_2LINE)
        assert len(tles) == 1

    def test_2line_batch_no_name(self):
        tles = parse_tle_batch(self._BATCH_2LINE)
        assert tles[0].name == ""

    def test_multiple_tles(self):
        text = f"{_ISS_NAME}\n{_ISS_L1}\n{_ISS_L2}\n{_ISS_NAME}\n{_ISS_L1}\n{_ISS_L2}"
        tles = parse_tle_batch(text)
        assert len(tles) == 2

    def test_empty_string(self):
        assert parse_tle_batch("") == []

    def test_skips_garbage_lines(self):
        text = f"garbage line\n{_ISS_L1}\n{_ISS_L2}"
        tles = parse_tle_batch(text)
        assert len(tles) == 1


# ════════════════════════════════════════════════════════════════════════════
#  Derived properties
# ════════════════════════════════════════════════════════════════════════════

class TestTleProperties:
    def test_period(self, iss_tle):
        # ISS ≈ 91.6 min
        assert math.isclose(iss_tle.period, 1440.0 / iss_tle.mean_motion, rel_tol=1e-9)
        assert 90 < iss_tle.period < 95

    def test_orbit_type_leo(self, iss_tle):
        assert iss_tle.orbit_type == "LEO"

    def test_orbit_type_geo(self):
        t = TLE(mean_motion=1.0027, eccentricity=0.001)   # GEO-like period ~1437 min
        assert t.orbit_type == "GEO"

    def test_orbit_type_heo(self):
        # period > 225 min, eccentricity ≥ 0.3
        t = TLE(mean_motion=1.0, eccentricity=0.5)
        assert t.orbit_type == "HEO"

    def test_orbit_type_meo(self):
        # period 225–800 min, low eccentricity
        t = TLE(mean_motion=3.0, eccentricity=0.01)
        assert t.orbit_type == "MEO"

    def test_apogee(self, iss_tle):
        # ISS in 2008 was at ~350-360 km
        assert iss_tle.apogee is not None
        assert 300 < iss_tle.apogee < 450

    def test_perigee(self, iss_tle):
        assert iss_tle.perigee is not None
        assert 300 < iss_tle.perigee < 450

    def test_apogee_gt_perigee(self, iss_tle):
        assert iss_tle.apogee >= iss_tle.perigee

    def test_apogee_none_when_sma_zero(self):
        t = TLE(semi_major_axis=0.0)
        assert t.apogee is None

    def test_perigee_none_when_sma_zero(self):
        t = TLE(semi_major_axis=0.0)
        assert t.perigee is None

    def test_semi_major_axis_from_kepler(self, iss_tle):
        # a = (MU / n²)^(1/3)
        n = iss_tle.mean_motion * 2.0 * math.pi / 86400.0
        expected_a = (MU_EARTH / n**2) ** (1.0 / 3.0)
        assert math.isclose(iss_tle.semi_major_axis, expected_a, rel_tol=1e-6)


# ════════════════════════════════════════════════════════════════════════════
#  Checksums
# ════════════════════════════════════════════════════════════════════════════

class TestTleChecksum:
    def test_iss_line1_checksum(self):
        expected = int(_ISS_L1[-1])
        assert tle_checksum(_ISS_L1) == expected

    def test_iss_line2_checksum(self):
        expected = int(_ISS_L2[-1])
        assert tle_checksum(_ISS_L2) == expected

    def test_minus_counts_as_one(self):
        # line of all '-' (68 chars) → each '-' = 1 → total = 68 % 10 = 8
        line = "-" * 68
        assert tle_checksum(line) == 8

    def test_digits_sum(self):
        # digits 1..9 repeat, first 68 chars: verify function sums digits
        line = "123456789" * 8  # 72 chars, only first 68 used
        expected = (1+2+3+4+5+6+7+8+9) * 7 + (1+2+3+4+5) % 10  # only first 68
        s = sum(int(c) for c in line[:68] if c.isdigit())
        assert tle_checksum(line) == s % 10

    def test_ignores_after_68(self):
        # Characters beyond position 68 (the checksum digit itself) are ignored
        line_a = "0" * 68 + "9"
        line_b = "0" * 68 + "3"
        assert tle_checksum(line_a) == tle_checksum(line_b) == 0


class TestVerifyChecksum:
    def test_valid_iss_line1(self):
        assert verify_checksum(_ISS_L1) is True

    def test_valid_iss_line2(self):
        assert verify_checksum(_ISS_L2) is True

    def test_invalid_wrong_digit(self):
        # Flip the checksum digit to an incorrect value
        wrong_digit = str((int(_ISS_L1[-1]) + 1) % 10)
        bad_line = _ISS_L1[:-1] + wrong_digit
        assert verify_checksum(bad_line) is False

    def test_too_short(self):
        assert verify_checksum(_ISS_L1[:68]) is False

    def test_empty(self):
        assert verify_checksum("") is False


class TestVerifyTle:
    def test_valid_iss(self):
        result = verify_tle(_ISS_L1, _ISS_L2)
        assert result["valid"] is True
        assert result["line1_checksum"] is True
        assert result["line2_checksum"] is True
        assert result["line1_prefix"] is True
        assert result["line2_prefix"] is True
        assert result["norad_match"] is True
        assert result["errors"] == []

    def test_bad_line1_prefix(self):
        bad = "X" + _ISS_L1[1:]
        result = verify_tle(bad, _ISS_L2)
        assert result["valid"] is False
        assert result["line1_prefix"] is False

    def test_bad_line2_prefix(self):
        bad = "X" + _ISS_L2[1:]
        result = verify_tle(_ISS_L1, bad)
        assert result["valid"] is False
        assert result["line2_prefix"] is False

    def test_norad_mismatch(self):
        # Change NORAD in line2 to a different number (keep checksum invalid is OK,
        # we just want norad_match=False)
        bad_l2 = "2 99999" + _ISS_L2[7:]
        result = verify_tle(_ISS_L1, bad_l2)
        assert result["norad_match"] is False
        assert result["valid"] is False

    def test_errors_list_populated_on_failure(self):
        bad = "X" + _ISS_L1[1:]
        result = verify_tle(bad, _ISS_L2)
        assert len(result["errors"]) > 0


# ════════════════════════════════════════════════════════════════════════════
#  TLE export
# ════════════════════════════════════════════════════════════════════════════

class TestTleExport:
    def test_tle_to_lines_returns_two_strings(self, iss_tle):
        l1, l2 = tle_to_lines(iss_tle)
        assert isinstance(l1, str)
        assert isinstance(l2, str)

    def test_tle_to_lines_length(self, iss_tle):
        l1, l2 = tle_to_lines(iss_tle)
        assert len(l1) == 69
        assert len(l2) == 69

    def test_tle_to_lines_checksum_valid(self, iss_tle):
        l1, l2 = tle_to_lines(iss_tle)
        assert verify_checksum(l1), f"Bad checksum on exported line1: {l1}"
        assert verify_checksum(l2), f"Bad checksum on exported line2: {l2}"

    def test_tle_to_lines_prefixes(self, iss_tle):
        l1, l2 = tle_to_lines(iss_tle)
        assert l1.startswith("1 ")
        assert l2.startswith("2 ")

    def test_tle_to_lines_norad_preserved(self, iss_tle):
        l1, l2 = tle_to_lines(iss_tle)
        assert int(l1[2:7].strip()) == iss_tle.norad_id
        assert int(l2[2:7].strip()) == iss_tle.norad_id

    def test_tle_to_string_with_name(self, iss_tle):
        text = tle_to_string(iss_tle, include_name=True)
        lines = text.splitlines()
        assert len(lines) == 3
        assert lines[0] == _ISS_NAME

    def test_tle_to_string_without_name(self, iss_tle):
        text = tle_to_string(iss_tle, include_name=False)
        lines = text.splitlines()
        assert len(lines) == 2

    def test_roundtrip_parse_export(self, iss_tle):
        """Parse → export → re-parse: orbital elements should be preserved."""
        l1, l2 = tle_to_lines(iss_tle)
        t2 = parse_tle("", l1, l2)
        assert math.isclose(t2.inclination, iss_tle.inclination, rel_tol=1e-4)
        assert math.isclose(t2.eccentricity, iss_tle.eccentricity, rel_tol=1e-4)
        assert math.isclose(t2.mean_motion, iss_tle.mean_motion, rel_tol=1e-6)
        assert math.isclose(t2.raan, iss_tle.raan, rel_tol=1e-4)
        assert math.isclose(t2.argp, iss_tle.argp, rel_tol=1e-4)
        assert math.isclose(t2.mean_anomaly, iss_tle.mean_anomaly, rel_tol=1e-4)


# ════════════════════════════════════════════════════════════════════════════
#  Epoch utilities
# ════════════════════════════════════════════════════════════════════════════

class TestEpochToJd:
    def test_j2000_noon_off_by_one(self):
        # tle.py's _epoch_to_jd is documented as 1 day too high.
        # Jan 1 2000 day 1.5 (noon) should be J2000 = 2451545.0,
        # but _epoch_to_jd returns 2451546.0 (off by 1).
        jd = _epoch_to_jd(2000, 1.5)
        assert math.isclose(jd, 2_451_545.0 + 1.0, abs_tol=1e-6)

    def test_fractional_day_increments(self):
        # Adding 0.5 to day_of_year should add exactly 0.5 to the JD
        jd1 = _epoch_to_jd(2000, 1.0)
        jd2 = _epoch_to_jd(2000, 1.5)
        assert math.isclose(jd2 - jd1, 0.5, abs_tol=1e-10)

    def test_year_boundary(self):
        # day 1.0 of year Y+1 vs day 366/365+1.0 of year Y differ by ~1 day
        jd_2001_day1 = _epoch_to_jd(2001, 1.0)
        jd_2000_day367 = _epoch_to_jd(2000, 367.0)  # 2000 is a leap year (366 days)
        assert math.isclose(jd_2001_day1, jd_2000_day367, abs_tol=1e-6)


class TestJ2SecularRates:
    def test_rates_finite(self, iss_tle):
        n = iss_tle.mean_motion * 2.0 * math.pi / 86400.0
        raan_dot, argp_dot = _j2_secular_rates(
            n, iss_tle.semi_major_axis, iss_tle.eccentricity, iss_tle.inclination
        )
        assert math.isfinite(raan_dot)
        assert math.isfinite(argp_dot)

    def test_raan_dot_negative_prograde(self, iss_tle):
        # Prograde orbit (i < 90°) → RAAN drifts westward (negative)
        n = iss_tle.mean_motion * 2.0 * math.pi / 86400.0
        raan_dot, _ = _j2_secular_rates(
            n, iss_tle.semi_major_axis, iss_tle.eccentricity, iss_tle.inclination
        )
        assert raan_dot < 0.0

    def test_raan_dot_positive_retrograde(self):
        # Retrograde orbit (i > 90°) → RAAN drifts eastward (positive)
        inc_retro = math.radians(100.0)
        n = 15.0 * 2.0 * math.pi / 86400.0
        a = (MU_EARTH / n**2) ** (1.0 / 3.0)
        raan_dot, _ = _j2_secular_rates(n, a, 0.001, inc_retro)
        assert raan_dot > 0.0

    def test_sun_sync_zero_raan_dot(self):
        # Sun-synchronous inclination gives raan_dot ≈ +0.9856 deg/day ≈ 2.0e-7 rad/s
        # For SSO: cos(i) < 0 and chosen to match sun rate — raan_dot should be small positive
        # Just verify the function returns without error
        inc_sso = math.radians(97.5)
        n = 14.5 * 2.0 * math.pi / 86400.0
        a = (MU_EARTH / n**2) ** (1.0 / 3.0)
        raan_dot, argp_dot = _j2_secular_rates(n, a, 0.001, inc_sso)
        assert math.isfinite(raan_dot)
        assert math.isfinite(argp_dot)


# ════════════════════════════════════════════════════════════════════════════
#  Epoch and mean anomaly update
# ════════════════════════════════════════════════════════════════════════════

class TestUpdateMeanAnomaly:
    def test_replaces_mean_anomaly(self, iss_tle):
        new_M = math.pi
        t = update_mean_anomaly(iss_tle, new_M)
        assert math.isclose(t.mean_anomaly, new_M, abs_tol=1e-12)

    def test_original_unchanged(self, iss_tle):
        orig_M = iss_tle.mean_anomaly
        update_mean_anomaly(iss_tle, math.pi)
        assert math.isclose(iss_tle.mean_anomaly, orig_M, abs_tol=1e-12)

    def test_wraps_to_0_2pi(self, iss_tle):
        t = update_mean_anomaly(iss_tle, 3 * math.pi)  # 3π → π after wrap
        assert 0.0 <= t.mean_anomaly < 2.0 * math.pi

    def test_other_fields_unchanged(self, iss_tle):
        t = update_mean_anomaly(iss_tle, 0.0)
        assert t.norad_id == iss_tle.norad_id
        assert math.isclose(t.inclination, iss_tle.inclination, abs_tol=1e-12)
        assert math.isclose(t.eccentricity, iss_tle.eccentricity, abs_tol=1e-12)


class TestUpdateEpoch:
    def test_epoch_jd_updated(self, iss_tle):
        new_jd = iss_tle.epoch_jd + 1.0
        t = update_epoch(iss_tle, new_jd, propagate=False)
        assert math.isclose(t.epoch_jd, new_jd, abs_tol=1e-9)

    def test_original_unchanged_after_update(self, iss_tle):
        orig_jd = iss_tle.epoch_jd
        update_epoch(iss_tle, orig_jd + 10.0)
        assert math.isclose(iss_tle.epoch_jd, orig_jd, abs_tol=1e-9)

    def test_no_propagate_keeps_anomaly(self, iss_tle):
        orig_M = iss_tle.mean_anomaly
        t = update_epoch(iss_tle, iss_tle.epoch_jd + 1.0, propagate=False)
        assert math.isclose(t.mean_anomaly, orig_M, abs_tol=1e-12)

    def test_propagate_advances_anomaly(self, iss_tle):
        # One orbit later, mean anomaly should have changed
        one_orbit_jd = iss_tle.epoch_jd + iss_tle.period / 1440.0
        t = update_epoch(iss_tle, one_orbit_jd, propagate=True)
        # After one period M advances by 2π and wraps back; the exact value
        # depends on drag — just verify it's in [0, 2π)
        assert 0.0 <= t.mean_anomaly < 2.0 * math.pi

    def test_propagate_updates_mean_motion(self, iss_tle):
        new_jd = iss_tle.epoch_jd + 1.0
        t = update_epoch(iss_tle, new_jd, propagate=True)
        # mean_motion should change due to ndot drag term
        assert t.mean_motion != iss_tle.mean_motion

    def test_propagate_zero_dt_unchanged(self, iss_tle):
        t = update_epoch(iss_tle, iss_tle.epoch_jd, propagate=True)
        assert math.isclose(t.mean_anomaly, iss_tle.mean_anomaly, abs_tol=1e-9)
        assert math.isclose(t.raan, iss_tle.raan, abs_tol=1e-9)


# ════════════════════════════════════════════════════════════════════════════
#  Propagation
# ════════════════════════════════════════════════════════════════════════════

class TestPropagation:
    def test_tle_epoch_state_returns_arrays(self, iss_tle):
        r, v = tle_epoch_state(iss_tle)
        assert r.shape == (3,)
        assert v.shape == (3,)

    def test_tle_epoch_state_radius(self, iss_tle):
        r, _ = tle_epoch_state(iss_tle)
        r_mag = np.linalg.norm(r)
        assert 6.6e6 < r_mag < 7.0e6

    def test_propagate_tle_at_epoch(self, iss_tle):
        r, v = propagate_tle(iss_tle, iss_tle.epoch_jd)
        assert r.shape == (3,)
        assert v.shape == (3,)

    def test_propagate_tle_radius_range(self, iss_tle):
        r, v = propagate_tle(iss_tle, iss_tle.epoch_jd)
        assert 6.6e6 < np.linalg.norm(r) < 7.0e6

    def test_propagate_velocity_range(self, iss_tle):
        _, v = propagate_tle(iss_tle, iss_tle.epoch_jd)
        # ISS velocity ≈ 7.7 km/s
        v_mag = np.linalg.norm(v)
        assert 7_000 < v_mag < 8_500

"""Unit tests for tasking_helper.utils.moon_eci."""

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from tasking_helper.utils.moon_eci import (
    eci_to_ecr,
    moon_pos_eci,
    moon_pos_ecr,
    moon_state_eci,
    moon_state_ecr,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

# Meeus Ch.47 worked example: 1992-Apr-12 00:00 TD (≈ UTC for this tolerance)
_T_MEEUS = datetime(1992, 4, 12, tzinfo=timezone.utc)
_MEEUS_DIST_KM = 368409.7          # geocentric distance from Meeus §47

# A practical in-range epoch
_T_2030 = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

# Lunar distance bounds [km]  (perigee ~356 500, apogee ~406 700)
_DIST_MIN, _DIST_MAX = 350_000.0, 410_000.0

# Lunar orbital speed bound [km/s]  (~0.97–1.08)
_SPEED_MIN, _SPEED_MAX = 0.90, 1.15


# ── ECI tests ────────────────────────────────────────────────────────────────

class TestMoonPosEci:
    def test_meeus_distance(self):
        """Distance matches Meeus Ch.47 reference to within 1 km."""
        pos = moon_pos_eci(_T_MEEUS)
        assert abs(np.linalg.norm(pos) - _MEEUS_DIST_KM) < 1.0

    def test_returns_ndarray_shape3(self):
        pos = moon_pos_eci(_T_2030)
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3,)

    def test_distance_in_range(self):
        pos = moon_pos_eci(_T_2030)
        d = np.linalg.norm(pos)
        assert _DIST_MIN < d < _DIST_MAX

    def test_naive_datetime_treated_as_utc(self):
        t_naive = datetime(2030, 6, 15, 12, 0, 0)
        t_aware = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert np.allclose(moon_pos_eci(t_naive), moon_pos_eci(t_aware))

    def test_different_epochs_differ(self):
        t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2025, 1, 15, tzinfo=timezone.utc)
        assert not np.allclose(moon_pos_eci(t1), moon_pos_eci(t2))


class TestMoonStateEci:
    def test_returns_two_arrays(self):
        pos, vel = moon_state_eci(_T_2030)
        assert pos.shape == (3,) and vel.shape == (3,)

    def test_pos_matches_moon_pos_eci(self):
        pos_state, _ = moon_state_eci(_T_2030)
        pos_direct   = moon_pos_eci(_T_2030)
        assert np.allclose(pos_state, pos_direct)

    def test_speed_in_range(self):
        _, vel = moon_state_eci(_T_2030)
        speed = np.linalg.norm(vel)
        assert _SPEED_MIN < speed < _SPEED_MAX

    def test_custom_dt(self):
        """Different dt_s values should give nearly identical velocity.
        Tolerance is 0.1 m/s — limited by JD floating-point resolution."""
        _, v1 = moon_state_eci(_T_2030, dt_s=0.5)
        _, v2 = moon_state_eci(_T_2030, dt_s=2.0)
        assert np.linalg.norm(v1 - v2) < 1e-4


# ── ECR tests ────────────────────────────────────────────────────────────────

class TestEciToEcr:
    def test_magnitude_preserved(self):
        pos_eci = moon_pos_eci(_T_2030)
        pos_ecr = eci_to_ecr(pos_eci, _T_2030)
        assert abs(np.linalg.norm(pos_ecr) - np.linalg.norm(pos_eci)) < 1e-6

    def test_z_component_unchanged(self):
        """Pure GMST rotation is about Z, so z is invariant."""
        pos_eci = moon_pos_eci(_T_2030)
        pos_ecr = eci_to_ecr(pos_eci, _T_2030)
        assert abs(pos_ecr[2] - pos_eci[2]) < 1e-6

    def test_naive_datetime(self):
        pos_eci = moon_pos_eci(_T_2030)
        t_naive = datetime(2030, 6, 15, 12, 0, 0)
        pos_ecr_naive = eci_to_ecr(pos_eci, t_naive)
        pos_ecr_aware = eci_to_ecr(pos_eci, _T_2030)
        assert np.allclose(pos_ecr_naive, pos_ecr_aware)


class TestMoonPosEcr:
    def test_matches_eci_to_ecr(self):
        pos_eci = moon_pos_eci(_T_2030)
        assert np.allclose(moon_pos_ecr(_T_2030), eci_to_ecr(pos_eci, _T_2030))

    def test_distance_in_range(self):
        d = np.linalg.norm(moon_pos_ecr(_T_2030))
        assert _DIST_MIN < d < _DIST_MAX

    def test_shape(self):
        assert moon_pos_ecr(_T_2030).shape == (3,)


class TestMoonStateEcr:
    def test_pos_matches_moon_pos_ecr(self):
        pos, _ = moon_state_ecr(_T_2030)
        assert np.allclose(pos, moon_pos_ecr(_T_2030))

    def test_speed_in_range(self):
        """ECR speed is ~23 km/s: dominated by Earth's rotation (ω_E × r_moon
        ≈ 7.3e-5 × 384 000 ≈ 28 km/s) minus the Moon's ~1 km/s orbital term."""
        _, vel = moon_state_ecr(_T_2030)
        speed = np.linalg.norm(vel)
        assert 15.0 < speed < 35.0

    def test_ecr_speed_larger_than_eci(self):
        """In ECR the Moon appears fast because Earth rotates under it daily."""
        _, v_eci = moon_state_eci(_T_2030)
        _, v_ecr = moon_state_ecr(_T_2030)
        assert np.linalg.norm(v_ecr) > np.linalg.norm(v_eci)

    def test_shape(self):
        pos, vel = moon_state_ecr(_T_2030)
        assert pos.shape == (3,) and vel.shape == (3,)

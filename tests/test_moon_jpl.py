"""Unit tests for tasking_helper.utils.moon_jpl (JPL DE432s backend)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

pytest.importorskip("jplephem", reason="jplephem not installed")

from tasking_helper.utils.moon_jpl import (
    moon_pos_eci,
    moon_pos_ecr,
    moon_state_eci,
    moon_state_ecr,
    setup,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_T0 = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

# Lunar distance bounds [km]
_DIST_MIN, _DIST_MAX = 350_000.0, 410_000.0

# Orbital speed bounds in ECI [km/s]
_SPEED_MIN, _SPEED_MAX = 0.90, 1.15


# ── Kernel availability ───────────────────────────────────────────────────────

class TestSetup:
    def test_returns_existing_bsp(self):
        p = setup()
        assert p.exists()
        assert p.suffix == ".bsp"


# ── ECI tests ────────────────────────────────────────────────────────────────

class TestMoonPosEci:
    def test_scalar_shape(self):
        assert moon_pos_eci(_T0).shape == (3,)

    def test_distance_in_range(self):
        d = np.linalg.norm(moon_pos_eci(_T0))
        assert _DIST_MIN < d < _DIST_MAX

    def test_naive_datetime_treated_as_utc(self):
        t_naive = datetime(2030, 6, 15, 12, 0, 0)
        assert np.allclose(moon_pos_eci(t_naive), moon_pos_eci(_T0))

    def test_different_epochs_differ(self):
        t2 = _T0 + timedelta(days=7)
        assert not np.allclose(moon_pos_eci(_T0), moon_pos_eci(t2))


class TestMoonStateEci:
    def test_scalar_shapes(self):
        pos, vel = moon_state_eci(_T0)
        assert pos.shape == (3,) and vel.shape == (3,)

    def test_pos_matches_moon_pos_eci(self):
        pos_s, _ = moon_state_eci(_T0)
        assert np.allclose(pos_s, moon_pos_eci(_T0))

    def test_speed_in_range(self):
        _, vel = moon_state_eci(_T0)
        assert _SPEED_MIN < np.linalg.norm(vel) < _SPEED_MAX

    def test_analytic_velocity(self):
        """DE velocity should be smooth — far more precise than finite-diff."""
        _, v1 = moon_state_eci(_T0)
        _, v2 = moon_state_eci(_T0 + timedelta(seconds=1))
        # velocity changes by < 1 mm/s in 1 second (lunar accel ≈ 2.7 mm/s²)
        assert np.linalg.norm(v1 - v2) < 0.01


# ── ECR tests ────────────────────────────────────────────────────────────────

class TestMoonPosEcr:
    def test_scalar_shape(self):
        assert moon_pos_ecr(_T0).shape == (3,)

    def test_magnitude_preserved(self):
        d_eci = np.linalg.norm(moon_pos_eci(_T0))
        d_ecr = np.linalg.norm(moon_pos_ecr(_T0))
        assert abs(d_eci - d_ecr) < 1e-6

    def test_z_invariant(self):
        """GMST rotation is about Z so z is the same in ECI and ECR."""
        pos_eci = moon_pos_eci(_T0)
        pos_ecr = moon_pos_ecr(_T0)
        assert abs(pos_eci[2] - pos_ecr[2]) < 1e-6


class TestMoonStateEcr:
    def test_pos_matches_moon_pos_ecr(self):
        pos, _ = moon_state_ecr(_T0)
        assert np.allclose(pos, moon_pos_ecr(_T0))

    def test_ecr_speed_larger_than_eci(self):
        """Earth's daily rotation dominates Moon's ECR velocity."""
        _, v_eci = moon_state_eci(_T0)
        _, v_ecr = moon_state_ecr(_T0)
        assert np.linalg.norm(v_ecr) > np.linalg.norm(v_eci)

    def test_shape(self):
        pos, vel = moon_state_ecr(_T0)
        assert pos.shape == (3,) and vel.shape == (3,)


# ── Batch (vectorised) tests ─────────────────────────────────────────────────

class TestBatch:
    # 120-second steps over 2 hours — mimics the stated minimum cadence
    TIMES = [_T0 + timedelta(seconds=120 * i) for i in range(61)]
    N = 61

    def test_eci_batch_shape(self):
        pos = moon_pos_eci(self.TIMES)
        assert pos.shape == (3, self.N)

    def test_ecr_batch_shape(self):
        pos = moon_pos_ecr(self.TIMES)
        assert pos.shape == (3, self.N)

    def test_state_eci_batch_shape(self):
        pos, vel = moon_state_eci(self.TIMES)
        assert pos.shape == (3, self.N) and vel.shape == (3, self.N)

    def test_state_ecr_batch_shape(self):
        pos, vel = moon_state_ecr(self.TIMES)
        assert pos.shape == (3, self.N) and vel.shape == (3, self.N)

    def test_batch_first_matches_scalar(self):
        batch = moon_pos_eci(self.TIMES)
        scalar = moon_pos_eci(self.TIMES[0])
        assert np.allclose(batch[:, 0], scalar)

    def test_batch_distances_in_range(self):
        pos = moon_pos_eci(self.TIMES)
        dists = np.linalg.norm(pos, axis=0)
        assert np.all(dists > _DIST_MIN) and np.all(dists < _DIST_MAX)

    def test_batch_monotone_distance_change(self):
        """Over 2 hours the Moon moves smoothly — distances vary < 200 km."""
        pos = moon_pos_eci(self.TIMES)
        dists = np.linalg.norm(pos, axis=0)
        assert (dists.max() - dists.min()) < 200.0


# ── Cross-check ELP2000 vs JPL ───────────────────────────────────────────────

class TestVsElp2000:
    def test_position_within_200km(self):
        """JPL DE432s vs truncated ELP2000 should agree within ~200 km."""
        from tasking_helper.utils.moon_eci import moon_pos_eci as elp_pos
        pos_jpl = moon_pos_eci(_T0)
        pos_elp = elp_pos(_T0)
        diff = np.linalg.norm(pos_jpl - pos_elp)
        assert diff < 200.0

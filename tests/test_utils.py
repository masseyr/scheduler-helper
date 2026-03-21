"""Unit tests for tasking_helper.utils.utils."""

import math

import numpy as np
import pytest

from tasking_helper.utils.utils import (
    E2_EARTH,
    F_EARTH,
    J2,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH,
    ecef_to_eci,
    ecef_to_lla,
    eci_to_ecef,
    gmst,
    julian_date,
    lla_to_ecef,
    normalize,
    rotation_matrix_axis_angle,
)

# J2000 noon 1-Jan-2000 UTC
_J2000_JD = 2_451_545.0


# ════════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_mu_earth(self):
        assert math.isclose(MU_EARTH, 3.986004418e14, rel_tol=1e-6)

    def test_r_earth(self):
        # WGS-84 semi-major axis [m]
        assert math.isclose(R_EARTH, 6_378_137.0, rel_tol=1e-8)

    def test_omega_earth(self):
        assert math.isclose(OMEGA_EARTH, 7.2921150e-5, rel_tol=1e-6)

    def test_j2(self):
        assert math.isclose(J2, 1.08263e-3, rel_tol=1e-4)

    def test_e2_earth_derived(self):
        # E2 = 2F - F²
        expected = 2 * F_EARTH - F_EARTH**2
        assert math.isclose(E2_EARTH, expected, rel_tol=1e-10)


# ════════════════════════════════════════════════════════════════════════════
#  normalize
# ════════════════════════════════════════════════════════════════════════════

class TestNormalize:
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_allclose(result, v, atol=1e-15)

    def test_magnitude_is_one(self):
        v = np.array([3.0, 4.0, 0.0])
        result = normalize(v)
        assert math.isclose(np.linalg.norm(result), 1.0, abs_tol=1e-15)

    def test_direction_preserved(self):
        v = np.array([1.0, 2.0, 3.0])
        u = normalize(v)
        # u should be parallel to v (cross product zero)
        cross = np.cross(u, v / np.linalg.norm(v))
        np.testing.assert_allclose(cross, np.zeros(3), atol=1e-14)

    def test_batch_array(self):
        v = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
        result = normalize(v)
        assert result.shape == (2, 3)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-15)

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError):
            normalize(np.array([0.0, 0.0, 0.0]))

    def test_zero_row_in_batch_raises(self):
        with pytest.raises(ValueError):
            normalize(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    def test_3d_array_raises(self):
        with pytest.raises(ValueError):
            normalize(np.zeros((2, 3, 3)))


# ════════════════════════════════════════════════════════════════════════════
#  rotation_matrix_axis_angle
# ════════════════════════════════════════════════════════════════════════════

class TestRotationMatrixAxisAngle:
    def test_zero_angle_is_identity(self):
        R = rotation_matrix_axis_angle([0, 0, 1], 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_90deg_z_rotates_x_to_y(self):
        R = rotation_matrix_axis_angle([0, 0, 1], math.pi / 2)
        result = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-14)

    def test_90deg_x_rotates_y_to_z(self):
        R = rotation_matrix_axis_angle([1, 0, 0], math.pi / 2)
        result = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-14)

    def test_180deg_flips(self):
        R = rotation_matrix_axis_angle([0, 0, 1], math.pi)
        result = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [-1.0, 0.0, 0.0], atol=1e-14)

    def test_is_orthogonal(self):
        R = rotation_matrix_axis_angle([1, 2, 3], 1.234)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)

    def test_det_is_one(self):
        R = rotation_matrix_axis_angle([1, 1, 1], math.pi / 3)
        assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-14)

    def test_axis_along_rotation_unchanged(self):
        axis = np.array([0.0, 0.0, 1.0])
        R = rotation_matrix_axis_angle(axis, math.pi / 4)
        result = R @ axis
        np.testing.assert_allclose(result, axis, atol=1e-14)

    def test_unnormalized_axis_accepted(self):
        # The function normalizes the axis internally
        R1 = rotation_matrix_axis_angle([0, 0, 2], math.pi / 2)
        R2 = rotation_matrix_axis_angle([0, 0, 1], math.pi / 2)
        np.testing.assert_allclose(R1, R2, atol=1e-14)


# ════════════════════════════════════════════════════════════════════════════
#  julian_date
# ════════════════════════════════════════════════════════════════════════════

class TestJulianDate:
    def test_j2000_noon(self):
        jd = julian_date(2000, 1, 1, 12.0, 0.0, 0.0)
        assert math.isclose(jd, _J2000_JD, abs_tol=1e-8)

    def test_j2000_midnight(self):
        jd = julian_date(2000, 1, 1, 0.0, 0.0, 0.0)
        assert math.isclose(jd, _J2000_JD - 0.5, abs_tol=1e-8)

    def test_one_day_step(self):
        jd1 = julian_date(2000, 1, 1, 12.0)
        jd2 = julian_date(2000, 1, 2, 12.0)
        assert math.isclose(jd2 - jd1, 1.0, abs_tol=1e-8)

    def test_one_hour_step(self):
        jd1 = julian_date(2000, 1, 1, 0.0)
        jd2 = julian_date(2000, 1, 1, 1.0)
        assert math.isclose(jd2 - jd1, 1.0 / 24.0, abs_tol=1e-9)

    def test_february_march_boundary(self):
        jd_feb28 = julian_date(2001, 2, 28)
        jd_mar1  = julian_date(2001, 3, 1)
        assert math.isclose(jd_mar1 - jd_feb28, 1.0, abs_tol=1e-8)

    def test_leap_year_feb29(self):
        # 2000 is a leap year: Feb 29 exists
        jd_feb29 = julian_date(2000, 2, 29)
        jd_mar1  = julian_date(2000, 3, 1)
        assert math.isclose(jd_mar1 - jd_feb29, 1.0, abs_tol=1e-8)

    def test_returns_float(self):
        assert isinstance(julian_date(2000, 1, 1), float)


# ════════════════════════════════════════════════════════════════════════════
#  gmst
# ════════════════════════════════════════════════════════════════════════════

class TestGmst:
    def test_returns_float(self):
        assert isinstance(gmst(_J2000_JD), float)

    def test_in_range_0_2pi(self):
        theta = gmst(_J2000_JD)
        assert 0.0 <= theta < 2.0 * math.pi

    def test_j2000_approximate_value(self):
        # GMST at J2000 noon ≈ 18.697374558 hours = 4.894961213 rad
        theta = gmst(_J2000_JD)
        assert math.isclose(theta, 4.894961213, abs_tol=1e-3)

    def test_one_sidereal_day(self):
        # One sidereal day ≈ 86164.1 s → JD step ≈ 0.99726958 days
        sidereal_day_jd = 86164.1 / 86400.0
        theta1 = gmst(_J2000_JD)
        theta2 = gmst(_J2000_JD + sidereal_day_jd)
        diff = (theta2 - theta1) % (2.0 * math.pi)
        assert math.isclose(diff, 0.0, abs_tol=1e-3)


# ════════════════════════════════════════════════════════════════════════════
#  ECI ↔ ECEF
# ════════════════════════════════════════════════════════════════════════════

class TestEciEcef:
    def test_roundtrip_single(self):
        r_eci = np.array([R_EARTH + 400e3, 0.0, 0.0])
        r_ecef = eci_to_ecef(r_eci, _J2000_JD)
        r_eci_back = ecef_to_eci(r_ecef, _J2000_JD)
        np.testing.assert_allclose(r_eci_back, r_eci, rtol=1e-12, atol=1e-6)

    def test_roundtrip_batch(self):
        r_eci = np.array([
            [R_EARTH + 400e3, 0.0, 0.0],
            [0.0, R_EARTH + 400e3, 0.0],
        ])
        r_ecef = eci_to_ecef(r_eci, _J2000_JD)
        r_back  = ecef_to_eci(r_ecef, _J2000_JD)
        np.testing.assert_allclose(r_back, r_eci, rtol=1e-12, atol=1e-6)

    def test_magnitude_preserved(self):
        r_eci = np.array([R_EARTH + 400e3, 100e3, 200e3])
        r_ecef = eci_to_ecef(r_eci, _J2000_JD)
        assert math.isclose(
            np.linalg.norm(r_ecef), np.linalg.norm(r_eci), rel_tol=1e-12
        )

    def test_z_component_unchanged(self):
        # Z axis is Earth's rotation axis → unaffected by GMST rotation
        r_eci = np.array([0.0, 0.0, R_EARTH])
        r_ecef = eci_to_ecef(r_eci, _J2000_JD)
        assert math.isclose(r_ecef[2], R_EARTH, rel_tol=1e-12)

    def test_eci_to_ecef_output_shape_1d(self):
        r = np.array([R_EARTH, 0.0, 0.0])
        result = eci_to_ecef(r, _J2000_JD)
        assert result.shape == (3,)

    def test_eci_to_ecef_output_shape_2d(self):
        r = np.zeros((5, 3))
        r[:, 0] = R_EARTH
        result = eci_to_ecef(r, _J2000_JD)
        assert result.shape == (5, 3)


# ════════════════════════════════════════════════════════════════════════════
#  ECEF ↔ LLA
# ════════════════════════════════════════════════════════════════════════════

class TestEcefLla:
    def test_equator_prime_meridian(self):
        # lat=0, lon=0, alt=0 → ECEF = [R_EARTH, 0, 0]
        r = lla_to_ecef(0.0, 0.0, 0.0)
        np.testing.assert_allclose(r, [R_EARTH, 0.0, 0.0], rtol=1e-9)

    def test_equator_90deg_east(self):
        # lat=0, lon=90°, alt=0 → ECEF = [0, R_EARTH, 0]
        r = lla_to_ecef(0.0, math.pi / 2, 0.0)
        np.testing.assert_allclose(r, [0.0, R_EARTH, 0.0], rtol=1e-9, atol=1e-6)

    def test_roundtrip_equator(self):
        lat, lon, alt = 0.0, 0.0, 0.0
        r = lla_to_ecef(lat, lon, alt)
        lla = ecef_to_lla(r)
        assert math.isclose(lla[0], lat, abs_tol=1e-8)
        assert math.isclose(lla[1], lon, abs_tol=1e-8)
        assert math.isclose(lla[2], alt, abs_tol=1e-2)

    def test_roundtrip_mid_latitude(self):
        lat = math.radians(45.0)
        lon = math.radians(-75.0)
        alt = 500_000.0
        r = lla_to_ecef(lat, lon, alt)
        lla = ecef_to_lla(r)
        assert math.isclose(lla[0], lat, abs_tol=1e-8)
        assert math.isclose(lla[1], lon, abs_tol=1e-8)
        assert math.isclose(lla[2], alt, abs_tol=1e-2)

    def test_roundtrip_high_latitude(self):
        lat = math.radians(70.0)
        lon = math.radians(25.0)
        alt = 0.0
        r = lla_to_ecef(lat, lon, alt)
        lla = ecef_to_lla(r)
        assert math.isclose(lla[0], lat, abs_tol=1e-7)
        assert math.isclose(lla[1], lon, abs_tol=1e-7)
        assert math.isclose(lla[2], alt, abs_tol=1.0)  # 1 m tolerance at surface

    def test_altitude_at_leo(self):
        # Position at 400 km above equator
        alt_in = 400_000.0
        r = lla_to_ecef(0.0, 0.0, alt_in)
        lla = ecef_to_lla(r)
        assert math.isclose(lla[2], alt_in, rel_tol=1e-6)

    def test_batch_roundtrip(self):
        lats = np.deg2rad([0.0, 30.0, -45.0])
        lons = np.deg2rad([0.0, 90.0, 180.0])
        alts = [0.0, 100e3, 400e3]
        ecefs = np.array([lla_to_ecef(la, lo, a)
                          for la, lo, a in zip(lats, lons, alts)])
        llas = ecef_to_lla(ecefs)
        np.testing.assert_allclose(llas[:, 0], lats, atol=1e-7)
        np.testing.assert_allclose(llas[:, 1], lons, atol=1e-7)
        np.testing.assert_allclose(llas[:, 2], alts, rtol=1e-5)

    def test_lla_to_ecef_output_shape(self):
        r = lla_to_ecef(0.0, 0.0, 0.0)
        assert r.shape == (3,)

    def test_ecef_to_lla_output_shape_1d(self):
        r = np.array([R_EARTH, 0.0, 0.0])
        lla = ecef_to_lla(r)
        assert lla.shape == (3,)

    def test_ecef_to_lla_output_shape_2d(self):
        r = np.tile([R_EARTH, 0.0, 0.0], (4, 1))
        lla = ecef_to_lla(r)
        assert lla.shape == (4, 3)

    def test_longitude_range(self):
        # Longitude should be in (-π, π]
        for lon_deg in [-180, -90, 0, 90, 179]:
            r = lla_to_ecef(0.0, math.radians(lon_deg), 0.0)
            lla = ecef_to_lla(r)
            assert -math.pi <= lla[1] <= math.pi

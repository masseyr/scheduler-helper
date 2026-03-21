"""Unit tests for tasking_helper.utils.satellite."""

import math
import warnings

import numpy as np
import pytest

from tasking_helper.utils.tle import parse_tle
from tasking_helper.utils.satellite import Satellite
from tasking_helper.utils.utils import MU_EARTH, R_EARTH

# ---------------------------------------------------------------------------
# Canonical ISS TLE (same as test_tle.py)
# ---------------------------------------------------------------------------
_ISS_NAME = "ISS (ZARYA)"
_ISS_L1   = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
_ISS_L2   = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Circular ISS-like orbit parameters used for from_lat_lon_alt tests
_ISS_MM   = 15.72        # rev/day (approximate)
_ISS_INC  = 51.6         # deg
_ISS_ALT  = 410_000.0    # m (≈ 410 km, consistent with the mean motion above)


# ════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def iss_sat() -> Satellite:
    tle = parse_tle(_ISS_NAME, _ISS_L1, _ISS_L2)
    return Satellite.from_tle(tle)


# ════════════════════════════════════════════════════════════════════════════
#  Construction
# ════════════════════════════════════════════════════════════════════════════

class TestConstruction:
    def test_from_tle(self):
        tle = parse_tle(_ISS_NAME, _ISS_L1, _ISS_L2)
        sat = Satellite.from_tle(tle)
        assert isinstance(sat, Satellite)

    def test_direct_construction(self):
        tle = parse_tle(_ISS_NAME, _ISS_L1, _ISS_L2)
        sat = Satellite(tle=tle)
        assert isinstance(sat, Satellite)

    def test_from_lat_lon_alt_mean_motion(self):
        jd = 2_451_545.0
        sat = Satellite.from_lat_lon_alt(
            lat=0.0, lon=0.0, alt=_ISS_ALT,
            jd=jd,
            inclination=_ISS_INC,
            mean_motion=_ISS_MM,
        )
        assert isinstance(sat, Satellite)

    def test_from_lat_lon_alt_semi_major_axis(self):
        jd = 2_451_545.0
        n = _ISS_MM * 2 * math.pi / 86400.0
        a = (MU_EARTH / n**2) ** (1.0 / 3.0)
        sat = Satellite.from_lat_lon_alt(
            lat=0.0, lon=0.0, alt=a - R_EARTH,
            jd=jd,
            inclination=_ISS_INC,
            semi_major_axis=a,
        )
        assert isinstance(sat, Satellite)

    def test_from_lat_lon_alt_requires_exactly_one_motion_param(self):
        jd = 2_451_545.0
        with pytest.raises(ValueError, match="exactly one"):
            Satellite.from_lat_lon_alt(
                lat=0.0, lon=0.0, alt=_ISS_ALT, jd=jd, inclination=_ISS_INC,
                mean_motion=_ISS_MM, semi_major_axis=6_780_000.0,  # both provided
            )
        with pytest.raises(ValueError, match="exactly one"):
            Satellite.from_lat_lon_alt(
                lat=0.0, lon=0.0, alt=_ISS_ALT, jd=jd, inclination=_ISS_INC,
                # neither provided
            )

    def test_from_lat_lon_alt_zero_inclination_raises(self):
        with pytest.raises(ValueError, match="Inclination too close"):
            Satellite.from_lat_lon_alt(
                lat=0.0, lon=0.0, alt=_ISS_ALT, jd=2_451_545.0,
                inclination=0.0, mean_motion=_ISS_MM,
            )

    def test_from_lat_lon_alt_mean_motion_preserved(self):
        jd = 2_451_545.0
        sat = Satellite.from_lat_lon_alt(
            lat=0.0, lon=0.0, alt=_ISS_ALT,
            jd=jd, inclination=_ISS_INC, mean_motion=_ISS_MM,
        )
        assert math.isclose(sat.mean_motion, _ISS_MM, rel_tol=1e-6)

    def test_from_lat_lon_alt_inclination_preserved(self):
        jd = 2_451_545.0
        sat = Satellite.from_lat_lon_alt(
            lat=0.0, lon=0.0, alt=_ISS_ALT,
            jd=jd, inclination=_ISS_INC, mean_motion=_ISS_MM,
        )
        assert math.isclose(np.rad2deg(sat.inclination), _ISS_INC, rel_tol=1e-6)

    def test_from_lat_lon_alt_epoch_jd_preserved(self):
        jd = 2_451_545.0
        sat = Satellite.from_lat_lon_alt(
            lat=0.0, lon=0.0, alt=_ISS_ALT,
            jd=jd, inclination=_ISS_INC, mean_motion=_ISS_MM,
        )
        assert math.isclose(sat.epoch_jd, jd, abs_tol=1e-9)

    def test_from_lat_lon_alt_warns_on_inconsistent_alt(self):
        # Provide an altitude that doesn't match the mean_motion → UserWarning
        jd = 2_451_545.0
        with pytest.warns(UserWarning, match="differs from orbital shape"):
            Satellite.from_lat_lon_alt(
                lat=0.0, lon=0.0, alt=_ISS_ALT + 200_000,  # 200 km too high
                jd=jd, inclination=_ISS_INC, mean_motion=_ISS_MM,
            )

    def test_from_lat_lon_alt_name_and_norad(self):
        jd = 2_451_545.0
        sat = Satellite.from_lat_lon_alt(
            lat=0.0, lon=0.0, alt=_ISS_ALT,
            jd=jd, inclination=_ISS_INC, mean_motion=_ISS_MM,
            name="TEST", norad_id=99999,
        )
        assert sat.name == "TEST"
        assert sat.norad_id == 99999


# ════════════════════════════════════════════════════════════════════════════
#  Metadata properties
# ════════════════════════════════════════════════════════════════════════════

class TestMetadataProperties:
    def test_name(self, iss_sat):
        assert iss_sat.name == _ISS_NAME

    def test_norad_id(self, iss_sat):
        assert iss_sat.norad_id == 25544

    def test_catalog_id(self, iss_sat):
        assert iss_sat.catalog_id == "25544"

    def test_epoch_jd(self, iss_sat):
        assert 2_454_500 < iss_sat.epoch_jd < 2_454_800


# ════════════════════════════════════════════════════════════════════════════
#  Keplerian element properties
# ════════════════════════════════════════════════════════════════════════════

class TestKeplerianProperties:
    def test_semi_major_axis(self, iss_sat):
        assert 6.7e6 < iss_sat.semi_major_axis < 6.85e6

    def test_eccentricity(self, iss_sat):
        assert math.isclose(iss_sat.eccentricity, 0.0006703, rel_tol=1e-4)

    def test_inclination(self, iss_sat):
        assert math.isclose(np.rad2deg(iss_sat.inclination), 51.6416, rel_tol=1e-4)

    def test_raan(self, iss_sat):
        assert math.isclose(np.rad2deg(iss_sat.raan), 247.4627, rel_tol=1e-4)

    def test_argp(self, iss_sat):
        assert math.isclose(np.rad2deg(iss_sat.argp), 130.5360, rel_tol=1e-4)

    def test_mean_anomaly(self, iss_sat):
        assert math.isclose(np.rad2deg(iss_sat.mean_anomaly), 325.0288, rel_tol=1e-4)

    def test_mean_motion(self, iss_sat):
        assert math.isclose(iss_sat.mean_motion, 15.72125391, rel_tol=1e-6)

    def test_period(self, iss_sat):
        # ISS period ≈ 91.6 min
        assert 90 < iss_sat.period < 95

    def test_apogee(self, iss_sat):
        # ISS in 2008 was at ~350-360 km
        assert 300 < iss_sat.apogee < 450

    def test_perigee(self, iss_sat):
        assert 300 < iss_sat.perigee < 450

    def test_orbit_type_leo(self, iss_sat):
        assert iss_sat.orbit_type == "LEO"

    def test_period_mean_motion_consistent(self, iss_sat):
        assert math.isclose(iss_sat.period, 1440.0 / iss_sat.mean_motion, rel_tol=1e-9)


# ════════════════════════════════════════════════════════════════════════════
#  repr
# ════════════════════════════════════════════════════════════════════════════

class TestRepr:
    def test_repr_contains_name(self, iss_sat):
        assert _ISS_NAME in repr(iss_sat)

    def test_repr_contains_norad(self, iss_sat):
        assert "25544" in repr(iss_sat)

    def test_repr_contains_orbit_type(self, iss_sat):
        assert "LEO" in repr(iss_sat)

    def test_repr_is_string(self, iss_sat):
        assert isinstance(repr(iss_sat), str)


# ════════════════════════════════════════════════════════════════════════════
#  Propagation methods
# ════════════════════════════════════════════════════════════════════════════

class TestPropagation:
    def test_state_eci_returns_arrays(self, iss_sat):
        r, v = iss_sat.state_eci(iss_sat.epoch_jd)
        assert r.shape == (3,)
        assert v.shape == (3,)

    def test_state_eci_radius(self, iss_sat):
        r, _ = iss_sat.state_eci(iss_sat.epoch_jd)
        assert 6.6e6 < np.linalg.norm(r) < 7.0e6

    def test_state_ecef_returns_arrays(self, iss_sat):
        r, v = iss_sat.state_ecef(iss_sat.epoch_jd)
        assert r.shape == (3,)
        assert v.shape == (3,)

    def test_eci_position_shape(self, iss_sat):
        r = iss_sat.eci_position(iss_sat.epoch_jd)
        assert r.shape == (3,)

    def test_ecef_position_shape(self, iss_sat):
        r = iss_sat.ecef_position(iss_sat.epoch_jd)
        assert r.shape == (3,)

    def test_ecef_velocity_shape(self, iss_sat):
        v = iss_sat.ecef_velocity(iss_sat.epoch_jd)
        assert v.shape == (3,)

    def test_lat_lon_alt_returns_three_floats(self, iss_sat):
        lat, lon, alt = iss_sat.lat_lon_alt(iss_sat.epoch_jd)
        assert isinstance(lat, float)
        assert isinstance(lon, float)
        assert isinstance(alt, float)

    def test_lat_lon_alt_altitude_range(self, iss_sat):
        _, _, alt = iss_sat.lat_lon_alt(iss_sat.epoch_jd)
        assert 350_000 < alt < 450_000

    def test_lat_range(self, iss_sat):
        lat, _, _ = iss_sat.lat_lon_alt(iss_sat.epoch_jd)
        assert -90 <= lat <= 90

    def test_lon_range(self, iss_sat):
        _, lon, _ = iss_sat.lat_lon_alt(iss_sat.epoch_jd)
        assert -180 <= lon <= 180

    def test_ecef_eci_magnitude_equal(self, iss_sat):
        r_eci = iss_sat.eci_position(iss_sat.epoch_jd)
        r_ecef = iss_sat.ecef_position(iss_sat.epoch_jd)
        assert math.isclose(
            np.linalg.norm(r_eci), np.linalg.norm(r_ecef), rel_tol=1e-10
        )

"""
coverage.py -- Ground sensor visibility and access-interval analysis.

Given a satellite ephemeris (ECI J2000 positions vs UTC time) and one or
more ground sensors, identifies access intervals: contiguous windows when
the object is within the sensor's maximum slant range and above its minimum
elevation angle.

Units:  km for distances,  degrees for angles and geodetic coordinates.
"""

import math
from datetime import datetime, timezone
from typing import List, Sequence

import numpy as np

from .keplerian import lla_to_ecr, eci_to_ecr


# ---------------------------------------------------------------------------
# GroundSensor
# ---------------------------------------------------------------------------

class GroundSensor:
    """
    Ground-based tracking sensor defined by geodetic position and limits.

    Parameters
    ----------
    name              : sensor label
    lat_deg           : geodetic latitude  [deg]  -90 ... +90
    lon_deg           : longitude          [deg]  -180 ... +360
    alt_km            : altitude above WGS-84 ellipsoid [km]
    max_range_km      : maximum slant range for tracking [km]
    min_elevation_deg : minimum elevation angle above local horizon [deg]
                        (default 5.0)
    """

    def __init__(
        self,
        name: str,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
        max_range_km: float,
        min_elevation_deg: float = 5.0,
    ):
        if max_range_km <= 0:
            raise ValueError("max_range_km must be positive")
        if not -90.0 <= lat_deg <= 90.0:
            raise ValueError(f"lat_deg out of range: {lat_deg}")

        self.name = name
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.alt_km = float(alt_km)
        self.max_range_km = float(max_range_km)
        self.min_elevation_deg = float(min_elevation_deg)

        # Static ECEF position (sensor does not move)
        self._ecef_km: np.ndarray = lla_to_ecr(lat_deg, lon_deg, alt_km)

        # Geodetic normal (local "up") in ECEF frame
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        self._up: np.ndarray = np.array([
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ])

    # -- properties ---------------------------------------------------------

    @property
    def ecef_km(self) -> np.ndarray:
        """Sensor ECEF position [km]."""
        return self._ecef_km

    # -- geometry -----------------------------------------------------------

    def elevation_and_range(self, obj_ecef_km: np.ndarray):
        """
        Return (elevation_deg, range_km) from sensor to object.

        elevation_deg is measured above the local horizontal plane;
        negative values indicate the object is below the horizon.
        """
        dr = np.asarray(obj_ecef_km) - self._ecef_km
        range_km = float(np.linalg.norm(dr))
        if range_km < 1e-9:
            return 90.0, 0.0
        sin_el = float(np.dot(dr, self._up)) / range_km
        sin_el = max(-1.0, min(1.0, sin_el))   # clamp for float safety
        el_deg = math.degrees(math.asin(sin_el))
        return el_deg, range_km

    def in_view(self, obj_ecef_km: np.ndarray):
        """
        Return (visible, elevation_deg, range_km).

        visible is True when range <= max_range_km AND elevation >= min_elevation_deg.
        """
        el, rng = self.elevation_and_range(obj_ecef_km)
        visible = (rng <= self.max_range_km) and (el >= self.min_elevation_deg)
        return visible, el, rng

    # -- display ------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"Sensor : {self.name}\n"
            f"  Location  : lat={self.lat_deg:.4f} deg  lon={self.lon_deg:.4f} deg"
            f"  alt={self.alt_km*1e3:.0f} m\n"
            f"  Max range : {self.max_range_km:.0f} km\n"
            f"  Min elev  : {self.min_elevation_deg:.1f} deg"
        )

    def __repr__(self) -> str:
        return (
            f"GroundSensor({self.name!r}, lat={self.lat_deg:.3f}, "
            f"lon={self.lon_deg:.3f}, alt={self.alt_km:.3f} km, "
            f"max_range={self.max_range_km:.0f} km, "
            f"min_el={self.min_elevation_deg:.1f} deg)"
        )


# ---------------------------------------------------------------------------
# AccessInterval
# ---------------------------------------------------------------------------

class AccessInterval:
    """
    Contiguous window during which an object is visible from a ground sensor.

    Attributes
    ----------
    sensor_name       : name of the GroundSensor
    object_id         : label of the tracked object
    start_time        : UTC datetime of first visible epoch
    end_time          : UTC datetime of last visible epoch
    duration_s        : window length in seconds  (end - start)
    min_range_km      : closest approach during window [km]
    max_elevation_deg : peak elevation angle during window [deg]
    peak_time         : UTC datetime of peak elevation
    """

    def __init__(
        self,
        sensor_name: str,
        object_id: str,
        start_time: datetime,
        end_time: datetime,
        duration_s: float,
        min_range_km: float,
        max_elevation_deg: float,
        peak_time: datetime,
    ):
        self.sensor_name = sensor_name
        self.object_id = object_id
        self.start_time = start_time
        self.end_time = end_time
        self.duration_s = float(duration_s)
        self.min_range_km = float(min_range_km)
        self.max_elevation_deg = float(max_elevation_deg)
        self.peak_time = peak_time

    def to_dict(self) -> dict:
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        return {
            "sensor":           self.sensor_name,
            "object":           self.object_id,
            "start_utc":        self.start_time.strftime(fmt),
            "end_utc":          self.end_time.strftime(fmt),
            "duration_s":       round(self.duration_s, 1),
            "min_range_km":     round(self.min_range_km, 3),
            "max_elevation_deg":round(self.max_elevation_deg, 3),
            "peak_time_utc":    self.peak_time.strftime(fmt),
        }

    def __str__(self) -> str:
        return (
            f"{self.sensor_name} -> {self.object_id}:  "
            f"{self.start_time.strftime('%Y-%m-%dT%H:%M:%SZ')} -- "
            f"{self.end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}  "
            f"dur={self.duration_s:.0f}s  "
            f"min_rng={self.min_range_km:.1f} km  "
            f"max_el={self.max_elevation_deg:.1f} deg"
        )

    def __repr__(self) -> str:
        return (
            f"AccessInterval({self.sensor_name!r}, {self.object_id!r}, "
            f"dur={self.duration_s:.0f}s)"
        )


# ---------------------------------------------------------------------------
# Access computation
# ---------------------------------------------------------------------------

def compute_access(
    times_utc: Sequence[datetime],
    pos_eci_km: np.ndarray,
    sensor: GroundSensor,
    object_id: str = "object",
) -> List[AccessInterval]:
    """
    Compute access intervals for one sensor vs one object ephemeris.

    Parameters
    ----------
    times_utc  : N UTC datetimes (any cadence)
    pos_eci_km : (N, 3)  ECI J2000 positions [km]
    sensor     : GroundSensor
    object_id  : string label for the tracked object

    Returns
    -------
    list of AccessInterval sorted by start_time
    """
    pos_eci_km = np.asarray(pos_eci_km)
    N = len(times_utc)
    if N == 0:
        return []

    # Convert all ECI positions to ECEF at each epoch
    ecef = np.empty((N, 3))
    for i, t in enumerate(times_utc):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        ecef[i] = eci_to_ecr(pos_eci_km[i], t)

    # Evaluate visibility, elevation, and range at each epoch
    visible   = np.empty(N, dtype=bool)
    elevations = np.empty(N)
    ranges     = np.empty(N)
    for i in range(N):
        v, el, rng = sensor.in_view(ecef[i])
        visible[i]    = v
        elevations[i] = el
        ranges[i]     = rng

    # Stitch contiguous visible epochs into AccessInterval objects
    intervals: List[AccessInterval] = []
    in_access = False
    i_start = 0

    for i in range(N):
        if visible[i] and not in_access:
            in_access = True
            i_start = i
        elif not visible[i] and in_access:
            _build_interval(intervals, sensor, object_id,
                            times_utc, elevations, ranges, i_start, i - 1)
            in_access = False

    if in_access:   # still in-access at end of ephemeris
        _build_interval(intervals, sensor, object_id,
                        times_utc, elevations, ranges, i_start, N - 1)

    return intervals


def compute_access_table(
    times_utc: Sequence[datetime],
    pos_eci_km: np.ndarray,
    sensors: Sequence[GroundSensor],
    object_id: str = "object",
) -> List[AccessInterval]:
    """
    Compute access intervals for multiple sensors vs one object.

    Returns a flat list of AccessInterval sorted by start_time.
    Suitable for feeding into a multi-sensor scheduler.
    """
    all_intervals: List[AccessInterval] = []
    for sensor in sensors:
        all_intervals.extend(
            compute_access(times_utc, pos_eci_km, sensor, object_id)
        )
    all_intervals.sort(key=lambda a: a.start_time)
    return all_intervals


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build_interval(
    intervals: list,
    sensor: GroundSensor,
    object_id: str,
    times_utc: Sequence[datetime],
    elevations: np.ndarray,
    ranges: np.ndarray,
    i0: int,
    i1: int,
) -> None:
    seg_el  = elevations[i0:i1 + 1]
    seg_rng = ranges[i0:i1 + 1]
    peak_off = int(np.argmax(seg_el))
    t0 = times_utc[i0]
    t1 = times_utc[i1]
    intervals.append(AccessInterval(
        sensor_name=sensor.name,
        object_id=object_id,
        start_time=t0,
        end_time=t1,
        duration_s=(t1 - t0).total_seconds(),
        min_range_km=float(np.min(seg_rng)),
        max_elevation_deg=float(np.max(seg_el)),
        peak_time=times_utc[i0 + peak_off],
    ))

"""
coverage.py -- Ground sensor visibility and access-interval analysis.

Given a satellite ephemeris (ECI J2000 positions vs UTC time) and one or
more ground sensors, identifies access intervals: contiguous windows when
the object is within the sensor's maximum slant range and above its minimum
elevation angle.

EphemerisInterpolator fits an order-8 Lagrange polynomial (9-point window)
keyed on UTC epoch (seconds since the first point) so positions can be
evaluated at any time within the span.  For each query, the 9 nearest sample
points are selected so the local polynomial adapts across the full arc.

When an interpolator is passed to compute_access, window start/end times are
refined by bisection to the requested tolerance (default 0.1 s).

Units:  km for distances,  degrees for angles and geodetic coordinates.
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

import numpy as np

from .keplerian import lla_to_ecr, eci_to_ecr


# ---------------------------------------------------------------------------
# Lagrange interpolation (pure numpy, no scipy dependency)
# ---------------------------------------------------------------------------

def _lagrange_eval(xs: np.ndarray, ys: np.ndarray, x: float) -> np.ndarray:
    """
    Evaluate a Lagrange polynomial at scalar x using all n supplied nodes.

    xs : (n,)    x-coordinates of the nodes
    ys : (n,) or (n, k)  function values at the nodes
    x  : scalar query point

    Uses the barycentric form for numerical stability.
    """
    n = len(xs)
    dx = x - xs                       # (n,)  differences
    exact = np.where(dx == 0.0)[0]
    if len(exact):
        return ys[exact[0]].copy()

    # Barycentric weights: w_i = 1 / prod_{j!=i}(x_i - x_j)
    w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if j != i:
                w[i] *= (xs[i] - xs[j])
    w = 1.0 / w

    # Barycentric formula
    terms = (w / dx)                  # (n,)
    if ys.ndim == 1:
        return float(np.dot(terms, ys) / terms.sum())
    return (terms[:, None] * ys).sum(axis=0) / terms.sum()


def _lagrange_window(
    xs_all: np.ndarray,
    ys_all: np.ndarray,
    x: float,
    n_pts: int,
) -> np.ndarray:
    """
    Select n_pts nodes centred on x, then evaluate the Lagrange polynomial.

    Clamps the window at the array boundaries so edge queries still work.
    """
    idx = int(np.searchsorted(xs_all, x))
    half = n_pts // 2
    i0 = max(0, min(idx - half, len(xs_all) - n_pts))
    i1 = i0 + n_pts
    return _lagrange_eval(xs_all[i0:i1], ys_all[i0:i1], x)


# ---------------------------------------------------------------------------
# EphemerisInterpolator
# ---------------------------------------------------------------------------

class EphemerisInterpolator:
    """
    Lagrange polynomial interpolator for an ECI J2000 position ephemeris.

    For each query the nearest n_points (default 9, giving an 8th-order
    polynomial) samples are used.  This matches the approach used by SPICE
    and most astrodynamics toolkits for satellite ephemerides.

    The x-axis is seconds since the first ephemeris epoch, so the interpolator
    is independent of absolute time format.

    Parameters
    ----------
    times_utc  : N UTC datetimes (must be strictly increasing)
    pos_eci_km : (N, 3)  ECI J2000 positions [km]
    n_points   : number of nodes per Lagrange window (default 9, i.e. order 8)
    """

    def __init__(
        self,
        times_utc: Sequence[datetime],
        pos_eci_km: np.ndarray,
        n_points: int = 9,
    ):
        times_utc  = list(times_utc)
        pos_eci_km = np.asarray(pos_eci_km, dtype=float)

        if len(times_utc) < 2:
            raise ValueError("At least 2 points are required for interpolation.")
        if pos_eci_km.shape != (len(times_utc), 3):
            raise ValueError(
                f"pos_eci_km shape {pos_eci_km.shape} does not match "
                f"{len(times_utc)} times."
            )
        if n_points < 2:
            raise ValueError("n_points must be >= 2.")
        if n_points > len(times_utc):
            raise ValueError(
                f"n_points ({n_points}) cannot exceed number of epochs ({len(times_utc)})."
            )

        self._t0    = times_utc[0]
        self._t_end = times_utc[-1]
        self._xs    = np.array(
            [(t - self._t0).total_seconds() for t in times_utc]
        )
        self._ys    = pos_eci_km
        self._n_pts = n_points

    # -- properties ---------------------------------------------------------

    @property
    def t_start(self) -> datetime:
        """UTC time of the first ephemeris point."""
        return self._t0

    @property
    def t_end(self) -> datetime:
        """UTC time of the last ephemeris point."""
        return self._t_end

    @property
    def span_s(self) -> float:
        """Total span of the ephemeris in seconds."""
        return float(self._xs[-1])

    # -- evaluation ---------------------------------------------------------

    def position(self, t: datetime) -> np.ndarray:
        """
        Interpolate ECI position [km] at time t (UTC datetime).

        t must lie within [t_start, t_end].
        """
        s = (t - self._t0).total_seconds()
        return _lagrange_window(self._xs, self._ys, s, self._n_pts)

    def position_array(self, times: Sequence[datetime]) -> np.ndarray:
        """
        Interpolate ECI positions [km] at a sequence of UTC datetimes.

        Returns (M, 3) ndarray.
        """
        return np.array([self.position(t) for t in times])

    def resample(self, dt_s: float = 1.0):
        """
        Generate a uniformly-spaced ephemeris at cadence dt_s seconds.

        Returns
        -------
        times_utc  : list of datetime
        pos_eci_km : (M, 3) ndarray
        """
        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")
        ss = np.arange(0.0, self._xs[-1] + dt_s * 0.5, dt_s)
        ss = ss[ss <= self._xs[-1]]
        times = [self._t0 + timedelta(seconds=float(s)) for s in ss]
        pos   = np.array([
            _lagrange_window(self._xs, self._ys, s, self._n_pts) for s in ss
        ])
        return times, pos

    def __repr__(self) -> str:
        n    = len(self._xs)
        dt   = float(self._xs[1] - self._xs[0]) if n > 1 else 0.0
        return (
            f"EphemerisInterpolator(Lagrange order {self._n_pts - 1}, "
            f"{n} points, span={self.span_s:.0f}s, dt_nominal={dt:.1f}s)"
        )


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

        self._ecef_km: np.ndarray = lla_to_ecr(lat_deg, lon_deg, alt_km)

        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        self._up: np.ndarray = np.array([
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ])

    @property
    def ecef_km(self) -> np.ndarray:
        return self._ecef_km

    def elevation_and_range(self, obj_ecef_km: np.ndarray):
        """Return (elevation_deg, range_km) from sensor to object."""
        dr = np.asarray(obj_ecef_km) - self._ecef_km
        range_km = float(np.linalg.norm(dr))
        if range_km < 1e-9:
            return 90.0, 0.0
        sin_el = float(np.dot(dr, self._up)) / range_km
        sin_el = max(-1.0, min(1.0, sin_el))
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
    start_time        : UTC datetime of window start (refined if interpolator used)
    end_time          : UTC datetime of window end   (refined if interpolator used)
    duration_s        : window length in seconds
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
            "sensor":            self.sensor_name,
            "object":            self.object_id,
            "start_utc":         self.start_time.strftime(fmt),
            "end_utc":           self.end_time.strftime(fmt),
            "duration_s":        round(self.duration_s, 3),
            "min_range_km":      round(self.min_range_km, 3),
            "max_elevation_deg": round(self.max_elevation_deg, 3),
            "peak_time_utc":     self.peak_time.strftime(fmt),
        }

    def __str__(self) -> str:
        return (
            f"{self.sensor_name} -> {self.object_id}:  "
            f"{self.start_time.strftime('%Y-%m-%dT%H:%M:%SZ')} -- "
            f"{self.end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}  "
            f"dur={self.duration_s:.1f}s  "
            f"min_rng={self.min_range_km:.1f} km  "
            f"max_el={self.max_elevation_deg:.1f} deg"
        )

    def __repr__(self) -> str:
        return (
            f"AccessInterval({self.sensor_name!r}, {self.object_id!r}, "
            f"dur={self.duration_s:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Access computation
# ---------------------------------------------------------------------------

def compute_access(
    times_utc: Sequence[datetime],
    pos_eci_km: np.ndarray,
    sensor: GroundSensor,
    object_id: str = "object",
    interpolator: Optional[EphemerisInterpolator] = None,
    refine_tol_s: float = 0.1,
) -> List[AccessInterval]:
    """
    Compute access intervals for one sensor vs one object ephemeris.

    Parameters
    ----------
    times_utc     : N UTC datetimes (any cadence)
    pos_eci_km    : (N, 3)  ECI J2000 positions [km]
    sensor        : GroundSensor
    object_id     : string label for the tracked object
    interpolator  : EphemerisInterpolator, optional.
                    When provided, window start/end times are refined by
                    bisection to within refine_tol_s seconds and peak
                    elevation is computed on the interpolated arc.
    refine_tol_s  : bisection stopping tolerance in seconds (default 0.1)

    Returns
    -------
    list of AccessInterval sorted by start_time
    """
    pos_eci_km = np.asarray(pos_eci_km)
    N = len(times_utc)
    if N == 0:
        return []

    times_utc = [
        t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)
        for t in times_utc
    ]

    # Convert ECI -> ECEF at each sample epoch
    ecef = np.empty((N, 3))
    for i, t in enumerate(times_utc):
        ecef[i] = eci_to_ecr(pos_eci_km[i], t)

    # Visibility, elevation, range at each sample
    visible    = np.empty(N, dtype=bool)
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
            _build_interval(
                intervals, sensor, object_id,
                times_utc, elevations, ranges, i_start, i - 1,
                interpolator=interpolator, refine_tol_s=refine_tol_s,
                t_before=times_utc[i_start - 1] if i_start > 0 else None,
                t_after=times_utc[i],
            )
            in_access = False

    if in_access:
        _build_interval(
            intervals, sensor, object_id,
            times_utc, elevations, ranges, i_start, N - 1,
            interpolator=interpolator, refine_tol_s=refine_tol_s,
            t_before=times_utc[i_start - 1] if i_start > 0 else None,
            t_after=None,
        )

    return intervals


def compute_access_table(
    times_utc: Sequence[datetime],
    pos_eci_km: np.ndarray,
    sensors: Sequence[GroundSensor],
    object_id: str = "object",
    interpolator: Optional[EphemerisInterpolator] = None,
    refine_tol_s: float = 0.1,
) -> List[AccessInterval]:
    """
    Compute access intervals for multiple sensors vs one object.

    Returns a flat list of AccessInterval sorted by start_time.
    Pass an EphemerisInterpolator to enable sub-sample boundary refinement.
    """
    all_intervals: List[AccessInterval] = []
    for sensor in sensors:
        all_intervals.extend(
            compute_access(
                times_utc, pos_eci_km, sensor, object_id,
                interpolator=interpolator, refine_tol_s=refine_tol_s,
            )
        )
    all_intervals.sort(key=lambda a: a.start_time)
    return all_intervals


# ---------------------------------------------------------------------------
# Internal helpers
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
    interpolator: Optional[EphemerisInterpolator],
    refine_tol_s: float,
    t_before: Optional[datetime],
    t_after: Optional[datetime],
) -> None:
    t0 = times_utc[i0]
    t1 = times_utc[i1]

    if interpolator is not None:
        # Refine boundaries via bisection on the interpolated arc
        if t_before is not None:
            t0 = _bisect_crossing(interpolator, sensor, t_before, t0,
                                  target_visible=True, tol_s=refine_tol_s)
        if t_after is not None:
            t1 = _bisect_crossing(interpolator, sensor, t1, t_after,
                                  target_visible=False, tol_s=refine_tol_s)

        # Re-evaluate on a finer grid clipped to the refined window
        dt_s     = (times_utc[i1] - times_utc[i0]).total_seconds() / max(i1 - i0, 1)
        fine_dt  = max(1.0, dt_s / 10.0)
        fine_times_all, fine_pos_all = interpolator.resample(dt_s=fine_dt)

        # Filter to refined window, reusing positions from resample
        indices = [k for k, t in enumerate(fine_times_all) if t0 <= t <= t1]
        if indices:
            fine_times   = [fine_times_all[k] for k in indices]
            fine_pos_arr = fine_pos_all[indices]
            fine_el, fine_rng = [], []
            for j, ft in enumerate(fine_times):
                fe = eci_to_ecr(fine_pos_arr[j], ft)
                _, el, rng = sensor.in_view(fe)
                fine_el.append(el)
                fine_rng.append(rng)
            peak_off = int(np.argmax(fine_el))
            max_el   = float(fine_el[peak_off])
            min_rng  = float(np.min(fine_rng))
            peak_t   = fine_times[peak_off]
        else:
            seg_el   = elevations[i0:i1 + 1]
            seg_rng  = ranges[i0:i1 + 1]
            peak_off = int(np.argmax(seg_el))
            max_el   = float(np.max(seg_el))
            min_rng  = float(np.min(seg_rng))
            peak_t   = times_utc[i0 + peak_off]
    else:
        seg_el   = elevations[i0:i1 + 1]
        seg_rng  = ranges[i0:i1 + 1]
        peak_off = int(np.argmax(seg_el))
        max_el   = float(np.max(seg_el))
        min_rng  = float(np.min(seg_rng))
        peak_t   = times_utc[i0 + peak_off]

    intervals.append(AccessInterval(
        sensor_name=sensor.name,
        object_id=object_id,
        start_time=t0,
        end_time=t1,
        duration_s=(t1 - t0).total_seconds(),
        min_range_km=min_rng,
        max_elevation_deg=max_el,
        peak_time=peak_t,
    ))


def _bisect_crossing(
    interpolator: EphemerisInterpolator,
    sensor: GroundSensor,
    t_a: datetime,
    t_b: datetime,
    target_visible: bool,
    tol_s: float = 0.1,
) -> datetime:
    """
    Binary search for the exact moment visibility transitions to target_visible.

    t_a : last epoch where visible == (not target_visible)
    t_b : first epoch where visible == target_visible
    Returns the refined crossing time (converged to tol_s seconds).
    """
    while (t_b - t_a).total_seconds() > tol_s:
        t_mid = t_a + timedelta(seconds=(t_b - t_a).total_seconds() / 2.0)
        pos_mid  = interpolator.position(t_mid)
        ecef_mid = eci_to_ecr(pos_mid, t_mid)
        v, _, _  = sensor.in_view(ecef_mid)
        if v == target_visible:
            t_b = t_mid
        else:
            t_a = t_mid
    return t_b

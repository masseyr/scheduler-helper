"""
stk_ephem.py -- STK ephemeris (.e) file reader with ECI/ECR interpolation.

Reads the standard STK text ephemeris format (stk.v.10.0 / stk.v.11.0),
stores raw ECI J2000 state vectors, and provides order-8 Lagrange
interpolation so that position and velocity can be queried at any epoch
within the arc.

Both ECI (native) and ECR (on-the-fly via GMST rotation) frames are
supported.  The ECR velocity includes the Earth-rotation correction:
    v_ecr = R(GMST) * v_eci + [omega_E * y_ecr, -omega_E * x_ecr, 0]

Supported STK data keywords:
    EphemerisTimePosVel  -- time, x, y, z, vx, vy, vz  (7 columns)
    EphemerisTimePos     -- time, x, y, z               (4 columns, no velocity)

Supported coordinate systems:
    J2000, ICRF  -- treated as ECI J2000 (converted to ECR on demand)
    Fixed        -- already ECR; ECR queries returned directly, ECI via inverse

Units: km for positions, km/s for velocities.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .coverage import _lagrange_window
from .jdate import datetime_to_jd
from .utils import gmst as _gmst, OMEGA_EARTH


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_MONTH = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,  "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# CoordinateSystems that are ECI (J2000-equivalent for our purposes)
_ECI_SYSTEMS = {"j2000", "icrf", "meanofepoch", "trueofepoch",
                "meanofdate", "trueofdate", "teme"}
_ECR_SYSTEMS = {"fixed", "ecf", "ecef"}


def _parse_epoch(s: str) -> datetime:
    """Parse STK epoch string: '1 Jun 2025 12:00:00.000' -> UTC datetime."""
    parts = s.strip().split()
    if len(parts) != 4:
        raise ValueError(f"Cannot parse STK epoch: {s!r}")
    day   = int(parts[0])
    month = _MONTH.get(parts[1].lower())
    if month is None:
        raise ValueError(f"Unknown month: {parts[1]!r}")
    year = int(parts[2])
    hms  = parts[3]
    fmt  = "%H:%M:%S.%f" if "." in hms else "%H:%M:%S"
    tt   = datetime.strptime(hms, fmt)
    return datetime(year, month, day, tt.hour, tt.minute, tt.second,
                    tt.microsecond, tzinfo=timezone.utc)


def _rotation_eci_to_ecr(t: datetime) -> np.ndarray:
    """3x3 rotation matrix ECI -> ECR at epoch t (GMST rotation)."""
    jd = datetime_to_jd(t)
    th = _gmst(jd)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c,  s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def _vel_eci_to_ecr(
    R: np.ndarray,
    v_eci: np.ndarray,
    r_ecr: np.ndarray,
) -> np.ndarray:
    """
    Convert ECI velocity to ECR velocity given the rotation matrix R and
    already-converted ECR position.

    v_ecr = R * v_eci + [omega_E * y_ecr, -omega_E * x_ecr, 0]
    """
    v = R @ v_eci
    v[0] += OMEGA_EARTH * r_ecr[1]
    v[1] -= OMEGA_EARTH * r_ecr[0]
    return v


# ---------------------------------------------------------------------------
# StkEphemeris
# ---------------------------------------------------------------------------

class StkEphemeris:
    """
    STK ephemeris with Lagrange interpolation in ECI and ECR frames.

    Do not construct directly -- use StkEphemeris.load(path).

    Attributes
    ----------
    t_start        : UTC datetime of first data point
    t_end          : UTC datetime of last data point
    span_s         : arc length in seconds
    n_points       : number of raw data points
    has_velocity   : True if the file contained velocity columns
    coordinate_system : CoordinateSystem field from the file header
    metadata       : dict of all header key-value pairs
    """

    def __init__(
        self,
        times_utc: list,
        pos_km: np.ndarray,
        vel_kms: Optional[np.ndarray],
        is_eci: bool,
        metadata: dict,
        n_interp: int = 9,
    ):
        if len(times_utc) < 2:
            raise ValueError("Ephemeris must have at least 2 points.")

        self._times   = times_utc
        self._pos     = np.asarray(pos_km,  dtype=float)
        self._vel     = np.asarray(vel_kms, dtype=float) if vel_kms is not None else None
        self._is_eci  = is_eci
        self.metadata = metadata
        self._n       = n_interp

        # x-axis for interpolation: seconds since first epoch
        t0 = times_utc[0]
        self._xs = np.array([(t - t0).total_seconds() for t in times_utc])
        self._t0 = t0

    # -- public properties --------------------------------------------------

    @property
    def t_start(self) -> datetime:
        return self._times[0]

    @property
    def t_end(self) -> datetime:
        return self._times[-1]

    @property
    def span_s(self) -> float:
        return float(self._xs[-1])

    @property
    def n_points(self) -> int:
        return len(self._times)

    @property
    def has_velocity(self) -> bool:
        return self._vel is not None

    @property
    def coordinate_system(self) -> str:
        return self.metadata.get("CoordinateSystem", "unknown")

    # -- internal interpolation ---------------------------------------------

    def _interp_pos(self, s: float) -> np.ndarray:
        return _lagrange_window(self._xs, self._pos, s, self._n)

    def _interp_vel(self, s: float) -> np.ndarray:
        if self._vel is None:
            raise RuntimeError(
                "Velocity not available: file contained EphemerisTimePos only."
            )
        return _lagrange_window(self._xs, self._vel, s, self._n)

    def _s(self, t: datetime) -> float:
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (t - self._t0).total_seconds()

    # -- ECI queries --------------------------------------------------------

    def position_eci(self, t: datetime) -> np.ndarray:
        """Interpolated ECI position [km] at t."""
        pos = self._interp_pos(self._s(t))
        if self._is_eci:
            return pos
        # Fixed -> ECI: inverse rotation
        return _rotation_eci_to_ecr(t).T @ pos

    def velocity_eci(self, t: datetime) -> np.ndarray:
        """Interpolated ECI velocity [km/s] at t."""
        vel = self._interp_vel(self._s(t))
        if self._is_eci:
            return vel
        # Fixed -> ECI: inverse rotation + Coriolis correction
        R   = _rotation_eci_to_ecr(t)
        r_ecr = self._interp_pos(self._s(t))
        # v_eci = R^T * v_ecr - R^T * [omega x r_ecr]  -- inverse of vel formula
        corr = np.array([OMEGA_EARTH * r_ecr[1], -OMEGA_EARTH * r_ecr[0], 0.0])
        return R.T @ (vel - corr)

    def state_eci(self, t: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolated (position [km], velocity [km/s]) in ECI at t."""
        return self.position_eci(t), self.velocity_eci(t)

    # -- ECR queries --------------------------------------------------------

    def position_ecr(self, t: datetime) -> np.ndarray:
        """Interpolated ECR position [km] at t."""
        if not self._is_eci:
            return self._interp_pos(self._s(t))
        R = _rotation_eci_to_ecr(t)
        return R @ self._interp_pos(self._s(t))

    def velocity_ecr(self, t: datetime) -> np.ndarray:
        """Interpolated ECR velocity [km/s] at t."""
        s   = self._s(t)
        R   = _rotation_eci_to_ecr(t)
        if not self._is_eci:
            return self._interp_vel(s)
        r_ecr = R @ self._interp_pos(s)
        return _vel_eci_to_ecr(R, self._interp_vel(s), r_ecr)

    def state_ecr(self, t: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolated (position [km], velocity [km/s]) in ECR at t."""
        s   = self._s(t)
        R   = _rotation_eci_to_ecr(t)
        if not self._is_eci:
            return self._interp_pos(s), self._interp_vel(s)
        r_ecr = R @ self._interp_pos(s)
        v_ecr = _vel_eci_to_ecr(R, self._interp_vel(s), r_ecr)
        return r_ecr, v_ecr

    # -- bulk resampling ----------------------------------------------------

    def resample_eci(self, dt_s: float = 10.0) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Resample ephemeris onto a uniform grid in ECI.

        Returns
        -------
        times_utc  : list of datetime
        pos_km     : (M, 3) ECI positions
        vel_kms    : (M, 3) ECI velocities  (raises if no velocity in file)
        """
        ss    = np.arange(0.0, self._xs[-1] + dt_s * 0.5, dt_s)
        ss    = ss[ss <= self._xs[-1]]
        times = [self._t0 + timedelta(seconds=float(s)) for s in ss]
        pos   = np.array([self.position_eci(t) for t in times])
        vel   = np.array([self.velocity_eci(t) for t in times])
        return times, pos, vel

    def resample_ecr(self, dt_s: float = 10.0) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Resample ephemeris onto a uniform grid in ECR.

        Returns
        -------
        times_utc  : list of datetime
        pos_km     : (M, 3) ECR positions
        vel_kms    : (M, 3) ECR velocities  (raises if no velocity in file)
        """
        ss    = np.arange(0.0, self._xs[-1] + dt_s * 0.5, dt_s)
        ss    = ss[ss <= self._xs[-1]]
        times = [self._t0 + timedelta(seconds=float(s)) for s in ss]
        pos   = np.array([self.position_ecr(t) for t in times])
        vel   = np.array([self.velocity_ecr(t) for t in times])
        return times, pos, vel

    # -- display ------------------------------------------------------------

    def summary(self) -> str:
        dt = self._xs[1] - self._xs[0] if len(self._xs) > 1 else 0.0
        lines = [
            "StkEphemeris",
            f"  File epoch  : {self.t_start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z"
            f" -- {self.t_end.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z",
            f"  Points      : {self.n_points}  (dt_nominal={dt:.3f}s)",
            f"  Span        : {self.span_s:.1f} s  ({self.span_s/3600:.3f} h)",
            f"  Frame       : {self.coordinate_system}",
            f"  Velocity    : {'yes' if self.has_velocity else 'no'}",
            f"  Interp      : Lagrange order {self._n - 1} ({self._n}-point window)",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"StkEphemeris({self.n_points} pts, "
            f"{self.coordinate_system}, "
            f"span={self.span_s:.0f}s)"
        )

    # -- loader -------------------------------------------------------------

    @classmethod
    def load(cls, path: str, n_interp: int = 9) -> "StkEphemeris":
        """
        Load an STK ephemeris file (.e) and return an StkEphemeris object.

        Parameters
        ----------
        path     : path to the STK .e file
        n_interp : number of Lagrange nodes per window (default 9 = order 8)

        Raises
        ------
        ValueError  if a required header field is missing or unrecognised
        IOError     if the file cannot be read
        """
        return _parse_stk_file(Path(path), n_interp)


# ---------------------------------------------------------------------------
# STK file parser
# ---------------------------------------------------------------------------

def load_stk_ephemeris(path: str, n_interp: int = 9) -> StkEphemeris:
    """
    Load an STK ephemeris (.e) file.

    Equivalent to StkEphemeris.load(path, n_interp).

    Parameters
    ----------
    path     : path to the .e file
    n_interp : Lagrange window size (default 9, i.e. order-8 polynomial)

    Returns
    -------
    StkEphemeris
    """
    return _parse_stk_file(Path(path), n_interp)


def _parse_stk_file(path: Path, n_interp: int) -> StkEphemeris:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    meta: dict = {}
    scenario_epoch: Optional[datetime] = None
    dist_scale = 1.0      # multiply raw values to get km
    coord_sys  = "J2000"
    has_vel    = False
    data_start = None     # line index where numeric data begins

    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        i  += 1

        if not raw or raw.startswith("#"):
            continue

        low = raw.lower()

        # Version / block markers
        if low.startswith("stk.v") or low == "begin ephemeris":
            continue
        if low == "end ephemeris":
            break

        # Data section keyword
        if low == "ephemeristimeposvel":
            has_vel    = True
            data_start = i   # next line is data
            break
        if low == "ephemeristimepos":
            has_vel    = False
            data_start = i
            break

        # Header key-value pairs
        m = re.match(r"^(\w+)\s+(.+)$", raw)
        if m:
            key, val = m.group(1), m.group(2).strip()
            meta[key] = val
            kl = key.lower()
            if kl == "scenarioepoch":
                scenario_epoch = _parse_epoch(val)
            elif kl == "coordinatesystem":
                coord_sys = val.strip()
            elif kl == "distanceunit":
                vl = val.lower()
                if "meter" in vl and "kilo" not in vl:
                    dist_scale = 1e-3   # m -> km
                # kilometres is already default

    if scenario_epoch is None:
        raise ValueError("STK file missing 'ScenarioEpoch' header.")
    if data_start is None:
        raise ValueError(
            "STK file missing data keyword "
            "(EphemerisTimePosVel or EphemerisTimePos)."
        )

    # Parse numeric data
    times, pos_rows, vel_rows = [], [], []
    n_cols = 7 if has_vel else 4

    for line in lines[data_start:]:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.lower().startswith("end"):
            break
        try:
            vals = [float(v) for v in raw.split()]
        except ValueError:
            continue
        if len(vals) < n_cols:
            continue

        t_s = vals[0]
        t   = scenario_epoch + timedelta(seconds=t_s)
        times.append(t)
        pos_rows.append([vals[1] * dist_scale,
                         vals[2] * dist_scale,
                         vals[3] * dist_scale])
        if has_vel:
            vel_rows.append([vals[4] * dist_scale,
                             vals[5] * dist_scale,
                             vals[6] * dist_scale])

    if len(times) < 2:
        raise ValueError(f"Too few data points in STK file: {len(times)}")

    pos_km  = np.array(pos_rows)
    vel_kms = np.array(vel_rows) if has_vel else None
    is_eci  = coord_sys.lower() in _ECI_SYSTEMS

    if coord_sys.lower() not in _ECI_SYSTEMS | _ECR_SYSTEMS:
        import warnings
        warnings.warn(
            f"Unrecognised CoordinateSystem '{coord_sys}'; "
            "assuming ECI (J2000).",
            stacklevel=3,
        )
        is_eci = True

    meta["CoordinateSystem"] = coord_sys
    return StkEphemeris(
        times_utc=times,
        pos_km=pos_km,
        vel_kms=vel_kms,
        is_eci=is_eci,
        metadata=meta,
        n_interp=n_interp,
    )

"""
moon_jpl — Moon ECI/ECR state vectors via JPL Development Ephemeris.

Uses jplephem to read a DE binary SPK kernel (.bsp).
If no kernel is provided, de432s.bsp is downloaded automatically from NAIF
(~10 MB; covers 1950-01-01 → 2050-01-01).

Dependencies: numpy, jplephem  (no astropy, no skyfield)

Frame
-----
Positions are returned in the ICRF/J2000.0 inertial frame (ECI).
ECR uses GMST rotation (no EOP corrections; ≲ 1 km error at lunar distance).

Time conversion
---------------
DE kernels expect Barycentric Dynamical Time (TDB).  UTC is converted with:
    JDE = JD_UTC + delta_t_s / 86400
Default delta_t_s = 69.184 s  (TT − UTC with 37 leap seconds; valid ≥ 2017).
Add 1 s per new leap second.  Typical position error from ΔT uncertainty: < 1 km.

Vectorised inputs
-----------------
All public functions accept a single datetime or any sequence of datetimes
(list, tuple, ndarray).  Scalar input → shape (3,); sequence → shape (3, N).

Public API
----------
setup(bsp_path=None)           download/verify BSP; returns Path to the file
moon_pos_eci(t, ...)           → ndarray [km]  ECI J2000.0
moon_state_eci(t, ...)         → (pos, vel)    ECI J2000.0  [km, km/s]
moon_pos_ecr(t, ...)           → ndarray [km]  ECR (ECEF)
moon_state_ecr(t, ...)         → (pos, vel)    ECR (ECEF)   [km, km/s]
"""

from __future__ import annotations

import math
import pathlib
import urllib.request
from datetime import datetime, timezone
from typing import Sequence, Union

import numpy as np

from .jdate import datetime_to_jd

__all__ = [
    "setup",
    "moon_pos_eci",
    "moon_state_eci",
    "moon_pos_ecr",
    "moon_state_ecr",
]

# ── Constants ─────────────────────────────────────────────────────────────────

_J2000       = 2_451_545.0
_OMEGA_E     = 7.292115e-5          # Earth rotation rate [rad/s]
_DEFAULT_DT  = 69.184               # TT − UTC [s]  (37 leap seconds + 32.184)

# ── Kernel location ───────────────────────────────────────────────────────────

_NAIF_URL    = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
    "/spk/planets/de432s.bsp"
)
_CACHE_DIR   = pathlib.Path.home() / ".cache" / "tasking_helper" / "kernels"
_DEFAULT_BSP = _CACHE_DIR / "de432s.bsp"

# Module-level kernel cache keyed by resolved path string
_kernel_cache: dict[str, object] = {}


# ── Setup / kernel management ─────────────────────────────────────────────────

def setup(bsp_path: str | pathlib.Path | None = None) -> pathlib.Path:
    """
    Ensure a DE kernel is available and return its Path.

    If *bsp_path* is None, de432s.bsp is downloaded from NAIF into
    ``~/.cache/tasking_helper/kernels/`` the first time this is called.

    Parameters
    ----------
    bsp_path : str | Path | None
        Path to an existing .bsp file, or None to use the default de432s.bsp.

    Returns
    -------
    pathlib.Path  resolved path to the .bsp file.
    """
    if bsp_path is not None:
        p = pathlib.Path(bsp_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Kernel not found: {p}")
        return p

    if not _DEFAULT_BSP.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[moon_jpl] Downloading de432s.bsp → {_DEFAULT_BSP}")
        print("           (≈10 MB from naif.jpl.nasa.gov; happens once)")
        urllib.request.urlretrieve(_NAIF_URL, _DEFAULT_BSP)
        print(f"[moon_jpl] Download complete ({_DEFAULT_BSP.stat().st_size // 1024} KB)")

    return _DEFAULT_BSP


def _get_kernel(bsp_path: str | pathlib.Path | None):
    """Open and cache the SPK kernel."""
    try:
        from jplephem.spk import SPK
    except ImportError as exc:
        raise ImportError(
            "moon_jpl requires jplephem: pip install jplephem"
        ) from exc

    path = str(setup(bsp_path))
    if path not in _kernel_cache:
        _kernel_cache[path] = SPK.open(path)
    return _kernel_cache[path]


# ── Time helpers ──────────────────────────────────────────────────────────────

def _to_jde(
    t: Union[datetime, Sequence[datetime]],
    delta_t_s: float,
) -> tuple[np.ndarray, bool]:
    """Convert UTC datetime(s) → JDE (TDB Julian Date).

    Returns (jde_array, scalar_flag).  scalar_flag is True when *t* was a
    single datetime (so callers can squeeze the output back to shape (3,)).
    """
    djd = delta_t_s / 86400.0
    if isinstance(t, datetime):
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return np.array([datetime_to_jd(t) + djd]), True

    ts = list(t)
    jds = np.empty(len(ts))
    for i, ti in enumerate(ts):
        if ti.tzinfo is None:
            ti = ti.replace(tzinfo=timezone.utc)
        jds[i] = datetime_to_jd(ti)
    return jds + djd, False


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _gmst_rad(jd: np.ndarray) -> np.ndarray:
    """GMST [rad] for scalar or array Julian Date (Meeus Ch. 12)."""
    D = jd - _J2000
    T = D / 36525.0
    theta = (280.46061837
             + 360.98564736629 * D
             + 0.000387933     * T**2
             - T**3            / 38710000.0)
    return np.radians(theta % 360.0)


def _rz_apply(pos: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Rotate pos by Rz(theta).
    pos   : (3,) or (3, N)
    theta : scalar or (N,)
    """
    c, s = np.cos(theta), np.sin(theta)
    x = c * pos[0] + s * pos[1]
    y = -s * pos[0] + c * pos[1]
    z = pos[2] + np.zeros_like(c)      # broadcast z to same shape as x, y
    return np.array([x, y, z])


def _vel_eci_to_ecr(
    vel_eci: np.ndarray,
    pos_ecr: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Transform ECI velocity → ECR velocity.
        v_ecr = Rz(θ) · v_eci − ω_E × r_ecr
    """
    c, s = np.cos(theta), np.sin(theta)
    vx = c * vel_eci[0] + s * vel_eci[1] + _OMEGA_E * pos_ecr[1]
    vy = -s * vel_eci[0] + c * vel_eci[1] - _OMEGA_E * pos_ecr[0]
    vz = vel_eci[2] + np.zeros_like(c)
    return np.array([vx, vy, vz])


# ── Core ephemeris query ──────────────────────────────────────────────────────

def _moon_geocentric(
    jde: np.ndarray,
    kernel,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return Moon geocentric position [km] and velocity [km/s] in ICRF/J2000.0.

    Combines segments:
      [3, 301] Moon  from Earth-Moon Barycenter (EMB)
      [3, 399] Earth from Earth-Moon Barycenter (EMB)
    → Moon geocentric = [3,301] − [3,399]
    """
    m_pos, m_vel = kernel[3, 301].compute_and_differentiate(jde)
    e_pos, e_vel = kernel[3, 399].compute_and_differentiate(jde)
    pos = m_pos - e_pos                       # km
    vel = (m_vel - e_vel) / 86400.0          # km/day → km/s
    return pos, vel


def _squeeze(arr: np.ndarray, scalar: bool) -> np.ndarray:
    """If the input was a single datetime, return shape (3,) instead of (3,1)."""
    return arr[:, 0] if scalar else arr


# ── Public API ────────────────────────────────────────────────────────────────

def moon_pos_eci(
    t: Union[datetime, Sequence[datetime]],
    bsp: str | pathlib.Path | None = None,
    delta_t_s: float = _DEFAULT_DT,
) -> np.ndarray:
    """
    Moon position in ECI (ICRF/J2000.0) [km].

    Parameters
    ----------
    t         : datetime or sequence of datetimes (UTC; naive assumed UTC).
    bsp       : path to a .bsp kernel, or None for the default de432s.bsp.
    delta_t_s : TT − UTC in seconds (default 69.184 s for ≥ 2017).

    Returns
    -------
    np.ndarray  shape (3,) for scalar *t*, or (3, N) for a sequence.
    """
    jde, scalar = _to_jde(t, delta_t_s)
    kernel = _get_kernel(bsp)
    pos, _ = _moon_geocentric(jde, kernel)
    return _squeeze(pos, scalar)


def moon_state_eci(
    t: Union[datetime, Sequence[datetime]],
    bsp: str | pathlib.Path | None = None,
    delta_t_s: float = _DEFAULT_DT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moon position [km] and velocity [km/s] in ECI (ICRF/J2000.0).

    Velocity is the analytic derivative from the DE kernel (not finite-diff).

    Returns
    -------
    pos : ndarray  (3,) or (3, N)
    vel : ndarray  (3,) or (3, N)
    """
    jde, scalar = _to_jde(t, delta_t_s)
    kernel = _get_kernel(bsp)
    pos, vel = _moon_geocentric(jde, kernel)
    return _squeeze(pos, scalar), _squeeze(vel, scalar)


def moon_pos_ecr(
    t: Union[datetime, Sequence[datetime]],
    bsp: str | pathlib.Path | None = None,
    delta_t_s: float = _DEFAULT_DT,
) -> np.ndarray:
    """
    Moon position in ECR (ECEF) [km].

    Returns
    -------
    np.ndarray  shape (3,) or (3, N).
    """
    jde, scalar = _to_jde(t, delta_t_s)
    kernel = _get_kernel(bsp)
    pos, _ = _moon_geocentric(jde, kernel)
    # Use JD (UTC) for GMST — jde minus the delta_t correction
    jd_utc = jde - delta_t_s / 86400.0
    theta   = _gmst_rad(jd_utc)
    pos_ecr = _rz_apply(pos, theta)
    return _squeeze(pos_ecr, scalar)


def moon_state_ecr(
    t: Union[datetime, Sequence[datetime]],
    bsp: str | pathlib.Path | None = None,
    delta_t_s: float = _DEFAULT_DT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moon position [km] and velocity [km/s] in ECR (ECEF).

    Velocity accounts for Earth's rotation:
        v_ecr = Rz(θ_GMST) · v_eci − ω_E × r_ecr

    Returns
    -------
    pos : ndarray  (3,) or (3, N)
    vel : ndarray  (3,) or (3, N)
    """
    jde, scalar = _to_jde(t, delta_t_s)
    kernel = _get_kernel(bsp)
    pos, vel = _moon_geocentric(jde, kernel)
    jd_utc  = jde - delta_t_s / 86400.0
    theta   = _gmst_rad(jd_utc)
    pos_ecr = _rz_apply(pos, theta)
    vel_ecr = _vel_eci_to_ecr(vel, pos_ecr, theta)
    return _squeeze(pos_ecr, scalar), _squeeze(vel_ecr, scalar)

"""
keplerian.py — Conversions between geodetic (lat/lon/alt), ECI/ECR Cartesian,
               and Keplerian orbital elements.

Coordinate frames
-----------------
LLA   Geodetic:  latitude [deg], longitude [deg], altitude [km]
      Uses the WGS-84 ellipsoid.
ECR   Earth-Centred Rotating (≡ ECEF): x, y, z [km]
ECI   Earth-Centred Inertial (GCRS ≈ J2000): x, y, z [km]

Frame rotation ECI ↔ ECR uses GMST (no EOP corrections; < 0.1 km error for LEO).

Keplerian elements
------------------
a_km     semi-major axis                    [km]
ecc      eccentricity                       (0 = circular, 0 ≤ ecc < 1)
incl_deg inclination                        [deg]
raan_deg right ascension of ascending node  [deg]
argp_deg argument of perigee               [deg]
M_deg    mean anomaly at epoch              [deg]

All public functions that accept angles expect degrees.
All public functions that return angles return degrees.
Distances are always in km; velocities in km/s.

Public API
----------
Geodetic ↔ Cartesian:
    lla_to_ecr(lat, lon, alt)                   → ndarray (3,) [km]
    ecr_to_lla(r_ecr)                           → (lat°, lon°, alt km)
    lla_to_eci(lat, lon, alt, t)                → ndarray (3,) [km]
    eci_to_lla(r_eci, t)                        → (lat°, lon°, alt km)
    eci_to_ecr(r_eci, t)                        → ndarray (3,) [km]
    ecr_to_eci(r_ecr, t)                        → ndarray (3,) [km]

Orbital mechanics:
    state_to_keplerian(pos_km, vel_kms)         → dict
    keplerian_to_state(a_km, ecc, incl_deg,
                       raan_deg, argp_deg, M_deg) → (pos [km], vel [km/s])

Pipelines:
    keplerian_to_lla(a_km, ecc, incl_deg,
                     raan_deg, argp_deg, M_deg, t) → (lat°, lon°, alt km)
    lla_to_keplerian(lat, lon, alt,
                     vel_eci_kms, t)              → dict

Notes
-----
* A single LLA position does not uniquely determine an orbit.  lla_to_keplerian
  therefore requires the ECI velocity vector as a separate argument.
* For near-circular orbits (ecc < 1e-4) argp_deg and M_deg become individually
  ill-defined; the returned sum (nu_deg + argp_deg = argument of latitude) remains
  meaningful.  For near-equatorial orbits (incl < 0.01°) raan_deg is similarly
  arbitrary.  The round-trip keplerian_to_state(state_to_keplerian(r, v)) is
  exact regardless of orbit type.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Union

import numpy as np

from .jdate import datetime_to_jd
from .utils import (
    eci_to_ecef as _eci_to_ecef_m,
    ecef_to_eci as _ecef_to_eci_m,
    ecef_to_lla as _ecef_to_lla,
    lla_to_ecef as _lla_to_ecef_m,
    gmst as _gmst,
    julian_date as _julian_date,
)

__all__ = [
    # Geodetic ↔ Cartesian
    "lla_to_ecr",
    "ecr_to_lla",
    "lla_to_eci",
    "eci_to_lla",
    "eci_to_ecr",
    "ecr_to_eci",
    # Orbital mechanics
    "state_to_keplerian",
    "keplerian_to_state",
    "solve_kepler",
    # Pipelines
    "keplerian_to_lla",
    "lla_to_keplerian",
]

# ── Constants ─────────────────────────────────────────────────────────────────

MU     = 398_600.4418   # Earth gravitational parameter [km³ s⁻²]
TWO_PI = 2.0 * math.pi


# ── Internal helpers ──────────────────────────────────────────────────────────

def _jd(t: datetime) -> float:
    """UTC datetime → Julian Date."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return datetime_to_jd(t)


def _deg(x: float) -> float:
    return math.degrees(x) % 360.0


# ── Geodetic ↔ Cartesian ──────────────────────────────────────────────────────

def lla_to_ecr(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    """
    Geodetic LLA → ECR position [km].

    Parameters
    ----------
    lat_deg : geodetic latitude  [deg]  −90 … +90
    lon_deg : longitude          [deg]  −180 … +360
    alt_km  : altitude above WGS-84 ellipsoid [km]

    Returns
    -------
    r_ecr : ndarray (3,)  [km]
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return _lla_to_ecef_m(lat, lon, alt_km * 1e3) / 1e3   # m → km


def ecr_to_lla(r_ecr: np.ndarray) -> tuple[float, float, float]:
    """
    ECR position [km] → geodetic (lat°, lon°, alt km).

    Uses Bowring's iterative method (5 iterations — accurate to < 0.1 mm).
    """
    lla = _ecef_to_lla(np.asarray(r_ecr) * 1e3)           # km → m
    lat_deg = math.degrees(float(lla[0]))
    lon_deg = math.degrees(float(lla[1]))
    alt_km  = float(lla[2]) / 1e3
    return lat_deg, lon_deg, alt_km


def eci_to_ecr(r_eci: np.ndarray, t: datetime) -> np.ndarray:
    """
    ECI position [km] → ECR position [km] at epoch t (UTC).

    Rotation by GMST (no polar motion correction).
    """
    jd = _jd(t)
    return _eci_to_ecef_m(np.asarray(r_eci) * 1e3, jd) / 1e3


def ecr_to_eci(r_ecr: np.ndarray, t: datetime) -> np.ndarray:
    """
    ECR position [km] → ECI position [km] at epoch t (UTC).
    """
    jd = _jd(t)
    return _ecef_to_eci_m(np.asarray(r_ecr) * 1e3, jd) / 1e3


def lla_to_eci(lat_deg: float, lon_deg: float, alt_km: float,
               t: datetime) -> np.ndarray:
    """
    Geodetic LLA → ECI position [km] at epoch t (UTC).

    Useful for fixed ground points (e.g., a sensor site or target).
    """
    r_ecr = lla_to_ecr(lat_deg, lon_deg, alt_km)
    return ecr_to_eci(r_ecr, t)


def eci_to_lla(r_eci: np.ndarray, t: datetime) -> tuple[float, float, float]:
    """
    ECI position [km] → geodetic (lat°, lon°, alt km) at epoch t (UTC).

    Useful for finding the subsatellite point and altitude of an orbiting body.
    """
    r_ecr = eci_to_ecr(r_eci, t)
    return ecr_to_lla(r_ecr)


# ── Kepler's equation ─────────────────────────────────────────────────────────

def solve_kepler(M_deg: float, ecc: float, tol: float = 1e-12) -> float:
    """
    Solve Kepler's equation  M = E − e·sin(E)  for the eccentric anomaly E.

    Parameters
    ----------
    M_deg : mean anomaly [deg]
    ecc   : eccentricity  (0 ≤ ecc < 1)
    tol   : convergence tolerance on E [rad]  (default 1e-12 rad ≈ 0.2 µm at LEO)

    Returns
    -------
    E_deg : eccentric anomaly [deg]

    Algorithm
    ---------
    Newton-Raphson with the Meeus starter
        E₀ = M + e·sin(M)·(1 + e·cos(M))
    Converges in < 10 iterations for ecc ≤ 0.95; uses a Halley step for
    ecc > 0.95 to ensure convergence at high eccentricity.
    """
    if not (0.0 <= ecc < 1.0):
        raise ValueError(f"Eccentricity must satisfy 0 ≤ ecc < 1, got {ecc}")
    M = math.radians(M_deg) % TWO_PI

    # Initial guess (Meeus, Astronomical Algorithms §30)
    E = M + ecc * math.sin(M) * (1.0 + ecc * math.cos(M))

    for _ in range(50):
        f  = E - ecc * math.sin(E) - M
        fp = 1.0 - ecc * math.cos(E)
        if ecc > 0.95:
            # Halley step for high eccentricity
            fpp = ecc * math.sin(E)
            dE  = -f / (fp - 0.5 * f * fpp / fp)
        else:
            dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break

    return math.degrees(E)


# ── ECI state vector ↔ Keplerian elements ─────────────────────────────────────

def state_to_keplerian(pos_km: np.ndarray,
                        vel_kms: np.ndarray) -> dict:
    """
    Convert an ECI state vector to osculating Keplerian elements.

    Parameters
    ----------
    pos_km  : ECI position [km]   shape (3,)
    vel_kms : ECI velocity [km/s] shape (3,)

    Returns
    -------
    dict with keys:
        a_km     semi-major axis [km]
        ecc      eccentricity
        incl_deg inclination [deg]
        raan_deg RAAN / Ω [deg]
        argp_deg argument of perigee / ω [deg]
        M_deg    mean anomaly [deg]
        nu_deg   true anomaly [deg]
        E_deg    eccentric anomaly [deg]
        T_s      orbital period [s]  (= 2π√(a³/μ))
        h_km2_s  specific angular momentum magnitude [km² s⁻¹]
        r_km     current orbital radius [km]

    Notes
    -----
    Near-circular orbits (ecc < 1e-4):  argp_deg is set to 0; M_deg carries
    the argument of latitude (ω + ν).
    Near-equatorial orbits (incl < 0.01°):  raan_deg is set to 0.
    """
    r_v  = np.asarray(pos_km,  dtype=float)
    v_v  = np.asarray(vel_kms, dtype=float)
    r    = float(np.linalg.norm(r_v))
    v    = float(np.linalg.norm(v_v))

    h_v  = np.cross(r_v, v_v)             # specific angular momentum [km² s⁻¹]
    h    = float(np.linalg.norm(h_v))
    n_v  = np.cross([0.0, 0.0, 1.0], h_v) # ascending-node vector
    n    = float(np.linalg.norm(n_v))

    # Eccentricity vector
    e_v  = np.cross(v_v, h_v) / MU - r_v / r
    ecc  = float(np.linalg.norm(e_v))

    # Semi-major axis via vis-viva
    energy = v ** 2 / 2.0 - MU / r
    a      = -MU / (2.0 * energy)

    # Inclination
    incl = math.acos(max(-1.0, min(1.0, h_v[2] / h)))

    # RAAN
    if n < 1e-10:
        raan = 0.0
    else:
        raan = math.acos(max(-1.0, min(1.0, n_v[0] / n)))
        if n_v[1] < 0:
            raan = TWO_PI - raan

    # Argument of perigee
    if n < 1e-10 or ecc < 1e-10:
        argp = 0.0
    else:
        argp = math.acos(max(-1.0, min(1.0, np.dot(n_v, e_v) / (n * ecc))))
        if e_v[2] < 0:
            argp = TWO_PI - argp

    # True anomaly
    if ecc < 1e-10:
        # Near-circular: use argument of latitude u = ω + ν
        if n < 1e-10:
            nu = math.acos(max(-1.0, min(1.0, r_v[0] / r)))
            if r_v[1] < 0:
                nu = TWO_PI - nu
        else:
            nu = math.acos(max(-1.0, min(1.0, np.dot(n_v, r_v) / (n * r))))
            if np.dot(r_v, v_v) < 0:
                nu = TWO_PI - nu
    else:
        nu = math.acos(max(-1.0, min(1.0, np.dot(e_v, r_v) / (ecc * r))))
        if np.dot(r_v, v_v) < 0:
            nu = TWO_PI - nu

    # Eccentric anomaly and mean anomaly
    E_anom = 2.0 * math.atan2(
        math.sqrt(max(0.0, 1.0 - ecc)) * math.sin(nu / 2.0),
        math.sqrt(max(0.0, 1.0 + ecc)) * math.cos(nu / 2.0),
    )
    M = (E_anom - ecc * math.sin(E_anom)) % TWO_PI
    T = TWO_PI * math.sqrt(a ** 3 / MU)

    return {
        "a_km":     a,
        "ecc":      ecc,
        "incl_deg": _deg(incl),
        "raan_deg": _deg(raan),
        "argp_deg": _deg(argp),
        "M_deg":    _deg(M),
        "nu_deg":   _deg(nu),
        "E_deg":    _deg(E_anom),
        "T_s":      T,
        "h_km2_s":  h,
        "r_km":     r,
    }


def keplerian_to_state(a_km:     float,
                        ecc:      float,
                        incl_deg: float,
                        raan_deg: float,
                        argp_deg: float,
                        M_deg:    float,
                        **_) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Keplerian elements to an ECI state vector.

    The ``**_`` signature absorbs extra keys so a dict returned by
    ``state_to_keplerian`` can be unpacked directly:

        pos, vel = keplerian_to_state(**state_to_keplerian(r, v))

    Parameters
    ----------
    a_km     : semi-major axis [km]
    ecc      : eccentricity  (0 ≤ ecc < 1)
    incl_deg : inclination [deg]
    raan_deg : RAAN / Ω [deg]
    argp_deg : argument of perigee / ω [deg]
    M_deg    : mean anomaly [deg]

    Returns
    -------
    pos : ndarray (3,) [km]   ECI position
    vel : ndarray (3,) [km/s] ECI velocity
    """
    i   = math.radians(incl_deg)
    Om  = math.radians(raan_deg)
    w   = math.radians(argp_deg)

    # Eccentric anomaly → true anomaly
    E   = math.radians(solve_kepler(M_deg, ecc))
    nu  = 2.0 * math.atan2(
        math.sqrt(1.0 + ecc) * math.sin(E / 2.0),
        math.sqrt(1.0 - ecc) * math.cos(E / 2.0),
    )

    # Orbit radius and semi-latus rectum
    r = a_km * (1.0 - ecc * math.cos(E))
    p = a_km * (1.0 - ecc ** 2)

    # Position and velocity in the perifocal (PQW) frame
    sqrt_mu_p = math.sqrt(MU / p)
    x_p  =  r * math.cos(nu)
    y_p  =  r * math.sin(nu)
    vx_p = -sqrt_mu_p * math.sin(nu)
    vy_p =  sqrt_mu_p * (ecc + math.cos(nu))

    # Perifocal → ECI rotation  (Rz(−Ω) · Rx(−i) · Rz(−ω))
    cO, sO = math.cos(Om), math.sin(Om)
    ci, si = math.cos(i),  math.sin(i)
    cw, sw = math.cos(w),  math.sin(w)

    # P̂ and Q̂ column vectors of the rotation matrix
    Px =  cO * cw - sO * sw * ci
    Py =  sO * cw + cO * sw * ci
    Pz =  sw * si
    Qx = -cO * sw - sO * cw * ci
    Qy = -sO * sw + cO * cw * ci
    Qz =  cw * si

    pos = np.array([Px * x_p  + Qx * y_p,
                    Py * x_p  + Qy * y_p,
                    Pz * x_p  + Qz * y_p])
    vel = np.array([Px * vx_p + Qx * vy_p,
                    Py * vx_p + Qy * vy_p,
                    Pz * vx_p + Qz * vy_p])
    return pos, vel


# ── Pipeline functions ────────────────────────────────────────────────────────

def keplerian_to_lla(a_km:     float,
                      ecc:      float,
                      incl_deg: float,
                      raan_deg: float,
                      argp_deg: float,
                      M_deg:    float,
                      t:        datetime,
                      **_) -> tuple[float, float, float]:
    """
    Keplerian elements + epoch → subsatellite geodetic position.

    Computes the ECI state from the elements, rotates to ECR using GMST at t,
    then converts to geodetic LLA.

    Parameters
    ----------
    a_km … M_deg : orbital elements (see keplerian_to_state)
    t            : UTC epoch (aware or naive datetime; naive assumed UTC)

    Returns
    -------
    (lat_deg, lon_deg, alt_km) : geodetic subsatellite point
        lat_deg  −90 … +90
        lon_deg  −180 … +180  (from atan2)
        alt_km   altitude above WGS-84 ellipsoid [km]
    """
    pos, _ = keplerian_to_state(a_km, ecc, incl_deg, raan_deg, argp_deg, M_deg)
    return eci_to_lla(pos, t)


def lla_to_keplerian(lat_deg:     float,
                      lon_deg:     float,
                      alt_km:      float,
                      vel_eci_kms: np.ndarray,
                      t:           datetime) -> dict:
    """
    Satellite geodetic position + ECI velocity + epoch → Keplerian elements.

    A position alone does not define an orbit; the ECI velocity must be
    supplied.  Use ``keplerian_to_state`` to generate a consistent velocity
    if you only have orbital elements.

    Parameters
    ----------
    lat_deg     : geodetic latitude  [deg]
    lon_deg     : longitude          [deg]
    alt_km      : altitude above WGS-84 ellipsoid [km]
    vel_eci_kms : ECI velocity [km/s]  shape (3,)
    t           : UTC epoch

    Returns
    -------
    dict : same keys as state_to_keplerian
    """
    pos_eci = lla_to_eci(lat_deg, lon_deg, alt_km, t)
    return state_to_keplerian(pos_eci, np.asarray(vel_eci_kms, dtype=float))

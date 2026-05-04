"""
moon_eci — Moon ECI/ECR state vectors via truncated ELP2000/82 series.

Implements Meeus "Astronomical Algorithms" 2nd ed., Chapters 22 and 47.
Includes the full 60-term tables for longitude/distance and latitude,
IAU 1976 precession, and IAU 1980 nutation (63-term table).

Typical accuracy over 2025–2050 (vs JPL DE432s):
  position  ≲ 100 km  (truncated ELP2000/82 inherent; epoch-dependent)
  velocity  ≲ 0.2 m/s  (central finite-difference, dt = 1 s)

For position accuracy < 10 km use moon_jpl.py (JPL DE kernel).

Dependencies: numpy, .jdate (datetime_to_jd)

Public API
----------
moon_pos_eci(t)    -> np.ndarray (3,)  [km]        ECI J2000.0
moon_state_eci(t)  -> (pos [km], vel [km/s])       ECI J2000.0
moon_pos_ecr(t)    -> np.ndarray (3,)  [km]        ECR (ECEF)
moon_state_ecr(t)  -> (pos [km], vel [km/s])       ECR (ECEF)
eci_to_ecr(pos, t) -> np.ndarray (3,)  [km]        generic ECI→ECR helper
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .jdate import datetime_to_jd

__all__ = [
    "moon_pos_eci",
    "moon_state_eci",
    "moon_pos_ecr",
    "moon_state_ecr",
    "eci_to_ecr",
]

# Earth's rotation rate [rad/s]  (IAU)
_OMEGA_E = 7.292115e-5

# ── J2000 Julian Date ─────────────────────────────────────────────────────────
_J2000 = 2_451_545.0

# ── Meeus Table 47.A ──────────────────────────────────────────────────────────
# Periodic terms for longitude Σl (units: 1e-6 deg) and
# distance Σr (units: 1e-3 km).
# Columns: D, M, M', F, Σl_coeff, Σr_coeff
_TABLE_LR = (
    # D   M   M'  F     Σl          Σr
    ( 0,  0,  1,  0,  6288774, -20905355),
    ( 2,  0, -1,  0,  1274027,  -3699111),
    ( 2,  0,  0,  0,   658314,  -2955968),
    ( 0,  0,  2,  0,   213618,   -569925),
    ( 0,  1,  0,  0,  -185116,     48888),
    ( 0,  0,  0,  2,  -114332,     -3149),
    ( 2,  0, -2,  0,    58793,    246158),
    ( 2, -1, -1,  0,    57066,   -152138),
    ( 2,  0,  1,  0,    53322,   -170733),
    ( 2, -1,  0,  0,    45758,   -204586),
    ( 0,  1, -1,  0,   -40923,   -129620),
    ( 1,  0,  0,  0,   -34720,    108743),
    ( 0,  1,  1,  0,   -30383,    104755),
    ( 2,  0,  0, -2,    15327,     10321),
    ( 0,  0,  1,  2,   -12528,         0),
    ( 0,  0,  1, -2,    10980,     79661),
    ( 4,  0, -1,  0,    10675,    -34782),
    ( 0,  0,  3,  0,    10034,    -23210),
    ( 4,  0, -2,  0,     8548,    -21636),
    ( 2,  1, -1,  0,    -7888,     24208),
    ( 2,  1,  0,  0,    -6766,     30824),
    ( 1,  0, -1,  0,    -5163,     -8379),
    ( 1,  1,  0,  0,     4987,    -16675),
    ( 2, -1,  1,  0,     4036,    -12831),
    ( 2,  0,  2,  0,     3994,    -10445),
    ( 4,  0,  0,  0,     3861,    -11650),
    ( 2,  0, -3,  0,     3665,     14403),
    ( 0,  1, -2,  0,    -2689,     -7003),
    ( 2,  0, -1,  2,    -2602,         0),
    ( 2, -1, -2,  0,     2390,     10056),
    ( 1,  0,  1,  0,    -2348,      6322),
    ( 2, -2,  0,  0,     2236,     -9884),
    ( 0,  1,  2,  0,    -2120,      5751),
    ( 0,  2,  0,  0,    -2069,         0),
    ( 2, -2, -1,  0,     2048,     -4950),
    ( 2,  0,  1, -2,    -1773,      4130),
    ( 2,  0,  0,  2,    -1595,         0),
    ( 4, -1, -1,  0,     1215,     -3958),
    ( 0,  0,  2,  2,    -1110,         0),
    ( 3,  0, -1,  0,     -892,      3258),
    ( 2,  1,  1,  0,     -810,      2616),
    ( 4, -1, -2,  0,      759,     -1897),
    ( 0,  2, -1,  0,     -713,     -2117),
    ( 2,  2, -1,  0,     -700,      2354),
    ( 2,  1, -2,  0,      691,         0),
    ( 2, -1,  0, -2,      596,         0),
    ( 4,  0,  1,  0,      549,     -1423),
    ( 0,  0,  4,  0,      537,     -1117),
    ( 4, -1,  0,  0,      520,     -1571),
    ( 1,  0, -2,  0,     -487,     -1739),
    ( 2,  1,  0, -2,     -399,         0),
    ( 0,  0,  2, -2,     -381,     -4421),
    ( 1,  1,  1,  0,      351,         0),
    ( 3,  0, -2,  0,     -340,         0),
    ( 4,  0, -3,  0,      330,         0),
    ( 2, -1,  2,  0,      327,         0),
    ( 0,  2,  1,  0,     -323,      1165),
    ( 1,  1, -1,  0,      299,         0),
    ( 2,  0,  3,  0,      294,         0),
    ( 2,  0, -1, -2,        0,      8752),
)

# ── Meeus Table 47.B ──────────────────────────────────────────────────────────
# Periodic terms for latitude Σb (units: 1e-6 deg).
# Columns: D, M, M', F, Σb_coeff
_TABLE_B = (
    # D   M   M'  F     Σb
    ( 0,  0,  0,  1,  5128122),
    ( 0,  0,  1,  1,   280602),
    ( 0,  0,  1, -1,   277693),
    ( 2,  0,  0, -1,   173237),
    ( 2,  0, -1,  1,    55413),
    ( 2,  0, -1, -1,    46271),
    ( 2,  0,  0,  1,    32573),
    ( 0,  0,  2,  1,    17198),
    ( 2,  0,  1, -1,     9266),
    ( 0,  0,  2, -1,     8822),
    ( 2, -1,  0, -1,     8216),
    ( 2,  0, -2, -1,     4324),
    ( 2,  0,  1,  1,     4200),
    ( 2,  1,  0, -1,    -3359),
    ( 2, -1, -1,  1,     2463),
    ( 2, -1,  0,  1,     2211),
    ( 2, -1, -1, -1,     2065),
    ( 0,  1, -1, -1,    -1870),
    ( 4,  0, -1, -1,     1828),
    ( 0,  1,  0,  1,    -1794),
    ( 0,  0,  0,  3,    -1749),
    ( 0,  1, -1,  1,    -1565),
    ( 1,  0,  0,  1,    -1491),
    ( 0,  1,  1,  1,    -1475),
    ( 0,  1,  1, -1,    -1410),
    ( 0,  1,  0, -1,    -1344),
    ( 1,  0,  0, -1,    -1335),
    ( 0,  0,  3,  1,     1107),
    ( 4,  0,  0, -1,     1021),
    ( 4,  0, -1,  1,      833),
    ( 0,  0,  1, -3,      777),
    ( 4,  0, -2,  1,      671),
    ( 2,  0,  0, -3,      607),
    ( 2,  0,  2, -1,      596),
    ( 2, -1,  1, -1,      491),
    ( 2,  0, -2,  1,     -451),
    ( 0,  0,  3, -1,      439),
    ( 2,  0,  2,  1,      422),
    ( 2,  0, -3, -1,      421),
    ( 2,  1, -1,  1,     -366),
    ( 2,  1,  0,  1,     -351),
    ( 4,  0,  0,  1,      331),
    ( 2, -1,  1,  1,      315),
    ( 2, -2,  0, -1,      302),
    ( 0,  0,  1,  3,     -283),
    ( 2,  1,  1, -1,     -229),
    ( 1,  1,  0, -1,      223),
    ( 1,  1,  0,  1,      223),
    ( 0,  1, -2, -1,     -220),
    ( 2,  1, -1, -1,     -220),
    ( 1,  0,  1,  1,     -185),
    ( 2, -1, -2, -1,      181),
    ( 0,  1,  2,  1,     -177),
    ( 4,  0, -2, -1,      176),
    ( 4, -1, -1, -1,      166),
    ( 1,  0,  1, -1,     -164),
    ( 4,  0,  1, -1,      132),
    ( 1,  0, -1, -1,     -119),
    ( 4, -1,  0, -1,      115),
    ( 2, -2,  0,  1,      107),
)


# ── IAU 1980 nutation table (Meeus Table 22.A, 63 terms) ─────────────────────
# Columns: l  l'  F  D  Ω | ψ_sin  ψ_T×10 | ε_cos  ε_T×10
# All coefficients in units of 0.0001 arcsec (or 0.0001 arcsec/century for T).
# Multiply T coefficient by T/10 during evaluation.
# arg = l*Mp + l'*M + F*F + D*D + Ω*Ω  (radians)
_NUT_TABLE = (
    # l   l'   F   D   Ω    ψ_sin    ψ_T×10   ε_cos   ε_T×10
    ( 0,   0,   0,  0,  1, -171996,   -1742,   92025,     89),
    (-2,   0,   0,  2,  2,  -13187,     -16,    5736,    -31),
    ( 0,   0,   0,  2,  2,   -2274,      -2,     977,     -5),
    ( 0,   0,   0,  0,  2,    2062,       2,    -895,      5),
    ( 0,   1,   0,  0,  0,    1426,     -34,      54,     -1),
    ( 0,   0,   1,  0,  0,     712,       1,      -7,      0),
    (-2,   1,   0,  2,  2,    -517,      12,     224,     -6),
    ( 0,   0,   0,  2,  1,    -386,      -4,     200,      0),
    ( 0,   0,   1,  2,  2,    -301,       0,     129,     -1),
    (-2,  -1,   0,  2,  2,     217,      -5,     -95,      3),
    (-2,   0,   1,  0,  0,    -158,       0,       0,      0),
    (-2,   0,   0,  2,  1,     129,       1,     -70,      0),
    ( 0,   0,  -1,  2,  2,     123,       0,     -53,      0),
    ( 2,   0,   0,  0,  0,      63,       0,       0,      0),
    ( 0,   0,   1,  0,  1,      63,       1,     -33,      0),
    ( 2,   0,  -1,  2,  2,     -59,       0,      26,      0),
    ( 0,   0,  -1,  0,  1,     -58,      -1,      32,      0),
    ( 0,   0,   1,  2,  1,     -51,       0,      27,      0),
    (-2,   0,   2,  0,  0,      48,       0,       0,      0),
    ( 0,   0,  -2,  2,  1,      46,       0,     -24,      0),
    ( 2,   0,   0,  2,  2,     -38,       0,      16,      0),
    ( 0,   0,   2,  2,  2,     -31,       0,      13,      0),
    ( 0,   0,   2,  0,  0,      29,       0,       0,      0),
    (-2,   0,   1,  2,  2,      29,       0,     -12,      0),
    ( 0,   0,   0,  2,  0,      26,       0,       0,      0),
    (-2,   0,   0,  2,  0,     -22,       0,       0,      0),
    ( 0,   0,  -1,  2,  1,      21,       0,     -10,      0),
    ( 0,   2,   0,  0,  0,      17,      -1,       0,      0),
    ( 2,   0,  -1,  0,  1,      16,       0,      -8,      0),
    (-2,   2,   0,  2,  2,     -16,       1,       7,      0),
    ( 0,   1,   0,  0,  1,     -15,       0,       9,      0),
    (-2,   0,   1,  0,  1,     -13,       0,       7,      0),
    ( 0,  -1,   0,  0,  1,     -12,       0,       6,      0),
    ( 0,   0,   2, -2,  0,      11,       0,       0,      0),
    ( 2,   0,  -1,  2,  1,     -10,       0,       5,      0),
    ( 2,   0,   1,  2,  2,      -8,       0,       3,      0),
    ( 0,   1,   0,  2,  2,      -7,       0,       3,      0),
    (-2,   1,   1,  0,  0,      -7,       0,       0,      0),
    ( 0,  -1,   0,  2,  2,      -7,       0,       3,      0),
    ( 2,   0,   0,  2,  1,      -6,       0,       3,      0),
    ( 2,   0,   1,  0,  0,      -6,       0,       0,      0),
    (-2,   0,   2,  2,  2,       5,       0,      -3,      0),
    (-2,   0,   1,  2,  1,       5,       0,      -3,      0),
    ( 2,   0,  -2,  0,  1,      -5,       0,       3,      0),
    ( 2,   0,   0,  0,  1,      -5,       0,       3,      0),
    ( 0,  -1,   1,  0,  0,      -5,       0,       0,      0),
    (-2,  -1,   0,  2,  1,      -5,       0,       3,      0),
    (-2,   0,   0,  0,  1,      -5,       0,       3,      0),
    ( 0,   0,   2,  2,  1,      -5,       0,       3,      0),
    (-2,   0,   2,  0,  1,       4,       0,       0,      0),
    (-2,   1,   0,  2,  1,       4,       0,      -2,      0),
    ( 0,   0,   1, -2,  0,       4,       0,       0,      0),
    (-1,   0,   1,  0,  0,      -4,       0,       0,      0),
    (-2,   1,   0,  0,  0,      -4,       0,       0,      0),
    ( 1,   0,   0,  0,  0,      -4,       0,       0,      0),
    ( 0,   0,   1,  2,  0,       3,       0,       0,      0),
    ( 0,   0,  -2,  2,  2,      -3,       0,       1,      0),
    (-1,  -1,   1,  0,  0,      -3,       0,       0,      0),
    ( 0,   1,   1,  0,  0,      -3,       0,       0,      0),
    ( 0,  -1,   1,  2,  2,      -3,       0,       1,      0),
    ( 2,  -1,  -1,  2,  2,      -3,       0,       1,      0),
    ( 0,   0,   3,  2,  2,      -3,       0,       1,      0),
    ( 2,  -1,   0,  2,  2,      -3,       0,       1,      0),
)

# ── Internal helpers ──────────────────────────────────────────────────────────

def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, s, 0.],
                     [-s, c, 0.],
                     [ 0., 0., 1.]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0., -s],
                     [ 0., 1.,  0.],
                     [ s, 0.,  c]])


def _precess_to_j2000(r_date: np.ndarray, T: float) -> np.ndarray:
    """Rotate mean-equatorial-of-date vector to J2000.0 (IAU 1976)."""
    as2r = math.pi / (180.0 * 3600.0)
    zeta_A = (2306.2181 * T + 1.39656  * T**2 - 0.000139 * T**3) * as2r
    theta_A = (2004.3109 * T - 0.85330  * T**2 - 0.000217 * T**3) * as2r
    z_A     = (2306.2181 * T + 1.09468  * T**2 + 0.018203 * T**3) * as2r
    # P = Rz(-z_A) · Ry(θ_A) · Rz(-ζ_A)  rotates J2000 → of-date
    # P^T rotates of-date → J2000
    P = _rz(-z_A) @ _ry(theta_A) @ _rz(-zeta_A)
    return P.T @ r_date


def _nutation(
    T: float,
    Mp_r: float,
    M_r: float,
    F_r: float,
    D_r: float,
) -> tuple[float, float]:
    """IAU 1980 nutation: returns (Δψ, Δε) in radians (Meeus Ch. 22)."""
    Om = (125.04452
          - 1934.136261 * T
          + 0.0020708   * T**2
          + T**3        / 450000.0)
    Om_r = math.radians(Om % 360.0)

    dpsi = deps = 0.0
    for l_c, lp_c, F_c, D_c, Om_c, psi_s, psi_T, eps_c, eps_T in _NUT_TABLE:
        arg = l_c * Mp_r + lp_c * M_r + F_c * F_r + D_c * D_r + Om_c * Om_r
        dpsi += (psi_s + psi_T * T / 10.0) * math.sin(arg)
        deps += (eps_c + eps_T * T / 10.0) * math.cos(arg)

    # 0.0001 arcsec → radians
    as2r = math.pi / (180.0 * 3600.0)
    return dpsi * 1e-4 * as2r, deps * 1e-4 * as2r


def _moon_pos_eci_jd(jd: float) -> np.ndarray:
    """Moon position [km] in ECI J2000.0 for a given Julian Date."""
    T = (jd - _J2000) / 36525.0

    # ── Fundamental arguments (degrees) ──────────────────────────────────────
    Lp = (218.3164477
          + 481267.88123421 * T
          - 0.0015786       * T**2
          + T**3 / 538841.0
          - T**4 / 65194000.0)
    D  = (297.8501921
          + 445267.1114034  * T
          - 0.0018819       * T**2
          + T**3 / 545868.0
          - T**4 / 113065000.0)
    M  = (357.5291092
          + 35999.0502909   * T
          - 0.0001536       * T**2
          + T**3 / 24490000.0)
    Mp = (134.9633964
          + 477198.8675055  * T
          + 0.0087414       * T**2
          + T**3 / 69699.0
          - T**4 / 14712000.0)
    F  = (93.2720950
          + 483202.0175233  * T
          - 0.0036539       * T**2
          - T**3 / 3526000.0
          + T**4 / 863310000.0)

    A1 = 119.75 + 131.849     * T
    A2 =  53.09 + 479264.290  * T
    A3 = 313.45 + 481266.484  * T

    # Eccentricity correction (applied to terms with |M_coeff| = 1 or 2)
    E  = 1.0 - 0.002516 * T - 0.0000074 * T**2

    def _r(deg: float) -> float:
        return math.radians(deg % 360.0)

    D_r, M_r, Mp_r, F_r = _r(D), _r(M), _r(Mp), _r(F)
    Lp_r, A1_r, A2_r, A3_r = _r(Lp), _r(A1), _r(A2), _r(A3)

    dpsi, deps = _nutation(T, Mp_r, M_r, F_r, D_r)

    # ── Σl and Σr ─────────────────────────────────────────────────────────────
    Sl = Sr = 0.0
    for D_c, M_c, Mp_c, F_c, sl, sr in _TABLE_LR:
        arg = D_c * D_r + M_c * M_r + Mp_c * Mp_r + F_c * F_r
        e   = E ** abs(M_c)
        Sl += e * sl * math.sin(arg)
        Sr += e * sr * math.cos(arg)

    Sl += (3958.0 * math.sin(A1_r)
           + 1962.0 * math.sin(Lp_r - F_r)
           +  318.0 * math.sin(A2_r))

    # ── Σb ────────────────────────────────────────────────────────────────────
    Sb = 0.0
    for D_c, M_c, Mp_c, F_c, sb in _TABLE_B:
        arg = D_c * D_r + M_c * M_r + Mp_c * Mp_r + F_c * F_r
        e   = E ** abs(M_c)
        Sb += e * sb * math.sin(arg)

    Sb += (-2235.0 * math.sin(Lp_r)
           +  382.0 * math.sin(A3_r)
           +  175.0 * math.sin(A1_r - F_r)
           +  175.0 * math.sin(A1_r + F_r)
           +  127.0 * math.sin(Lp_r - Mp_r)
           -  115.0 * math.sin(Lp_r + Mp_r))

    # ── Ecliptic longitude, latitude, distance ────────────────────────────────
    # dpsi converts mean → apparent (true) ecliptic longitude
    lam  = math.radians((Lp + Sl / 1e6) % 360.0) + dpsi
    beta = math.radians(Sb / 1e6)
    dist = 385000.56 + Sr / 1e3      # km

    # ── True obliquity of the ecliptic (mean + nutation in obliquity) ─────────
    eps = math.radians(
        23.439291111
        - 0.013004167 * T
        - 1.638889e-7 * T**2
        + 5.036111e-7 * T**3
    ) + deps
    

    # ── Ecliptic → mean equatorial of date ────────────────────────────────────
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    cos_l, sin_l = math.cos(lam),  math.sin(lam)
    cos_e, sin_e = math.cos(eps),  math.sin(eps)

    x = dist * cos_b * cos_l
    y = dist * (cos_b * sin_l * cos_e - sin_b * sin_e)
    z = dist * (cos_b * sin_l * sin_e + sin_b * cos_e)

    r_date = np.array([x, y, z])

    # ── Precess to J2000.0 ────────────────────────────────────────────────────
    _pr = _precess_to_j2000(r_date, T)
    print(round(jd,5), eps, _pr)
    return _pr


# ── Public API ────────────────────────────────────────────────────────────────

def moon_pos_eci(t: datetime) -> np.ndarray:
    """
    Moon position in ECI J2000.0 [km] for UTC datetime *t*.

    Parameters
    ----------
    t : datetime
        UTC epoch.  Naive datetimes are assumed UTC.

    Returns
    -------
    np.ndarray, shape (3,)
        [x, y, z] in km, ECI J2000.0 frame.
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return _moon_pos_eci_jd(datetime_to_jd(t))


def moon_state_eci(
    t: datetime,
    dt_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moon position [km] and velocity [km/s] in ECI J2000.0 for UTC datetime *t*.

    Velocity is estimated by a centred finite difference of width 2·*dt_s*.

    Parameters
    ----------
    t    : datetime  UTC epoch (naive datetimes assumed UTC).
    dt_s : float     Half-step for finite difference [seconds].  Default 1 s.

    Returns
    -------
    pos : np.ndarray shape (3,)  [km]
    vel : np.ndarray shape (3,)  [km/s]
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    jd  = datetime_to_jd(t)
    dj  = dt_s / 86400.0
    pos = _moon_pos_eci_jd(jd)
    vel = (_moon_pos_eci_jd(jd + dj) - _moon_pos_eci_jd(jd - dj)) / (2.0 * dt_s)
    return pos, vel


# ── ECR (ECEF) helpers ────────────────────────────────────────────────────────

def _gmst_rad(jd: float) -> float:
    """Greenwich Mean Sidereal Time [rad] for Julian Date *jd* (Meeus Ch. 12)."""
    D = jd - _J2000
    T = D / 36525.0
    theta = (280.46061837
             + 360.98564736629 * D
             + 0.000387933     * T**2
             - T**3 / 38710000.0)
    return math.radians(theta % 360.0)


def eci_to_ecr(pos_eci: np.ndarray, t: datetime) -> np.ndarray:
    """
    Rotate an ECI J2000.0 position vector to ECR (ECEF) via GMST.

    Parameters
    ----------
    pos_eci : np.ndarray shape (3,)  [any unit]
    t       : datetime  UTC epoch (naive assumed UTC).

    Returns
    -------
    np.ndarray shape (3,)  in the same unit as *pos_eci*.
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    theta = _gmst_rad(datetime_to_jd(t))
    return _rz(theta) @ pos_eci


def moon_pos_ecr(t: datetime) -> np.ndarray:
    """
    Moon position in ECR (ECEF) [km] for UTC datetime *t*.

    Parameters
    ----------
    t : datetime  UTC epoch (naive assumed UTC).

    Returns
    -------
    np.ndarray shape (3,)  [km]
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    jd    = datetime_to_jd(t)
    theta = _gmst_rad(jd)
    return _rz(theta) @ _moon_pos_eci_jd(jd)


def moon_state_ecr(
    t: datetime,
    dt_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moon position [km] and velocity [km/s] in ECR (ECEF) for UTC datetime *t*.

    The ECR velocity accounts for Earth's rotation:
        v_ecr = R(θ)·v_eci − ω_E × r_ecr

    Parameters
    ----------
    t    : datetime  UTC epoch (naive assumed UTC).
    dt_s : float     Finite-difference half-step [seconds].  Default 1 s.

    Returns
    -------
    pos : np.ndarray shape (3,)  [km]
    vel : np.ndarray shape (3,)  [km/s]
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    jd   = datetime_to_jd(t)
    dj   = dt_s / 86400.0
    theta = _gmst_rad(jd)
    R     = _rz(theta)

    pos_eci = _moon_pos_eci_jd(jd)
    vel_eci = (_moon_pos_eci_jd(jd + dj) - _moon_pos_eci_jd(jd - dj)) / (2.0 * dt_s)

    pos_ecr = R @ pos_eci
    vel_ecr = R @ vel_eci - np.cross(np.array([0., 0., _OMEGA_E]), pos_ecr)
    return pos_ecr, vel_ecr

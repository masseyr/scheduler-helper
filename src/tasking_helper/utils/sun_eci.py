"""
sun_eci -- Geocentric Sun ECI/ECR state vectors via VSOP87 (truncated).

Implements Meeus "Astronomical Algorithms" 2nd ed., Chapter 33 (VSOP87 theory
for Earth's heliocentric coordinates), plus IAU 1976 precession to J2000.0.

The Earth's heliocentric position (L, B, R) is computed from the VSOP87
series; the geocentric Sun position is the opposite vector:
    lam_sun = L + 180deg,  beta_sun = -B,  Delta = R

Typical accuracy over 2025-2050 (vs JPL DE432s):
  position  <~ 1 000 km  (VSOP87 truncated, ~1 arcsec)
  velocity  <~ 0.001 m/s  (central finite-difference, dt = 60 s)

For position accuracy < 10 km use sun_jpl.py (JPL DE kernel).

Dependencies: numpy, .jdate (datetime_to_jd)

Public API
----------
sun_pos_eci(t)    -> np.ndarray (3,)  [km]        ECI J2000.0
sun_state_eci(t)  -> (pos [km], vel [km/s])       ECI J2000.0
sun_pos_ecr(t)    -> np.ndarray (3,)  [km]        ECR (ECEF)
sun_state_ecr(t)  -> (pos [km], vel [km/s])       ECR (ECEF)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .jdate import datetime_to_jd

__all__ = [
    "sun_pos_eci",
    "sun_state_eci",
    "sun_pos_ecr",
    "sun_state_ecr",
]

_J2000   = 2_451_545.0
_AU_KM   = 149_597_870.7      # km per astronomical unit
_OMEGA_E = 7.292115e-5         # Earth rotation rate [rad/s]

# -- VSOP87 Earth heliocentric series (Meeus Table 33.A) -----------------------
# Each sub-table: rows of (A, B, C); contribution = A * cos(B + C * tau)
# tau = (JDE - J2000) / 365250  [Julian Ephemeris Millennia from J2000.0]
# L [rad] = (Sigma Ln * tau^n) / 1e8
# B [rad] = (Sigma Bn * tau^n) / 1e8
# R [AU]  = (Sigma Rn * tau^n) / 1e8

_L0 = (
    (175347046, 0,         0           ),
    (3341656,   4.6692568, 6283.0758500),
    (34894,     4.62610,   12566.15170 ),
    (3497,      2.7441,    5753.3849   ),
    (3418,      2.8289,    3.5230      ),
    (3136,      3.6277,    77713.7715  ),
    (2676,      4.4181,    7860.4194   ),
    (2343,      6.1352,    3930.2097   ),
    (1324,      0.7425,    11506.7698  ),
    (1273,      2.0371,    529.6910    ),
    (1199,      1.1096,    1577.3435   ),
    (990,       5.233,     5884.927    ),
    (902,       2.045,     26.298      ),
    (857,       3.508,     398.149     ),
    (780,       1.179,     5223.694    ),
    (753,       2.533,     5507.553    ),
    (505,       4.583,     18849.228   ),
    (492,       4.205,     775.523     ),
    (357,       2.920,     0.067       ),
    (317,       5.849,     11790.629   ),
    (284,       1.899,     796.298     ),
    (271,       0.315,     10977.079   ),
    (243,       0.345,     5486.778    ),
    (206,       4.806,     2544.314    ),
    (205,       1.869,     5573.143    ),
    (202,       2.458,     6069.777    ),
    (156,       0.833,     213.299     ),
    (132,       3.411,     2942.463    ),
    (126,       1.083,     20.775      ),
    (115,       0.645,     0.980       ),
    (103,       0.636,     4694.003    ),
    (99,        6.21,      15720.84    ),
    (98,        0.68,      7084.90     ),
    (86,        5.98,      161000.69   ),
    (86,        1.27,      17260.15    ),
    (65,        1.43,      17789.84    ),
    (63,        1.05,      5088.63     ),
    (57,        3.44,      7860.42     ),
    (56,        4.39,      4690.48     ),
    (49,        0.49,      6496.37     ),
    (45,        3.63,      6309.38     ),
    (43,        1.04,      83996.85    ),
    (39,        3.44,      4292.33     ),
    (38,        2.78,      12139.55    ),
    (37,        1.78,      5088.63     ),
    (37,        1.01,      12036.46    ),
)

_L1 = (
    (628331966747, 0,        0           ),
    (206059,       2.678235, 6283.075850 ),
    (4303,         2.6351,   12566.1517  ),
    (425,          1.590,    3.523       ),
    (119,          5.796,    26.298      ),
    (109,          2.966,    1577.344    ),
    (93,           2.59,     18849.23    ),
    (72,           1.14,     529.69      ),
    (68,           1.87,     398.15      ),
    (67,           4.41,     5507.55     ),
    (59,           2.89,     5223.69     ),
    (56,           2.17,     155.42      ),
    (45,           0.40,     796.30      ),
    (36,           0.47,     775.52      ),
    (29,           2.65,     7.11        ),
    (21,           5.34,     0.98        ),
    (19,           1.85,     5486.78     ),
    (19,           4.97,     213.30      ),
    (17,           2.99,     6275.96     ),
    (16,           0.03,     2544.31     ),
    (16,           1.43,     2146.17     ),
    (15,           1.21,     10977.08    ),
    (12,           2.83,     1748.02     ),
    (12,           3.26,     5088.63     ),
    (12,           5.27,     1194.45     ),
    (12,           2.08,     4694.00     ),
    (11,           0.77,     553.57      ),
    (10,           1.30,     6286.60     ),
    (10,           4.24,     1349.87     ),
    (9,            2.70,     242.73      ),
    (9,            5.64,     951.72      ),
    (8,            5.30,     2352.87     ),
    (6,            2.65,     9437.76     ),
    (6,            4.67,     4690.48     ),
)

_L2 = (
    (52919,  0,    0        ),
    (8720,   1.0721, 6283.0758),
    (309,    0.867, 12566.152),
    (27,     0.05,  3.52     ),
    (16,     5.19,  26.30    ),
    (16,     3.68,  155.42   ),
    (10,     0.76,  18849.23 ),
)

_L3 = (
    (289, 5.844, 6283.076 ),
    (35,  0,     0        ),
    (17,  5.49,  12566.15 ),
)

_L4 = (
    (114, 3.1416, 0),
)

_L5 = (
    (1, 3.14, 0),
)

_B0 = (
    (280, 3.199, 84334.662),
    (102, 5.422, 5507.553 ),
    (80,  3.88,  5223.69  ),
    (44,  3.70,  2352.87  ),
    (32,  4.00,  1577.34  ),
)

_B1 = (
    (9, 3.90, 5507.55),
    (6, 1.73, 5223.69),
)

_R0 = (
    (100013989, 0,         0           ),
    (1670700,   3.0984635, 6283.075850 ),
    (13956,     3.05525,   12566.1517  ),
    (3084,      5.1985,    77713.7715  ),
    (1628,      1.1739,    5753.3849   ),
    (1576,      2.8469,    7860.4194   ),
    (925,       5.453,     11506.770   ),
    (542,       4.564,     3930.210    ),
    (472,       3.661,     5884.927    ),
    (346,       0.964,     5507.553    ),
    (329,       5.900,     5223.694    ),
    (307,       0.299,     5573.143    ),
    (243,       4.273,     11790.629   ),
    (212,       5.847,     1577.344    ),
    (186,       5.022,     10977.079   ),
    (175,       3.012,     18849.228   ),
    (110,       5.055,     5486.778    ),
    (98,        0.89,      6069.78     ),
    (86,        5.69,      15720.84    ),
    (86,        1.27,      161000.69   ),
    (65,        0.27,      17789.84    ),
    (63,        0.92,      529.69      ),
    (57,        2.01,      83996.85    ),
    (56,        5.24,      71430.70    ),
    (49,        3.25,      2544.31     ),
    (47,        2.58,      775.52      ),
    (45,        5.54,      9437.76     ),
    (43,        6.01,      10447.39    ),
    (39,        5.36,      5573.14     ),
    (38,        2.39,      1748.02     ),
    (37,        0.83,      7084.90     ),
    (37,        4.90,      14712.32    ),
    (36,        1.67,      4690.48     ),
    (35,        1.84,      4292.33     ),
    (33,        0.24,      6275.96     ),
    (32,        0.18,      12139.55    ),
    (32,        1.78,      16730.46    ),
    (28,        1.21,      5088.63     ),
    (28,        1.90,      398.15      ),
    (27,        5.09,      11243.69    ),
)

_R1 = (
    (103019, 1.107490, 6283.075850),
    (1721,   1.0644,   12566.1517 ),
    (702,    3.142,    0          ),
    (32,     1.02,     18849.23   ),
    (31,     2.84,     5507.55    ),
    (25,     1.32,     5223.69    ),
    (18,     1.42,     1577.34    ),
    (10,     5.91,     10977.08   ),
    (9,      1.42,     6275.96    ),
    (9,      0.27,     5486.78    ),
)

_R2 = (
    (4359, 5.7846, 6283.0758),
    (124,  5.579,  12566.152),
    (12,   3.14,   0        ),
    (9,    3.63,   77713.77 ),
)

_R3 = (
    (145, 4.273, 6283.076 ),
    (7,   3.92,  12566.15 ),
)

_R4 = (
    (4, 2.56, 6283.08),
)


# -- Internal helpers ----------------------------------------------------------

def _vsop_sum(terms: tuple, tau: float) -> float:
    return sum(A * math.cos(B + C * tau) for A, B, C in terms)


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
    """Rotate mean-equatorial-of-date vector to J2000.0 (IAU 1976 precession)."""
    as2r = math.pi / (180.0 * 3600.0)
    zeta_A  = (2306.2181 * T + 1.39656  * T**2 - 0.000139 * T**3) * as2r
    theta_A = (2004.3109 * T - 0.85330  * T**2 - 0.000217 * T**3) * as2r
    z_A     = (2306.2181 * T + 1.09468  * T**2 + 0.018203 * T**3) * as2r
    P = _rz(-z_A) @ _ry(theta_A) @ _rz(-zeta_A)
    return P.T @ r_date


def _sun_pos_eci_jd(jd: float) -> np.ndarray:
    """Sun geometric position [km] in ECI J2000.0 for Julian Date *jd*."""
    tau = (jd - _J2000) / 365250.0   # Julian Ephemeris Millennia
    T   = tau * 10.0                  # Julian Centuries

    # -- Earth heliocentric longitude L [rad] ----------------------------------
    L0 = _vsop_sum(_L0, tau)
    L1 = _vsop_sum(_L1, tau)
    L2 = _vsop_sum(_L2, tau)
    L3 = _vsop_sum(_L3, tau)
    L4 = _vsop_sum(_L4, tau)
    L5 = _vsop_sum(_L5, tau)
    L = (L0 + L1*tau + L2*tau**2 + L3*tau**3 + L4*tau**4 + L5*tau**5) / 1e8
    L = L % (2.0 * math.pi)

    # -- Earth heliocentric latitude B [rad] -----------------------------------
    B0 = _vsop_sum(_B0, tau)
    B1 = _vsop_sum(_B1, tau)
    B = (B0 + B1*tau) / 1e8

    # -- Earth-Sun distance R [AU] ---------------------------------------------
    R0 = _vsop_sum(_R0, tau)
    R1 = _vsop_sum(_R1, tau)
    R2 = _vsop_sum(_R2, tau)
    R3 = _vsop_sum(_R3, tau)
    R4 = _vsop_sum(_R4, tau)
    R = (R0 + R1*tau + R2*tau**2 + R3*tau**3 + R4*tau**4) / 1e8  # AU

    # -- Geocentric Sun: flip direction, negate latitude -----------------------
    lam = (L + math.pi) % (2.0 * math.pi)   # geocentric ecliptic longitude
    beta = -B                                 # geocentric ecliptic latitude
    R_km = R * _AU_KM

    # -- Mean obliquity of the ecliptic ----------------------------------------
    eps = math.radians(
        23.439291111
        - 0.013004167 * T
        - 1.638889e-7 * T**2
        + 5.036111e-7 * T**3
    )

    # -- Ecliptic -> mean equatorial of date ------------------------------------
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    cos_l, sin_l = math.cos(lam),  math.sin(lam)
    cos_e, sin_e = math.cos(eps),  math.sin(eps)

    x = R_km * cos_b * cos_l
    y = R_km * (cos_b * sin_l * cos_e - sin_b * sin_e)
    z = R_km * (cos_b * sin_l * sin_e + sin_b * cos_e)

    # -- Precess to J2000.0 ----------------------------------------------------
    return _precess_to_j2000(np.array([x, y, z]), T)


def _gmst_rad(jd: float) -> float:
    """Greenwich Mean Sidereal Time [rad] (Meeus Ch. 12)."""
    D = jd - _J2000
    T = D / 36525.0
    theta = (280.46061837
             + 360.98564736629 * D
             + 0.000387933     * T**2
             - T**3 / 38710000.0)
    return math.radians(theta % 360.0)


# -- Public API ----------------------------------------------------------------

def sun_pos_eci(t: datetime) -> np.ndarray:
    """
    Sun geometric position in ECI J2000.0 [km] for UTC datetime *t*.

    Parameters
    ----------
    t : datetime  UTC epoch.  Naive datetimes are assumed UTC.

    Returns
    -------
    np.ndarray, shape (3,)  [x, y, z] in km, ECI J2000.0.
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return _sun_pos_eci_jd(datetime_to_jd(t))


def sun_state_eci(
    t: datetime,
    dt_s: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sun position [km] and velocity [km/s] in ECI J2000.0 for UTC datetime *t*.

    Velocity is estimated by a centred finite difference of width 2**dt_s*.

    Parameters
    ----------
    t    : datetime  UTC epoch (naive assumed UTC).
    dt_s : float     Half-step for finite difference [seconds].  Default 60 s.

    Returns
    -------
    pos : np.ndarray shape (3,)  [km]
    vel : np.ndarray shape (3,)  [km/s]
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    jd  = datetime_to_jd(t)
    dj  = dt_s / 86400.0
    pos = _sun_pos_eci_jd(jd)
    vel = (_sun_pos_eci_jd(jd + dj) - _sun_pos_eci_jd(jd - dj)) / (2.0 * dt_s)
    return pos, vel


def sun_pos_ecr(t: datetime) -> np.ndarray:
    """
    Sun position in ECR (ECEF) [km] for UTC datetime *t*.

    Returns
    -------
    np.ndarray shape (3,)  [km]
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    jd    = datetime_to_jd(t)
    theta = _gmst_rad(jd)
    R     = np.array([[ math.cos(theta), math.sin(theta), 0.],
                      [-math.sin(theta), math.cos(theta), 0.],
                      [0.,               0.,               1.]])
    return R @ _sun_pos_eci_jd(jd)


def sun_state_ecr(
    t: datetime,
    dt_s: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sun position [km] and velocity [km/s] in ECR (ECEF) for UTC datetime *t*.

    ECR velocity accounts for Earth's rotation:
        v_ecr = R(theta)*v_eci - omega_E x r_ecr

    Note: |v_ecr| ~= 10 900 km/s at 1 AU (Earth's daily rotation dominates).

    Returns
    -------
    pos : np.ndarray shape (3,)  [km]
    vel : np.ndarray shape (3,)  [km/s]
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    jd    = datetime_to_jd(t)
    dj    = dt_s / 86400.0
    theta = _gmst_rad(jd)
    R     = np.array([[ math.cos(theta), math.sin(theta), 0.],
                      [-math.sin(theta), math.cos(theta), 0.],
                      [0.,               0.,               1.]])

    pos_eci = _sun_pos_eci_jd(jd)
    vel_eci = (_sun_pos_eci_jd(jd + dj) - _sun_pos_eci_jd(jd - dj)) / (2.0 * dt_s)

    pos_ecr = R @ pos_eci
    vel_ecr = R @ vel_eci - np.cross(np.array([0., 0., _OMEGA_E]), pos_ecr)
    return pos_ecr, vel_ecr

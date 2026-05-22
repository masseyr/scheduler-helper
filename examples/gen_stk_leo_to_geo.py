"""
gen_stk_leo_to_geo.py -- Generate STK ephemeris (.e) files for a LEO-to-GEO
                          Hohmann transfer and an optional ASAT direct-ascent
                          trajectory in J2000 ECI.

Hohmann trajectory design
-------------------------
The transfer ellipse has:
    Perigee : 300 km  (LEO insertion altitude)
    Apogee  : 35 786 km  (geostationary altitude)

The Keplerian half-period (perigee -> apogee) is ~5.27 h, so a 6-hour
simulation captures the complete outbound parabolic trajectory from LEO to
GEO and a short return leg.

The 300-km starting point can be placed at any geographic location by
supplying --start-lat and --start-lon.  RAAN and argument of perigee are
derived automatically so that the orbit passes through that point at t=0
(ascending pass through the perigee).

ASAT direct-ascent trajectory
------------------------------
An optional non-ballistic direct-ascent interceptor trajectory can be
generated alongside the transfer.  The interceptor launches from a ground
site (default: Cape Canaveral 28.5 N, 80.6 W) and ascends along a
great-circle arc in the ECI frame, reaching the target's position at the
specified intercept time.

Output
------
STK v10 EphemerisTimePosVel files, 10-second cadence, coordinate system J2000.

Usage
-----
    python examples/gen_stk_leo_to_geo.py [options] [output]

    --start-lat LAT   Geodetic latitude of 300-km perigee start point [deg]
    --start-lon LON   Longitude of 300-km perigee start point [deg]
    --asat-out  PATH  Also generate an ASAT direct-ascent file at this path
    --asat-lat  LAT   ASAT launch site latitude  [deg, default 28.5]
    --asat-lon  LON   ASAT launch site longitude [deg, default -80.6]
    --intercept-t  S  Intercept time from epoch  [s,   default 1800]

    Default Hohmann output: leo_to_geo_transfer.e in the current directory.
"""

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# -- Project imports ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tasking_helper.utils.keplerian import keplerian_to_state, solve_kepler, lla_to_eci

# ---------------------------------------------------------------------------
# Orbit parameters
# ---------------------------------------------------------------------------

R_EARTH = 6378.137       # WGS-84 equatorial radius [km]
MU      = 398600.4418    # Earth gravitational parameter [km^3/s^2]

ALT_PERIGEE_KM = 300.0            # LEO perigee altitude
ALT_APOGEE_KM  = 35_786.0        # GEO apogee altitude

r_p = R_EARTH + ALT_PERIGEE_KM   # perigee radius [km]
r_a = R_EARTH + ALT_APOGEE_KM    # apogee radius [km]

A_KM = (r_p + r_a) / 2.0         # semi-major axis
ECC  = (r_a - r_p) / (r_a + r_p) # eccentricity

INCL_DEG = 28.5   # inclination

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

EPOCH      = datetime(2027, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
DURATION_S = 6 * 3600   # 6 hours
DT_S       = 1.0        # cadence [s]

# ---------------------------------------------------------------------------
# STK file helpers
# ---------------------------------------------------------------------------

_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar",  4: "Apr",  5: "May",  6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _stk_epoch(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return (f"{dt.day} {_MONTH_ABBR[dt.month]} {dt.year} "
            f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.{ms:03d}")


def write_stk_ephem(path: str,
                    epoch: datetime,
                    time_pos_vel: list,
                    comments: list = None) -> None:
    """Write an STK v10 EphemerisTimePosVel file."""
    n_pts = len(time_pos_vel)
    lines = [
        "stk.v.12.0\n\n"
    ]
    if comments:
        for c in comments:
            lines.append(f"# {c}")
    
    lines += [
        "\nBEGIN Ephemeris\n",
        f"  NumberOfEphemerisPoints  {n_pts}\n",
        f"  ScenarioEpoch            {_stk_epoch(epoch)}\n",
        "  InterpolationMethod      Lagrange\n",
        "  InterpolationOrder       8\n",
        "  CentralBody              Earth\n",
        "  CoordinateSystem         J2000\n",
        "    DistanceUnit             Kilometers\n",
    ]

    lines.append("  EphemerisTimePosVel\n")


    for t_s, pos, vel in time_pos_vel:
        lines.append(
            f"{t_s:.16E}"
            f" {pos[0]:.16E} {pos[1]:.16E} {pos[2]:.16E}"
            f" {vel[0]:.16E} {vel[1]:.16E} {vel[2]:.16E}"
        )

    lines += ["\nEND Ephemeris\n"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def _slerp(u0: np.ndarray, u1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit vectors at fraction t."""
    omega = math.acos(max(-1.0, min(1.0, float(np.dot(u0, u1)))))
    if omega < 1e-10:
        return u0 + t * (u1 - u0)
    s = math.sin(omega)
    return (math.sin((1.0 - t) * omega) * u0 + math.sin(t * omega) * u1) / s


def start_point_to_elements(
    lat_deg: float,
    lon_deg: float,
    incl_deg: float,
    epoch: datetime,
) -> tuple:
    """
    Compute (RAAN_deg, ARGP_deg) for a Hohmann orbit whose perigee (300 km)
    lies at the given geographic location at t=0 (ascending pass).

    Uses the constraint that the orbit plane normal is perpendicular to the
    perigee position vector to solve for RAAN, then derives ARGP from the
    angle between the ascending node and the perigee direction.

    Raises ValueError if |geocentric declination| > inclination.
    """
    r_eci = np.array(lla_to_eci(lat_deg, lon_deg, ALT_PERIGEE_KM, epoch))
    r     = float(np.linalg.norm(r_eci))
    e_hat = r_eci / r

    sin_delta = r_eci[2] / r
    incl      = math.radians(incl_deg)
    sin_incl  = math.sin(incl)
    cos_incl  = math.cos(incl)
    cos_delta = math.sqrt(max(0.0, 1.0 - sin_delta ** 2))

    if abs(sin_delta) > abs(sin_incl) + 1e-9:
        raise ValueError(
            f"Start latitude {lat_deg:.2f}° unreachable at inclination {incl_deg:.2f}° "
            f"(geocentric dec {math.degrees(math.asin(sin_delta)):.2f}°)"
        )

    # Solve sin(RAAN - phi) = -cot(i) * tan(delta)
    phi   = math.atan2(r_eci[1], r_eci[0])
    ratio = -(cos_incl / sin_incl) * (sin_delta / cos_delta) if cos_delta > 1e-10 else 0.0
    alpha = math.asin(max(-1.0, min(1.0, ratio)))

    raan_candidates = [
        (phi + alpha)              % (2 * math.pi),
        (phi + math.pi - alpha)   % (2 * math.pi),
    ]

    # Select the ascending-pass solution (northward velocity at start)
    raan = raan_candidates[0]
    for raan_c in raan_candidates:
        h_hat = np.array([sin_incl * math.sin(raan_c),
                          -sin_incl * math.cos(raan_c),
                          cos_incl])
        if np.cross(h_hat, e_hat)[2] >= 0.0:
            raan = raan_c
            break

    h_hat = np.array([sin_incl * math.sin(raan),
                      -sin_incl * math.cos(raan),
                      cos_incl])

    # Ascending node direction: N = z x h
    N     = np.cross(np.array([0.0, 0.0, 1.0]), h_hat)
    N_mag = float(np.linalg.norm(N))
    N_hat = N / N_mag if N_mag > 1e-10 else np.array([1.0, 0.0, 0.0])

    # ARGP: signed angle from ascending node to perigee direction in orbital plane
    argp = math.atan2(
        float(np.dot(np.cross(N_hat, e_hat), h_hat)),
        float(np.dot(N_hat, e_hat)),
    )

    return math.degrees(raan) % 360.0, math.degrees(argp) % 360.0


# ---------------------------------------------------------------------------
# Keplerian propagation
# ---------------------------------------------------------------------------

def propagate_keplerian(
    a_km:      float,
    ecc:       float,
    incl_deg:  float,
    raan_deg:  float,
    argp_deg:  float,
    M0_deg:    float,
    duration_s: float,
    dt_s:       float,
) -> list:
    """Propagate a two-body Keplerian orbit; return [(t_s, pos_km, vel_kms)]."""
    n_rad_s = math.sqrt(MU / a_km ** 3)
    ts      = np.arange(0.0, duration_s + dt_s * 0.5, dt_s)
    rows    = []
    for t_s in ts:
        M_deg = (M0_deg + math.degrees(n_rad_s * t_s)) % 360.0
        pos, vel = keplerian_to_state(
            a_km=a_km, ecc=ecc,
            incl_deg=incl_deg, raan_deg=raan_deg,
            argp_deg=argp_deg, M_deg=M_deg,
        )
        rows.append((float(t_s), pos, vel))
    return rows


# ---------------------------------------------------------------------------
# ASAT direct-ascent trajectory
# ---------------------------------------------------------------------------

def _append_orbital_insertion(
    pos_arr: np.ndarray,
    vel_arr: np.ndarray,
    ts: np.ndarray,
    r_orbit: float,
    speed_orbit: float,
    dt_s: float,
    n_transition: int = 50,
    n_orbital: int = 100,
) -> tuple:
    """
    Append a smooth orbital insertion to a climbing trajectory.

    Transition phase (n_transition steps):
        The velocity direction rotates from the current heading to purely
        tangential using cosine blending.  Position stays on the sphere of
        radius r_orbit by advancing only the tangential component each step.

    Orbital phase (n_orbital steps):
        Analytical circular motion at constant radius r_orbit and constant
        speed speed_orbit in the plane established at the end of the transition.

    The junction with the climbing phase is C1 continuous: position and
    velocity are matched at the seam.

    Returns
    -------
    (pos_arr, vel_arr, ts) — extended numpy arrays
    """
    p_end = pos_arr[-1].copy()
    v_end = vel_arr[-1].copy()
    t_end = float(ts[-1])

    r_hat = p_end / np.linalg.norm(p_end)

    # Tangential direction: remove radial component of v_end
    v_tan_vec = v_end - np.dot(v_end, r_hat) * r_hat
    v_tan_mag = float(np.linalg.norm(v_tan_vec))
    if v_tan_mag < 1e-10:
        ref   = np.array([0.0, 1.0, 0.0]) if abs(r_hat[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        t_hat = np.cross(r_hat, ref)
    else:
        t_hat = v_tan_vec / v_tan_mag
    t_hat = t_hat / np.linalg.norm(t_hat)

    v_hat_start = v_end / np.linalg.norm(v_end)

    # --- Transition phase ---------------------------------------------------
    pos_trans = []
    vel_trans = []
    cur_dir   = r_hat.copy()   # unit position direction on sphere

    for k in range(1, n_transition + 1):
        theta = k / n_transition
        blend = 0.5 * (1.0 - math.cos(math.pi * theta))   # cosine ease-in-out

        v_dir = _slerp(v_hat_start, t_hat, blend)
        v_dir = v_dir / np.linalg.norm(v_dir)
        v_new = speed_orbit * v_dir

        # Advance position on sphere using only the tangential velocity component
        v_tan_now = v_new - np.dot(v_new, cur_dir) * cur_dir
        dangle = np.linalg.norm(v_tan_now) * dt_s / r_orbit
        if np.linalg.norm(v_tan_now) > 1e-10:
            axis    = v_tan_now / np.linalg.norm(v_tan_now)
            cur_dir = cur_dir * math.cos(dangle) + axis * math.sin(dangle)
            cur_dir = cur_dir / np.linalg.norm(cur_dir)

        pos_trans.append(r_orbit * cur_dir)
        vel_trans.append(v_new)

    # --- Orbital phase ------------------------------------------------------
    r_orb_hat = cur_dir.copy()
    # Ensure t_hat is perpendicular to the final position direction
    t_orb_hat = t_hat - np.dot(t_hat, r_orb_hat) * r_orb_hat
    t_orb_hat = t_orb_hat / np.linalg.norm(t_orb_hat)

    omega = speed_orbit / r_orbit   # angular velocity [rad/s]

    pos_orb = []
    vel_orb = []
    for k in range(n_orbital):
        tau   = k * dt_s
        angle = omega * tau
        p = r_orbit * ( math.cos(angle) * r_orb_hat + math.sin(angle) * t_orb_hat)
        v = speed_orbit * (-math.sin(angle) * r_orb_hat + math.cos(angle) * t_orb_hat)
        pos_orb.append(p)
        vel_orb.append(v)

    # --- Assemble -----------------------------------------------------------
    n_extra = n_transition + n_orbital
    ts_extra = t_end + dt_s * np.arange(1, n_extra + 1)

    pos_all = np.vstack(
        [pos_arr] +
        [p.reshape(1, 3) for p in pos_trans] +
        [p.reshape(1, 3) for p in pos_orb]
    )
    vel_all = np.vstack(
        [vel_arr] +
        [v.reshape(1, 3) for v in vel_trans] +
        [v.reshape(1, 3) for v in vel_orb]
    )
    ts_all = np.concatenate([ts, ts_extra])

    return pos_all, vel_all, ts_all


def generate_asat_trajectory(
    target_rows: list,
    launch_lat_deg: float,
    launch_lon_deg: float,
    launch_alt_km: float,
    t_intercept_s: float,
    epoch: datetime,
    dt_s: float = 1.0,
    speed_end_kms: float = 2.0,
) -> tuple:
    """
    Direct-ascent ASAT trajectory with Newtonian gravitational deceleration.

    The interceptor follows a great-circle arc in the J2000 ECI frame from the
    launch site to the target position.  Speed at each point is given by energy
    conservation (two-body gravity):

        v(r)^2 = v0^2 + 2*MU*(1/r - 1/r_launch)

    The launch speed v0 is derived so that v(r_target) = speed_end_kms.
    Flight time T_flight is computed by integrating ds/v(r) along the arc;
    it may differ from t_intercept_s (which is used only to pick the target
    position from the Hohmann rows).

    Parameters
    ----------
    target_rows    : Hohmann rows list of (t_s, pos_km, vel_kms)
    launch_lat_deg : geodetic latitude of launch site [deg]
    launch_lon_deg : longitude of launch site [deg]
    launch_alt_km  : altitude of launch site [km]
    t_intercept_s  : time used to sample the target position from the Hohmann orbit [s]
    epoch          : UTC datetime for t=0
    dt_s           : time step [s]
    speed_end_kms  : speed at intercept [km/s]; determines launch speed via energy conservation

    Returns
    -------
    (rows, v0_kms, v_end_kms, T_flight_s) where rows = [(t_s, pos_km, vel_kms)]
    """
    # Interpolate target ECI position at t_intercept_s
    t_arr    = np.array([r[0] for r in target_rows])
    idx      = int(np.clip(np.searchsorted(t_arr, t_intercept_s), 1, len(target_rows) - 1))
    t0, p0, _ = target_rows[idx - 1]
    t1, p1, _ = target_rows[idx]
    frac_int = (t_intercept_s - float(t0)) / (float(t1) - float(t0)) if t1 != t0 else 0.0
    r_target = np.array(p0, dtype=float) + frac_int * (np.array(p1) - np.array(p0))

    # Launch site ECI position at epoch
    r_launch = np.array(lla_to_eci(launch_lat_deg, launch_lon_deg, launch_alt_km, epoch),
                        dtype=float)

    r_mag_0 = float(np.linalg.norm(r_launch))
    r_mag_1 = float(np.linalg.norm(r_target))
    u0      = r_launch / r_mag_0
    u1      = r_target / r_mag_1

    # Build fine SLERP path and compute cumulative arc length and radii
    N_fine     = 10_000
    fracs_fine = np.linspace(0.0, 1.0, N_fine)
    fine_pos   = np.array([
        (r_mag_0 + (r_mag_1 - r_mag_0) * f) * _slerp(u0, u1, f)
        for f in fracs_fine
    ])
    radii      = np.linalg.norm(fine_pos, axis=1)
    arc_cum    = np.zeros(N_fine)
    arc_cum[1:] = np.cumsum(np.linalg.norm(np.diff(fine_pos, axis=0), axis=1))

    # Launch speed from energy conservation: v0^2 = v_end^2 + 2*MU*(1/r0 - 1/r1)
    delta_grav = 2.0 * MU * (1.0 / r_mag_0 - 1.0 / r_mag_1)
    v0_sq = speed_end_kms ** 2 + delta_grav
    if v0_sq < 0:
        raise ValueError(
            f"speed_end_kms={speed_end_kms:.2f} km/s too large: vehicle cannot reach "
            f"r_target={r_mag_1:.1f} km from r_launch={r_mag_0:.1f} km"
        )
    v0 = math.sqrt(v0_sq)

    # Speed at each fine position via energy conservation
    v_fine = np.sqrt(np.maximum(1e-12, v0 ** 2 + 2.0 * MU * (1.0 / radii - 1.0 / r_mag_0)))

    # Integrate dt = ds / v to get time as function of arc position
    ds       = np.diff(arc_cum)
    v_mid    = (v_fine[:-1] + v_fine[1:]) / 2.0
    time_cum = np.zeros(N_fine)
    time_cum[1:] = np.cumsum(ds / np.maximum(v_mid, 1e-12))
    T_flight = float(time_cum[-1])

    # Resample climbing phase at regular time steps by inverting t(arc) → arc(t)
    ts       = np.arange(0.0, T_flight + dt_s * 0.5, dt_s)
    arc_at_t = np.interp(ts, time_cum, arc_cum)
    pos_arr  = np.column_stack([
        np.interp(arc_at_t, arc_cum, fine_pos[:, i]) for i in range(3)
    ])
    vel_arr  = np.gradient(pos_arr, dt_s, axis=0)

    # Append smooth orbital insertion (transition + constant-radius orbit)
    pos_arr, vel_arr, ts = _append_orbital_insertion(
        pos_arr, vel_arr, ts, r_mag_1, speed_end_kms, dt_s,
    )

    rows = [(float(ts[i]), pos_arr[i], vel_arr[i]) for i in range(len(ts))]
    return rows, float(v0), float(v_fine[-1]), float(T_flight)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate STK .e ephemeris files for a LEO-to-GEO Hohmann "
                    "transfer and optional ASAT direct-ascent trajectory."
    )
    ap.add_argument(
        "output", nargs="?", default="leo_to_geo_transfer.e",
        help="Hohmann transfer output file [default: leo_to_geo_transfer.e]",
    )
    ap.add_argument(
        "--start-lat", type=float, default=None, metavar="DEG",
        help="Geodetic latitude of the 300-km perigee start point [deg]",
    )
    ap.add_argument(
        "--start-lon", type=float, default=None, metavar="DEG",
        help="Longitude of the 300-km perigee start point [deg]",
    )
    ap.add_argument(
        "--asat-out", default=None, metavar="PATH",
        help="Write ASAT direct-ascent trajectory to this .e file",
    )
    ap.add_argument(
        "--asat-lat", type=float, default=28.5, metavar="DEG",
        help="ASAT launch site latitude [deg, default 28.5 = KSC]",
    )
    ap.add_argument(
        "--asat-lon", type=float, default=-80.6, metavar="DEG",
        help="ASAT launch site longitude [deg, default -80.6 = KSC]",
    )
    ap.add_argument(
        "--intercept-t", type=float, default=1800.0, metavar="S",
        help="ASAT intercept time from epoch [s, default 1800 = 30 min]",
    )
    ap.add_argument(
        "--asat-start-alt", type=float, default=0.0, metavar="KM",
        help="ASAT launch site altitude [km, default 0 = ground]",
    )
    ap.add_argument(
        "--speed-end", type=float, default=2.0, metavar="KMS",
        help="ASAT intercept speed [km/s, default 2.0]; launch speed derived via energy conservation",
    )
    args = ap.parse_args()

    # Orbital elements: RAAN and ARGP from start-point or defaults
    if args.start_lat is not None:
        lat = args.start_lat
        lon = args.start_lon if args.start_lon is not None else 0.0
        raan_deg, argp_deg = start_point_to_elements(lat, lon, INCL_DEG, EPOCH)
        print(f"Start point       : lat={lat:.2f} deg  lon={lon:.2f} deg  alt={ALT_PERIGEE_KM:.0f} km")
        print(f"  Derived RAAN    : {raan_deg:.4f} deg")
        print(f"  Derived ARGP    : {argp_deg:.4f} deg")
        print()
    else:
        raan_deg = 0.0
        argp_deg = 0.0

    M0_DEG = 0.0   # start at perigee

    n      = math.sqrt(MU / A_KM ** 3)
    T_s    = 2 * math.pi / n
    T_half = T_s / 2.0

    print("Hohmann transfer parameters")
    print(f"  Perigee altitude  : {ALT_PERIGEE_KM:.0f} km  (r = {r_p:.3f} km)")
    print(f"  Apogee altitude   : {ALT_APOGEE_KM:.0f} km  (r = {r_a:.3f} km)")
    print(f"  Semi-major axis   : {A_KM:.3f} km")
    print(f"  Eccentricity      : {ECC:.6f}")
    print(f"  Inclination       : {INCL_DEG} deg")
    print(f"  RAAN              : {raan_deg:.4f} deg")
    print(f"  ARGP              : {argp_deg:.4f} deg")
    print(f"  Orbital period    : {T_s/3600:.4f} h")
    print(f"  Half-period       : {T_half/3600:.4f} h  (perigee -> GEO apogee)")
    print(f"  Simulation        : {DURATION_S/3600:.1f} h  ({DURATION_S/DT_S:.0f} steps at {DT_S:.0f} s)")

    M_end = (M0_DEG + math.degrees(n * DURATION_S)) % 360.0
    E_end = math.radians(solve_kepler(M_end, ECC))
    r_end = A_KM * (1.0 - ECC * math.cos(E_end))
    print(f"  End altitude      : {r_end - R_EARTH:.0f} km  "
          f"(M = {M_end:.1f} deg, {'past apogee' if M_end > 180 else 'before apogee'})")
    print()

    print("Propagating Hohmann trajectory ... ", end="", flush=True)
    rows = propagate_keplerian(
        A_KM, ECC, INCL_DEG, raan_deg, argp_deg, M0_DEG,
        DURATION_S, DT_S,
    )
    print(f"{len(rows)} points")

    comments = [
        f"Hohmann transfer: {ALT_PERIGEE_KM:.0f} km LEO -> {ALT_APOGEE_KM:.0f} km GEO",
        f"a={A_KM:.3f} km  ecc={ECC:.6f}  incl={INCL_DEG} deg",
        f"RAAN={raan_deg:.4f} deg  ARGP={argp_deg:.4f} deg",
        f"Half-period (perigee to GEO apogee) = {T_half/3600:.4f} h",
        f"Simulation covers {DURATION_S/3600:.1f} h starting at perigee (M=0)",
        "Unperturbed two-body Keplerian propagation",
    ]
    write_stk_ephem(args.output, EPOCH, rows, comments=comments)
    print(f"Written -> {args.output}")

    # Altitude profile sanity check (every 6 min)
    alts = [(t_s / 3600, float(np.linalg.norm(pos)) - R_EARTH)
            for t_s, pos, _ in rows[::36]]
    print()
    print("Altitude profile (every 6 min):")
    print(f"  {'Time [h]':>9}  {'Alt [km]':>10}")
    print("  " + "-" * 22)
    for t_h, alt in alts:
        print(f"  {t_h:>9.3f}  {alt:>10.0f}")

    # ASAT direct-ascent trajectory
    if args.asat_out:
        t_int = args.intercept_t
        # Target altitude at intercept
        t_arr = np.array([r[0] for r in rows])
        idx   = int(np.clip(np.searchsorted(t_arr, t_int), 1, len(rows) - 1))
        _, p_int, _ = rows[idx]
        alt_int = float(np.linalg.norm(p_int)) - R_EARTH

        print()
        print("ASAT direct-ascent parameters")
        print(f"  Launch site       : lat={args.asat_lat:.2f} deg  lon={args.asat_lon:.2f} deg  "
              f"alt={args.asat_start_alt:.1f} km")
        print(f"  Target sample time: {t_int:.0f} s  ({t_int/60:.1f} min) from Hohmann orbit")
        print(f"  Target altitude   : {alt_int:.0f} km")
        print(f"  Requested v_end   : {args.speed_end:.2f} km/s")
        print()

        print("Propagating ASAT trajectory ... ", end="", flush=True)
        asat_rows, v0_act, v_end_act, T_flight = generate_asat_trajectory(
            rows, args.asat_lat, args.asat_lon, args.asat_start_alt,
            t_int, EPOCH, DT_S, args.speed_end,
        )
        print(f"{len(asat_rows)} points")
        mean_decel = (v_end_act - v0_act) / T_flight  # km/s^2 (negative)
        print(f"  Launch speed      : {v0_act:.3f} km/s  (energy conservation)")
        print(f"  Intercept speed   : {v_end_act:.3f} km/s")
        print(f"  Flight time       : {T_flight:.1f} s  ({T_flight/60:.2f} min)")
        print(f"  Mean deceleration : {mean_decel*1000:.4f} m/s^2")

        asat_comments = [
            "ASAT direct-ascent trajectory -- Newtonian gravitational deceleration",
            f"Launch site: lat={args.asat_lat:.2f} deg  lon={args.asat_lon:.2f} deg  "
            f"alt={args.asat_start_alt:.1f} km",
            f"Aimed at Hohmann position sampled at t={t_int:.0f} s (alt={alt_int:.0f} km)",
            f"Launch speed: {v0_act:.3f} km/s  (derived from energy conservation)",
            f"Intercept speed: {v_end_act:.3f} km/s",
            f"Flight time: {T_flight:.1f} s ({T_flight/3600:.4f} h)",
            "Speed profile: v(r)^2 = v0^2 + 2*MU*(1/r - 1/r_launch)",
            "Path: great-circle SLERP in J2000 ECI",
        ]
        write_stk_ephem(args.asat_out, EPOCH, asat_rows, comments=asat_comments)
        print(f"Written -> {args.asat_out}")


if __name__ == "__main__":
    main()

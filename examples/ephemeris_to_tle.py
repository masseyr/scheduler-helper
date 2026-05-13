"""
ephemeris_to_tle.py -- Fit TLE(s) to an ECI state-vector ephemeris.

Input CSV columns (header row optional, auto-skipped):
    time_utc, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms

  time_utc  ISO-8601 UTC string ("2025-01-01T00:00:00Z")
            OR Unix timestamp in seconds (float)
  x/y/z     ECI J2000 position [km]
  vx/vy/vz  ECI J2000 velocity [km/s]

Algorithm
---------
1. Compute osculating Keplerian elements at arc midpoint -> initial TLE guess.
2. Propagate candidate TLE with SGP4 and minimise RMS position residual
   (Nelder-Mead, scipy).  Subsample to ~one point per --subsample seconds
   for speed; evaluate final statistics on the full arc.
3. If max residual > --threshold, split the arc at the midpoint and recurse.
   Stop splitting when the arc is shorter than --min-arc minutes.

Frame note
----------
SGP4 propagates in TEME; ECI J2000 and TEME differ by < 1 km for most
epochs -- acceptable at TLE accuracy.  For sub-km fidelity, transform
your ephemeris to TEME before fitting.

Dependencies
------------
    pip install sgp4 numpy scipy

Usage
-----
    python examples/ephemeris_to_tle.py data.csv
    python examples/ephemeris_to_tle.py data.csv --threshold 2.0 --name MY-SAT
    python examples/ephemeris_to_tle.py data.csv --out tles.txt --satnum 25544
    python examples/ephemeris_to_tle.py data.csv --min-arc 10 --subsample 30

Options
-------
--threshold  km   Max residual to accept a segment [default: 2.0]
--min-arc   min   Shortest arc that may be split further [default: 15]
--subsample  s    Optimizer sample interval in seconds  [default: 60]
--name       str  Satellite name line (default: EPHEMERIS)
--satnum     int  Catalog number for TLE lines (default: 99999)
--out        file Write TLEs to file instead of stdout
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, differential_evolution

from tasking_helper.utils.jdate import datetime_to_jd, jd_to_datetime

try:
    from sgp4.api import Satrec, WGS84
except ImportError as exc:
    raise SystemExit("sgp4 is required:  pip install sgp4") from exc

# -- Constants -----------------------------------------------------------------

MU          = 398_600.4418      # km^3 s^-2
TWO_PI      = 2.0 * math.pi
_SGP4_EPOCH = 2_433_281.5       # JD of 1949 Dec 31 00:00 UT

# -- TLE formatting -------------------------------------------------------------

def _tle_checksum(line: str) -> int:
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0)
               for c in line[:68]) % 10


def _tle_exp_float(value: float) -> str:
    """Format as +/-NNNNN+/-N  (8 chars) -- used for B* and nddot fields."""
    if abs(value) < 1e-30:
        return " 00000-0"
    sign = " " if value >= 0 else "-"
    av   = abs(value)
    exp  = math.floor(math.log10(av)) + 1   # mantissa in [0.1, 1.0)
    if abs(exp) > 9:                         # out of representable range
        return " 00000-0"
    m    = min(round(av * 10 ** (5 - exp)), 99_999)
    es   = "+" if exp >= 0 else "-"
    return f"{sign}{m:05d}{es}{abs(exp):1d}"


def _tle_ndot(ndot_rev_day2: float) -> str:
    """Format ndot as +/-.NNNNNNNN  (10 chars)."""
    sign = " " if ndot_rev_day2 >= 0 else "-"
    m    = min(round(abs(ndot_rev_day2) * 1e8), 99_999_999)
    return f"{sign}.{m:08d}"


def format_tle(name: str, elems: dict,
               satnum: int = 99999,
               cospar: str = "99999A  ",
               element_set: int = 1) -> str:
    """Return 3-line TLE string (name + line 1 + line 2)."""
    a_km     = elems["a_km"]
    ecc      = float(np.clip(elems["ecc"], 0.0, 0.9999999))
    incl_d   = math.degrees(elems["incl_rad"]) % 360.0
    raan_d   = math.degrees(elems["raan_rad"]) % 360.0
    argp_d   = math.degrees(elems["argp_rad"]) % 360.0
    M_d      = math.degrees(elems["M_rad"])    % 360.0
    bstar    = elems.get("bstar", 0.0)
    epoch_jd = elems["epoch_jd"]

    n_rev_day = math.sqrt(MU / a_km ** 3) * 86_400.0 / TWO_PI

    epoch_dt  = jd_to_datetime(epoch_jd)
    year2     = epoch_dt.year % 100
    doy       = epoch_dt.timetuple().tm_yday
    frac_day  = (epoch_dt.hour * 3600 + epoch_dt.minute * 60
                 + epoch_dt.second + epoch_dt.microsecond * 1e-6) / 86_400.0
    frac_i    = min(round(frac_day * 1e8), 99_999_999)
    epoch_str = f"{year2:02d}{doy:03d}.{frac_i:08d}"  # 14 chars

    cospar8   = f"{cospar:<8s}"[:8]
    ecc7      = f"{min(round(ecc * 1e7), 9_999_999):07d}"

    # Line 1 (68 chars before checksum)
    l1b = (f"1 {satnum:05d}U {cospar8} {epoch_str} "
           f"{_tle_ndot(0.0)} {_tle_exp_float(0.0)} {_tle_exp_float(bstar)} "
           f"0 {element_set:4d}")
    l1b = f"{l1b:<68s}"[:68]
    l1  = l1b + str(_tle_checksum(l1b))

    # Line 2 (68 chars before checksum)
    l2b = (f"2 {satnum:05d} {incl_d:8.4f} {raan_d:8.4f} {ecc7} "
           f"{argp_d:8.4f} {M_d:8.4f} {n_rev_day:11.8f}    0")
    l2b = f"{l2b:<68s}"[:68]
    l2  = l2b + str(_tle_checksum(l2b))

    return f"{name}\n{l1}\n{l2}"


# -- State vector <-> Keplerian elements ----------------------------------------

def state_to_keplerian(pos_km: np.ndarray, vel_kms: np.ndarray):
    """
    Osculating Keplerian elements from an ECI state vector.

    Returns (a_km, ecc, incl_rad, raan_rad, argp_rad, M_rad).
    """
    r_v  = pos_km
    v_v  = vel_kms
    r    = np.linalg.norm(r_v)
    v    = np.linalg.norm(v_v)

    h_v  = np.cross(r_v, v_v)
    h    = np.linalg.norm(h_v)
    n_v  = np.cross(np.array([0.0, 0.0, 1.0]), h_v)
    n    = np.linalg.norm(n_v)

    e_v  = np.cross(v_v, h_v) / MU - r_v / r
    ecc  = np.linalg.norm(e_v)

    a    = -MU / (2.0 * (v ** 2 / 2.0 - MU / r))
    incl = math.acos(np.clip(h_v[2] / h, -1.0, 1.0))

    if n < 1e-10:
        raan = 0.0
    else:
        raan = math.acos(np.clip(n_v[0] / n, -1.0, 1.0))
        if n_v[1] < 0:
            raan = TWO_PI - raan

    if n < 1e-10 or ecc < 1e-10:
        argp = 0.0
    else:
        argp = math.acos(np.clip(np.dot(n_v, e_v) / (n * ecc), -1.0, 1.0))
        if e_v[2] < 0:
            argp = TWO_PI - argp

    if ecc < 1e-10:
        nu = 0.0
    else:
        nu = math.acos(np.clip(np.dot(e_v, r_v) / (ecc * r), -1.0, 1.0))
        if np.dot(r_v, v_v) < 0:
            nu = TWO_PI - nu

    E = 2.0 * math.atan2(math.sqrt(1.0 - ecc) * math.sin(nu / 2.0),
                          math.sqrt(1.0 + ecc) * math.cos(nu / 2.0))
    M = (E - ecc * math.sin(E)) % TWO_PI
    return a, ecc, incl, raan, argp, M


# -- SGP4 helpers --------------------------------------------------------------

def _make_satrec(a_km: float, ecc: float, incl: float, raan: float,
                 argp: float, M: float, epoch_jd: float, bstar: float = 0.0) -> Satrec:
    n_rad_min = math.sqrt(MU / a_km ** 3) * 60.0
    sat = Satrec()
    sat.sgp4init(WGS84, "i", 99999, epoch_jd - _SGP4_EPOCH,
                 bstar, 0.0, 0.0, ecc, argp, incl, M, n_rad_min, raan)
    return sat


def _propagate(sat: Satrec, jd: np.ndarray) -> np.ndarray:
    """Return (N, 3) position array [km]; failed rows set to NaN."""
    try:
        jd_i = np.floor(jd)
        jd_f = jd - jd_i
        err, r, _ = sat.sgp4_array(jd_i, jd_f)
        pos = np.asarray(r, dtype=float)
        pos[err != 0] = np.nan
        return pos
    except AttributeError:
        N   = len(jd)
        pos = np.empty((N, 3))
        for i in range(N):
            e, r, _ = sat.sgp4(jd[i], 0.0)
            pos[i]  = r if e == 0 else [np.nan, np.nan, np.nan]
        return pos


# -- TLE fitting ---------------------------------------------------------------

# Typical magnitudes used to scale optimizer parameters to O(1)
_SCALES = np.array([7_000.0, 0.01, 1.0, 1.0, 1.0, 1.0, 1e-4])


def _objective(x_scaled: np.ndarray,
               epoch_jd: float,
               jd_obs:   np.ndarray,
               pos_obs:  np.ndarray) -> float:
    a, ecc, incl, raan, argp, M, bstar = x_scaled * _SCALES
    if not (6_371.0 < a < 500_000.0 and 0.0 <= ecc < 0.98):
        return 1e15
    try:
        sat = _make_satrec(a, ecc, incl, raan, argp, M, epoch_jd, bstar)
        pos = _propagate(sat, jd_obs)
        if np.any(np.isnan(pos)):
            return 1e15
        return float(np.mean(np.sum((pos - pos_obs) ** 2, axis=1)))
    except Exception:
        return 1e15


def fit_single_tle(jd_times:    np.ndarray,
                   pos_eci:     np.ndarray,
                   vel_eci:     np.ndarray,
                   subsample_s: int = 60) -> tuple[Satrec, float, float, dict]:
    """
    Fit a single TLE to the arc via Nelder-Mead + L-BFGS-B refinement.

    Strategy:
      1. Nelder-Mead starting from osculating elements at arc midpoint.
      2. If that fails badly (RMS > 500 km), retry with differential_evolution
         over a tight band around the initial guess.
      3. L-BFGS-B polish of the best result found.

    Returns (satrec, rms_km, max_km, elems_dict).
    """
    N = len(jd_times)
    arc_s    = max(1.0, (jd_times[-1] - jd_times[0]) * 86_400.0)
    cadence  = arc_s / max(1, N - 1)
    step     = max(1, round(subsample_s / cadence))
    idx      = np.arange(0, N, step)
    jd_s, ps = jd_times[idx], pos_eci[idx]

    mid      = N // 2
    epoch_jd = jd_times[mid]
    a0, e0, i0, r0, w0, M0 = state_to_keplerian(pos_eci[mid], vel_eci[mid])

    x0 = np.array([a0, max(e0, 1e-6), i0, r0, w0, M0, 1e-4]) / _SCALES
    obj = lambda x: _objective(x, epoch_jd, jd_s, ps)  # noqa: E731

    # Stage 1: Nelder-Mead
    r1 = minimize(obj, x0, method="Nelder-Mead",
                  options={"maxiter": 50_000, "xatol": 1e-9,
                           "fatol": 1e-5, "adaptive": True})
    best = r1

    # Stage 2: fallback to differential_evolution if stage 1 is poor
    if best.fun > 500.0 ** 2:
        tol = np.array([0.05, 0.005, 0.05, 0.05, 0.10, 0.10, 5.0])
        bounds = [(x0[j] - tol[j], x0[j] + tol[j]) for j in range(7)]
        r2 = differential_evolution(obj, bounds, maxiter=500, tol=1e-8,
                                    seed=0, mutation=(0.5, 1.0),
                                    recombination=0.7, polish=False)
        if r2.fun < best.fun:
            best = r2

    # Stage 3: L-BFGS-B polish
    tol_b = np.array([0.02, 0.002, 0.02, 0.02, 0.05, 0.05, 2.0])
    bounds_b = [(best.x[j] - tol_b[j], best.x[j] + tol_b[j]) for j in range(7)]
    r3 = minimize(obj, best.x, method="L-BFGS-B", bounds=bounds_b,
                  options={"maxiter": 1_000, "ftol": 1e-12})
    if r3.fun < best.fun:
        best = r3

    a, ecc, incl, raan, argp, M, bstar = best.x * _SCALES
    ecc  = float(np.clip(ecc, 0.0, 0.99))
    sat  = _make_satrec(a, ecc, incl, raan, argp, M, epoch_jd, bstar)

    pos_fit   = _propagate(sat, jd_times)
    residuals = np.linalg.norm(pos_fit - pos_eci, axis=1)
    rms       = float(np.sqrt(np.nanmean(residuals ** 2)))
    maxr      = float(np.nanmax(residuals))

    elems = dict(a_km=a, ecc=ecc, incl_rad=incl, raan_rad=raan,
                 argp_rad=argp, M_rad=M, bstar=bstar, epoch_jd=epoch_jd)
    return sat, rms, maxr, elems


def fit_tles_recursive(
    jd_times:     np.ndarray,
    pos_eci:      np.ndarray,
    vel_eci:      np.ndarray,
    threshold_km: float = 2.0,
    min_points:   int   = 900,
    subsample_s:  int   = 60,
    depth:        int   = 0,
) -> list[dict]:
    """
    Recursively fit TLEs, splitting the arc if max residual > threshold.

    Returns list of result dicts with keys:
        satrec, rms_km, max_km, elems, jd_start, jd_end, n_pts
    """
    sat, rms, maxr, elems = fit_single_tle(jd_times, pos_eci, vel_eci, subsample_s)
    entry = dict(satrec=sat, rms_km=rms, max_km=maxr, elems=elems,
                 jd_start=jd_times[0], jd_end=jd_times[-1], n_pts=len(jd_times))

    if maxr <= threshold_km or len(jd_times) < min_points * 2:
        return [entry]

    indent = "  " * depth
    t0 = jd_to_datetime(jd_times[0]).strftime("%H:%M:%S")
    t1 = jd_to_datetime(jd_times[-1]).strftime("%H:%M:%S")
    print(f"{indent}[depth {depth}] {t0}-{t1}  max {maxr:.1f} km "
          f"> {threshold_km:.1f} km -> splitting", file=sys.stderr)

    mid   = len(jd_times) // 2
    left  = fit_tles_recursive(jd_times[:mid], pos_eci[:mid], vel_eci[:mid],
                                threshold_km, min_points, subsample_s, depth + 1)
    right = fit_tles_recursive(jd_times[mid:], pos_eci[mid:], vel_eci[mid:],
                                threshold_km, min_points, subsample_s, depth + 1)
    return left + right


# -- CSV reader -----------------------------------------------------------------

def _parse_time(s: str) -> datetime:
    s = s.strip().strip('"')
    try:
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    except ValueError:
        pass
    s = s.replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time: {s!r}")


def read_ephemeris(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read CSV ephemeris file.

    Expects columns: time_utc, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms.
    A header row (any non-numeric first field) is skipped automatically.

    Returns (jd_times [N], pos_eci [N,3], vel_eci [N,3]).
    """
    rows: list[tuple] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",") if "," in line else line.split()
            if len(parts) < 7:
                continue
            try:
                float(parts[1])          # numeric check -- skip header
            except ValueError:
                continue
            try:
                dt  = _parse_time(parts[0])
                xyz = [float(v) for v in parts[1:4]]
                vel = [float(v) for v in parts[4:7]]
                rows.append((datetime_to_jd(dt), xyz, vel))
            except Exception as exc:
                print(f"  Warning: skipping line {lineno}: {exc}", file=sys.stderr)

    if not rows:
        raise ValueError(f"No valid data rows found in {path}")

    rows.sort(key=lambda r: r[0])
    jd  = np.array([r[0] for r in rows])
    pos = np.array([r[1] for r in rows])
    vel = np.array([r[2] for r in rows])
    return jd, pos, vel


# -- Main ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit TLE(s) to an ECI state-vector ephemeris CSV.")
    ap.add_argument("csv",
                    help="Input CSV: time_utc, x_km, y_km, z_km, vx_kms, vy_kms, vz_kms")
    ap.add_argument("--threshold", type=float, default=2.0, metavar="KM",
                    help="Max residual per segment [km] (default 2.0)")
    ap.add_argument("--min-arc",   type=float, default=15.0, metavar="MIN",
                    help="Minimum arc length before refusing to split [min] (default 15)")
    ap.add_argument("--subsample", type=int,   default=60,   metavar="S",
                    help="Optimizer sample interval [s] (default 60)")
    ap.add_argument("--name",      default="EPHEMERIS",
                    help="Satellite name line (default: EPHEMERIS)")
    ap.add_argument("--satnum",    type=int,   default=99999,
                    help="Catalog number in TLE lines (default: 99999)")
    ap.add_argument("--out",       default=None, metavar="FILE",
                    help="Write output to file (default: stdout)")
    args = ap.parse_args()

    # -- Load data --------------------------------------------------------------
    print(f"Reading {args.csv} ...", file=sys.stderr)
    jd_times, pos_eci, vel_eci = read_ephemeris(args.csv)
    N     = len(jd_times)
    arc_m = (jd_times[-1] - jd_times[0]) * 1440.0
    t0s   = jd_to_datetime(jd_times[0]).strftime("%Y-%m-%d %H:%M:%S")
    t1s   = jd_to_datetime(jd_times[-1]).strftime("%Y-%m-%d %H:%M:%S")
    print(f"Loaded {N} points over {arc_m:.1f} min  ({t0s} to {t1s} UTC)",
          file=sys.stderr)

    cadence_s = max(1.0, (jd_times[-1] - jd_times[0]) * 86_400.0 / max(1, N - 1))
    min_pts   = max(60, round(args.min_arc * 60.0 / cadence_s))

    # -- Fit --------------------------------------------------------------------
    print(f"Fitting: threshold={args.threshold} km, "
          f"min_arc={args.min_arc} min, subsample={args.subsample} s ...",
          file=sys.stderr)

    results = fit_tles_recursive(
        jd_times, pos_eci, vel_eci,
        threshold_km=args.threshold,
        min_points=min_pts,
        subsample_s=args.subsample,
    )

    # -- Build output -----------------------------------------------------------
    lines: list[str] = []
    lines.append(f"# Ephemeris-to-TLE fit: {Path(args.csv).name}")
    lines.append(f"# Source  : {N} points, {arc_m:.1f} min arc")
    lines.append(f"# Segments: {len(results)},  threshold: {args.threshold} km")
    lines.append("")

    for i, res in enumerate(results, 1):
        s0 = jd_to_datetime(res["jd_start"]).strftime("%Y-%m-%d %H:%M:%S")
        s1 = jd_to_datetime(res["jd_end"]).strftime("%Y-%m-%d %H:%M:%S")
        arc_i = (res["jd_end"] - res["jd_start"]) * 1440.0
        lines.append(f"# Segment {i}/{len(results)}: {s0} to {s1} UTC  ({arc_i:.1f} min)")
        lines.append(f"#   RMS {res['rms_km']:.3f} km   max {res['max_km']:.3f} km   "
                     f"{res['n_pts']} pts")
        lines.append(format_tle(args.name, res["elems"],
                                satnum=args.satnum, element_set=i))
        lines.append("")

    output = "\n".join(lines)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Wrote {len(results)} TLE(s) to {args.out}", file=sys.stderr)
    else:
        print(output)

    # -- Summary table ----------------------------------------------------------
    hdr = f"  {'#':>3}  {'Epoch (UTC)':>19}  {'Arc [min]':>9}  {'RMS [km]':>8}  {'Max [km]':>8}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(f"\nFit summary ({len(results)} segment(s)):", file=sys.stderr)
    print(hdr, file=sys.stderr)
    print(sep, file=sys.stderr)
    for i, res in enumerate(results, 1):
        epoch_s = jd_to_datetime(res["elems"]["epoch_jd"]).strftime("%Y-%m-%d %H:%M:%S")
        arc_i   = (res["jd_end"] - res["jd_start"]) * 1440.0
        print(f"  {i:>3}  {epoch_s:>19}  {arc_i:>9.1f}  "
              f"{res['rms_km']:>8.3f}  {res['max_km']:>8.3f}", file=sys.stderr)
    print(sep, file=sys.stderr)


if __name__ == "__main__":
    main()

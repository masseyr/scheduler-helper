#!/usr/bin/env python3
"""
mk_covariance — Generate position covariance files for a TLE catalog.

Uses user-supplied parameters file (CSV/JSON) and built-in
orbit-class defaults.

The COVGEN model represents position uncertainty in the TNW frame
(T = along-track, N = orbit-normal, W = cross-track) as a polynomial
in days elapsed since the data epoch:

    σ_d(Δt) = c_d  +  l_d · Δt  +  q_d · Δt²       [m, m/day, m/day²]

Three output frames are supported:
  --ECI     Full upper-triangle of the 3×3 ECI covariance (C11..C33)  [m²]
  --TNW     Diagonal of the 3×3 TNW covariance (C11, C22, C33)        [m²]
  --COVGEN  Raw polynomial parameters (cT cN cW lT lN lW qT qN qW)

Usage (CLI):
    python mk_covariance.py -e 2024-01-15T00:00:00 --ECI \\
        -c sats.tle -o cov.tsv

    python mk_covariance.py -e 21015.00000000 --TNW \\
        -c sats.tle --covgen-file params.csv -o cov.tsv

Usage (module):
    from mk_covariance import parse_epoch, propagate, CovgenParams, eci_covariance
"""

import importlib.util
import math
import json
import argparse
import sys
import types
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.jdate import epoch_to_jd, jd_to_datetime, fmt_epoch, parse_epoch

# ---------------------------------------------------------------------------
# Bootstrap tle.py's relative imports (same approach as make_satcat.py)
# ---------------------------------------------------------------------------

def _bootstrap_tle_module():
    here = pathlib.Path(__file__).parent
    pkg  = '_mkcov_tle_pkg'

    if pkg in sys.modules:
        return sys.modules[f'{pkg}.tle']

    pkg_mod = types.ModuleType(pkg)
    pkg_mod.__path__    = [str(here)]
    pkg_mod.__package__ = pkg
    sys.modules[pkg] = pkg_mod

    def _load(stem, path):
        name = f'{pkg}.{stem}'
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load('utils', here / 'utils.py')

    # Load orbits.py if present; otherwise stub the two propagation helpers.
    # mk_covariance uses its own propagator, so the stubs are fine here too.
    orbits_path = here / 'orbits.py'
    if orbits_path.exists():
        _load('orbits', orbits_path)
    else:
        orbits = types.ModuleType(f'{pkg}.orbits')
        orbits.__package__      = pkg
        orbits.keplerian_to_eci = None
        orbits._solve_kepler    = None
        sys.modules[f'{pkg}.orbits'] = orbits

    return _load('tle', here / 'tle.py')


_tle_mod = _bootstrap_tle_module()
_parse_tle_batch = _tle_mod.parse_tle_batch   # (text: str) -> list[TLE]

# ---------------------------------------------------------------------------
# Self-contained J2 + drag propagator
# (mirrors tle.propagate_tle without requiring orbits.py)
# ---------------------------------------------------------------------------

_MU = 3.986004418e14   # m³/s²
_RE = 6_378_137.0      # m
_J2 = 1.08263e-3
_TWO_PI = 2.0 * math.pi
_REV_DAY_TO_RAD_S = _TWO_PI / 86400.0


def _solve_kepler(M: float, e: float) -> float:
    """Newton–Raphson solution of Kepler's equation M = E − e sin E."""
    E = M + e * math.sin(M)
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    return E


def _keplerian_to_eci(a: float, e: float, i: float,
                      raan: float, argp: float,
                      nu: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Keplerian elements to ECI position (m) and velocity (m/s)."""
    p        = a * (1.0 - e * e)
    r_mag    = p / (1.0 + e * math.cos(nu))
    sqrt_mup = math.sqrt(_MU / p)

    r_pf = r_mag * np.array([math.cos(nu), math.sin(nu), 0.0])
    v_pf = sqrt_mup * np.array([-math.sin(nu), e + math.cos(nu), 0.0])

    cr, sr = math.cos(raan), math.sin(raan)
    ci, si = math.cos(i),    math.sin(i)
    ca, sa = math.cos(argp), math.sin(argp)

    # Perifocal → ECI rotation (313 Euler: Rz(-raan)·Rx(-i)·Rz(-argp))
    R = np.array([
        [ cr*ca - sr*sa*ci,  -cr*sa - sr*ca*ci,  sr*si],
        [ sr*ca + cr*sa*ci,  -sr*sa + cr*ca*ci, -cr*si],
        [ sa*si,               ca*si,             ci  ],
    ])
    return R @ r_pf, R @ v_pf


def propagate(tle, jd_target: float) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate *tle* to Julian Date *jd_target*.

    Applies J2 secular rates on RAAN and argp, and a simple drag correction
    via TLE ndot.  Returns (r_eci [m], v_eci [m/s]).
    """
    # tle.epoch_jd is computed by tle.py's _epoch_to_jd which is 1 day too high;
    # subtract 1.0 to recover the correct epoch before computing dt.
    dt  = (jd_target - (tle.epoch_jd - 1.0)) * 86400.0  # seconds
    n   = tle.mean_motion * _REV_DAY_TO_RAD_S
    e   = tle.eccentricity
    a   = tle.semi_major_axis
    inc = tle.inclination

    # J2 secular rates
    p          = a * (1.0 - e * e)
    j2_factor  = -1.5 * n * _J2 * (_RE / p) ** 2
    raan_dot   = j2_factor * math.cos(inc)
    argp_dot   = j2_factor * (2.0 - 2.5 * math.sin(inc) ** 2)

    # Drag via ndot
    n_dot  = tle.ndot * _REV_DAY_TO_RAD_S / 86400.0     # rad/s²
    n_t    = n + n_dot * dt
    a_t    = (_MU / n_t ** 2) ** (1.0 / 3.0)

    raan_t = tle.raan + raan_dot * dt
    argp_t = tle.argp + argp_dot * dt
    M_t    = (tle.mean_anomaly + n * dt + 0.5 * n_dot * dt ** 2) % _TWO_PI

    E  = _solve_kepler(M_t, e)
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(E / 2.0),
        math.sqrt(1.0 - e) * math.cos(E / 2.0),
    )
    return _keplerian_to_eci(a_t, e, inc, raan_t, argp_t, nu)


# ---------------------------------------------------------------------------
# COVGEN parameters and orbit-class defaults
# ---------------------------------------------------------------------------

@dataclass
class CovgenParams:
    """Polynomial covariance growth parameters in the TNW frame.

    Position uncertainty in direction d at age Δt days after data_epoch:
        σ_d(Δt) = c_d  +  l_d · Δt  +  q_d · Δt²    [m, m/day, m/day²]
    """
    cT: float = 200.0;  cN: float = 50.0;  cW: float = 100.0
    lT: float = 500.0;  lN: float = 50.0;  lW: float = 100.0
    qT: float = 100.0;  qN: float = 10.0;  qW: float = 20.0


# Built-in orbit-class defaults.  Units: m, m/day, m/day²
_DEFAULT_COVGEN: Dict[str, CovgenParams] = {
    'LEO': CovgenParams(cT=200,  cN=50,   cW=100,
                        lT=500,  lN=50,   lW=100,
                        qT=100,  qN=10,   qW=20),
    'MEO': CovgenParams(cT=500,  cN=100,  cW=200,
                        lT=2000, lN=200,  lW=500,
                        qT=400,  qN=40,   qW=100),
    'GEO': CovgenParams(cT=2000, cN=500,  cW=500,
                        lT=5000, lN=1000, lW=1000,
                        qT=1000, qN=200,  qW=200),
    'HEO': CovgenParams(cT=5000, cN=1000, cW=2000,
                        lT=10000,lN=2000, lW=4000,
                        qT=2000, qN=400,  qW=800),
}


def covgen_class(a_km: float, ecc: float) -> str:
    """Classify an orbit by semi-major axis (km) and eccentricity.

    Returns one of: 'LEO', 'MEO', 'GEO', 'HEO'.
    Thresholds match common SSA practice.
    """
    if ecc >= 0.25:
        return 'HEO'
    alt_km = a_km - 6378.137
    if alt_km < 2000.0:
        return 'LEO'
    if 34000.0 < alt_km < 37500.0:   # GEO belt ≈ 35786 km
        return 'GEO'
    return 'MEO'


def default_params(a_km: float, ecc: float) -> CovgenParams:
    """Return default CovgenParams for the orbit class of (a_km, ecc)."""
    return _DEFAULT_COVGEN[covgen_class(a_km, ecc)]


# ---------------------------------------------------------------------------
# Load COVGEN parameters from file
# ---------------------------------------------------------------------------

def load_covgen_file(path: str) -> Dict[str, CovgenParams]:
    """Load COVGEN parameters from a CSV or JSON file.

    CSV format (with header):
        key, cT, cN, cW, lT, lN, lW, qT, qN, qW
    where *key* is either a numeric catalog number or an orbit class name.

    JSON format:
        {"25544": {"cT": 200, "cN": 50, ...}, "LEO": {...}, ...}

    Returns a dict mapping key (str) → CovgenParams.
    Keys that are purely numeric are also stored as int for lookup convenience.
    """
    p = pathlib.Path(path)
    if p.suffix.lower() == '.json':
        with open(p) as fh:
            raw = json.load(fh)
        return {k: CovgenParams(**v) for k, v in raw.items()}

    # CSV
    params = {}
    fields_order = ['cT', 'cN', 'cW', 'lT', 'lN', 'lW', 'qT', 'qN', 'qW']
    with open(p) as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            cols = [c.strip() for c in line.split(',')]
            if cols[0].lower() in ('key', 'catnum', 'class', 'id'):
                continue   # skip header
            if len(cols) < 10:
                print(f'WARNING: {path}:{lineno}: expected 10 columns, got {len(cols)} — skipped',
                      file=sys.stderr)
                continue
            try:
                key = cols[0]
                vals = {f: float(cols[i + 1]) for i, f in enumerate(fields_order)}
                params[key] = CovgenParams(**vals)
            except ValueError as exc:
                print(f'WARNING: {path}:{lineno}: {exc} — skipped', file=sys.stderr)
    return params


def lookup_params(catnum: int, a_km: float, ecc: float,
                  param_map: Dict[str, CovgenParams]) -> CovgenParams:
    """Return CovgenParams for *catnum*, falling back to orbit-class defaults.

    Lookup order:
      1. Exact catalog number (int or str) in *param_map*
      2. Orbit class name ('LEO', 'MEO', 'GEO', 'HEO') in *param_map*
      3. Built-in orbit-class default
    """
    for key in (str(catnum), catnum):
        if key in param_map:
            return param_map[key]
    cls = covgen_class(a_km, ecc)
    if cls in param_map:
        return param_map[cls]
    return _DEFAULT_COVGEN[cls]


# ---------------------------------------------------------------------------
# Covariance computation
# ---------------------------------------------------------------------------

def _tnw_sigmas(params: CovgenParams,
                data_epoch_jd: float,
                sat_epoch_jd: float) -> Tuple[float, float, float]:
    """Evaluate σ_T, σ_N, σ_W (m) at *sat_epoch_jd* given *data_epoch_jd*."""
    dt = sat_epoch_jd - data_epoch_jd        # days
    sT = params.cT + params.lT * dt + params.qT * dt * dt
    sN = params.cN + params.lN * dt + params.qN * dt * dt
    sW = params.cW + params.lW * dt + params.qW * dt * dt
    return sT, sN, sW


def tnw_covariance(params: CovgenParams,
                   data_epoch_jd: float,
                   sat_epoch_jd: float) -> Tuple[float, float, float]:
    """Return (C_TT, C_NN, C_WW) — TNW diagonal covariance values (m²).

    Off-diagonal terms are zero by the COVGEN model assumption.
    """
    sT, sN, sW = _tnw_sigmas(params, data_epoch_jd, sat_epoch_jd)
    return sT * sT, sN * sN, sW * sW


def eci_covariance(params: CovgenParams,
                   data_epoch_jd: float,
                   sat_epoch_jd: float,
                   r_eci: np.ndarray,
                   v_eci: np.ndarray) -> np.ndarray:
    """Return 3×3 ECI position covariance matrix (m²).

    Rotates the diagonal TNW covariance to ECI using the instantaneous
    TNW frame axes derived from the satellite state (r_eci, v_eci).

    TNW convention:
      T̂ = v̂            (along-track, velocity direction)
      N̂ = (r×v)/|r×v|  (orbit normal)
      Ŵ = T̂ × N̂       (completes right-hand frame, ≈ radial for circular)
    """
    sT, sN, sW = _tnw_sigmas(params, data_epoch_jd, sat_epoch_jd)

    t_hat = v_eci / np.linalg.norm(v_eci)
    h     = np.cross(r_eci, v_eci)
    n_hat = h / np.linalg.norm(h)
    w_hat = np.cross(t_hat, n_hat)

    # R columns = TNW axes expressed in ECI
    R     = np.column_stack([t_hat, n_hat, w_hat])
    C_tnw = np.diag([sT * sT, sN * sN, sW * sW])

    return R @ C_tnw @ R.T


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_eci(records: List[Dict], output_path: str) -> None:
    """Write ECI covariance file (upper-triangle, tab-separated)."""
    header = '#CatalogNumber\tEpoch\tC11\tC12\tC13\tC22\tC23\tC33'
    with open(output_path, 'w') as fh:
        fh.write(header + '\n')
        for rec in records:
            C   = rec['C_eci']
            row = '\t'.join([
                str(rec['catnum']),
                fmt_epoch(rec['epoch_jd']),
                f'{C[0,0]:.6e}', f'{C[0,1]:.6e}', f'{C[0,2]:.6e}',
                f'{C[1,1]:.6e}', f'{C[1,2]:.6e}', f'{C[2,2]:.6e}',
            ])
            fh.write(row + '\n')


def write_tnw(records: List[Dict], output_path: str) -> None:
    """Write TNW diagonal covariance file (tab-separated)."""
    header = '#CatalogNumber\tEpoch\tC11\tC22\tC33'
    with open(output_path, 'w') as fh:
        fh.write(header + '\n')
        for rec in records:
            ctt, cnn, cww = rec['C_tnw']
            row = '\t'.join([
                str(rec['catnum']),
                fmt_epoch(rec['epoch_jd']),
                f'{ctt:.6e}', f'{cnn:.6e}', f'{cww:.6e}',
            ])
            fh.write(row + '\n')


def write_covgen(records: List[Dict], output_path: str) -> None:
    """Write raw COVGEN polynomial parameters (tab-separated)."""
    header = '#CatalogNumber\tcT\tcN\tcW\tlT\tlN\tlW\tqT\tqN\tqW'
    with open(output_path, 'w') as fh:
        fh.write(header + '\n')
        for rec in records:
            p   = rec['params']
            row = '\t'.join([
                str(rec['catnum']),
                f'{p.cT:.6g}', f'{p.cN:.6g}', f'{p.cW:.6g}',
                f'{p.lT:.6g}', f'{p.lN:.6g}', f'{p.lW:.6g}',
                f'{p.qT:.6g}', f'{p.qN:.6g}', f'{p.qW:.6g}',
            ])
            fh.write(row + '\n')


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='mk_covariance',
        description='Generate position covariance files for a TLE catalog '
                    'using the COVGEN polynomial model.')

    p.add_argument('-e', '--epoch', required=True, metavar='EPOCH',
                   help="Target epoch: 'YYYY-mm-ddTHH:MM:SS' or TLE format 'yyddd.ddddddd'")
    p.add_argument('-c', '--cfile', required=True, metavar='FILE',
                   help='TLE catalog file (2-line or 3-line format)')
    p.add_argument('-o', '--output', required=True, metavar='FILE',
                   help='Output filename')

    frame = p.add_mutually_exclusive_group(required=True)
    frame.add_argument('--ECI',    dest='frame', action='store_const', const='ECI',
                       help='Output ECI covariance (upper triangle C11..C33)')
    frame.add_argument('--TNW',    dest='frame', action='store_const', const='TNW',
                       help='Output TNW diagonal covariance (C11, C22, C33)')
    frame.add_argument('--COVGEN', dest='frame', action='store_const', const='COVGEN',
                       help='Output raw COVGEN polynomial parameters')

    p.add_argument('--covgen-file', metavar='FILE',
                   help='CSV or JSON file of COVGEN parameters '
                        '(per catalog number or orbit class)')
    p.add_argument('--data-epoch', metavar='EPOCH',
                   help='COVGEN data epoch (default: same as --epoch, giving σ = c_d)')
    p.add_argument('-I', '--id', type=int, default=0, metavar='N',
                   help='COVGEN run ID (informational only; not used in computation)')
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    # Parse target epoch
    try:
        epoch_jd = parse_epoch(args.epoch)
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    # Parse data epoch (defaults to target epoch → Δt = 0, σ = c_d)
    if args.data_epoch:
        try:
            data_epoch_jd = parse_epoch(args.data_epoch)
        except ValueError as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 1
    else:
        data_epoch_jd = epoch_jd

    # Load TLE catalog
    try:
        with open(args.cfile) as fh:
            tles = _parse_tle_batch(fh.read())
    except OSError as exc:
        print(f'ERROR: Cannot read catalog file: {exc}', file=sys.stderr)
        return 1

    if not tles:
        print('ERROR: No valid TLE records parsed.', file=sys.stderr)
        return 1

    # Load COVGEN parameters file (optional)
    param_map: Dict[str, CovgenParams] = {}
    if args.covgen_file:
        try:
            param_map = load_covgen_file(args.covgen_file)
        except OSError as exc:
            print(f'ERROR: Cannot read COVGEN file: {exc}', file=sys.stderr)
            return 1

    # Process each TLE
    records = []
    for tle in tles:
        try:
            r_eci, v_eci = propagate(tle, epoch_jd)
        except Exception as exc:
            print(f'WARNING: Propagation failed for {tle.norad_id} ({tle.name}): {exc}',
                  file=sys.stderr)
            continue

        a_km   = tle.semi_major_axis / 1000.0
        params = lookup_params(tle.norad_id, a_km, tle.eccentricity, param_map)

        rec = {
            'catnum':    tle.norad_id,
            'epoch_jd':  epoch_jd,
            'params':    params,
            'r_eci':     r_eci,
            'v_eci':     v_eci,
        }

        if args.frame in ('ECI',):
            rec['C_eci'] = eci_covariance(params, data_epoch_jd, epoch_jd,
                                          r_eci, v_eci)
        if args.frame in ('TNW', 'ECI'):
            rec['C_tnw'] = tnw_covariance(params, data_epoch_jd, epoch_jd)

        records.append(rec)

    if not records:
        print('ERROR: No records to write (all propagations failed?)', file=sys.stderr)
        return 1

    # Write output
    try:
        if args.frame == 'ECI':
            write_eci(records, args.output)
        elif args.frame == 'TNW':
            write_tnw(records, args.output)
        else:
            write_covgen(records, args.output)
    except OSError as exc:
        print(f'ERROR: Cannot write output file: {exc}', file=sys.stderr)
        return 1

    print(f'Wrote {len(records)} records [{args.frame}] to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

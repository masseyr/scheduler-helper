#!/usr/bin/env python3
"""
make_satcat — Generate LOST-format satellite catalog files from TLE data.

Uses direct TLE input; size and RCS
values are derived from a user-supplied diameter estimate using the NASA
Size Estimation Model (nasa_sem.py).

Two output formats are supported:
  v1  (mk_satcat_file):
        CatalogNumber, Name, IntlDes, Country, Launch, Decay,
        ObjType, SizeType, Size (m), CSA (m^2),
        UHF RCS, L Band RCS, S Band RCS, X Band RCS, C Band RCS

  v2  (mk_satcat_file2):
        CatalogNumber, Name, IntlDes, Country, Launch, Decay,
        a (km), e, i (deg), ObjType, SizeType, Size (m), CSA (m^2),
        UHF RCS, L Band RCS, S Band RCS, X Band RCS

Usage (module):
    from make_satcat import parse_tle_file, make_record, write_satcat

Usage (CLI):
    python make_satcat.py -t tle_file.txt -d 0.5 -o catalog.csv
    python make_satcat.py -t tle_file.txt -D sizes.csv -o catalog.csv --format v2
    python make_satcat.py --help
"""

import importlib.util
import math
import argparse
import sys
import types
import pathlib
from typing import Dict, List, Optional

from utils.nasa_sem import avgrcs, radar_band_center

# ---------------------------------------------------------------------------
# Bootstrap tle.py's relative imports
# ---------------------------------------------------------------------------
# tle.py lives in the same directory but was written as part of a package
# (uses `from .utils import …` and `from .orbits import …`).  We load it
# here as a virtual sub-module so that its relative imports resolve without
# requiring the helper folder to be an installed package.
#
# Only parse_tle / parse_tle_batch are used; the propagation helpers
# (keplerian_to_eci, _solve_kepler) are stubbed out.

def _bootstrap_tle_module():
    """Load tle.py resolving its relative imports against the local folder."""
    here = pathlib.Path(__file__).parent
    pkg  = '_make_satcat_pkg'

    if pkg in sys.modules:
        return sys.modules[f'{pkg}.tle']

    # Register a fake package rooted at this directory
    pkg_mod = types.ModuleType(pkg)
    pkg_mod.__path__    = [str(here)]
    pkg_mod.__package__ = pkg
    sys.modules[pkg] = pkg_mod

    # Load utils.py as <pkg>.utils  (no relative imports of its own)
    _load_submodule(pkg, here / 'utils.py')

    # Stub orbits — keplerian_to_eci / _solve_kepler are only used in
    # propagation functions we don't call from here.
    orbits = types.ModuleType(f'{pkg}.orbits')
    orbits.__package__      = pkg
    orbits.keplerian_to_eci = None
    orbits._solve_kepler    = None
    sys.modules[f'{pkg}.orbits'] = orbits

    return _load_submodule(pkg, here / 'tle.py')


def _load_submodule(pkg: str, path: pathlib.Path):
    name = f'{pkg}.{path.stem}'
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_tle_mod = _bootstrap_tle_module()
_parse_tle_batch = _tle_mod.parse_tle_batch   # (text: str) -> list[TLE]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RE_KM = 6378.137   # Earth equatorial radius (km) — matches utils.R_EARTH/1000

# Output band order (both formats share UHF/L/S/X; v1 appends C)
_BANDS_V1 = ['UHF', 'L', 'S', 'X', 'C']
_BANDS_V2 = ['UHF', 'L', 'S', 'X']

# Size type strings (mirror the C++ SizeTypeEnum names)
SIZE_DATABASE = 'DATABASE'
SIZE_RCSVAL   = 'RCSVAL'
SIZE_NASA_SEM = 'NASA_SEM'
SIZE_DEFAULT  = 'DEFAULT'
SIZE_NONE     = 'NO_SIZE_DATA'


# ---------------------------------------------------------------------------
# TLE → record dict
# ---------------------------------------------------------------------------

def _launch_date_from_intldes(intldes: str) -> str:
    """Return 'YYYY-01-01' derived from the two-digit year prefix of IntlDes."""
    if len(intldes) < 2:
        return 'UNKNOWN'
    try:
        yy = int(intldes[:2])
    except ValueError:
        return 'UNKNOWN'
    year = 1900 + yy if yy >= 57 else 2000 + yy
    return f'{year:04d}-01-01'


def _infer_obj_type(intldes: str, name: str) -> str:
    """Heuristically infer object type from IntlDes piece code and name."""
    name_up = name.upper()
    piece = intldes[5:].strip().lstrip('0') if len(intldes) > 5 else ''

    if 'R/B' in name_up or 'ROCKET' in name_up:
        return 'ROCKETBODY'
    if piece.startswith('B') and 'DEB' not in name_up:
        return 'ROCKETBODY'
    if 'DEB' in name_up or 'DEBRIS' in name_up or 'FRAG' in name_up:
        return 'DEBRIS'
    if piece == 'A' or not piece:
        return 'PAYLOAD'
    return 'DEBRIS'


def _tle_to_rec(tle) -> Dict:
    """Convert a tle.TLE dataclass to a make_satcat record dict.

    semi_major_axis is stored in metres by tle.py (derived from MU_EARTH in
    m³/s²); we convert to km here so the rest of the module is consistent
    with the original C++ conventions.
    """
    intldes = tle.intl_designator or ''
    name    = tle.name or intldes

    a_km   = tle.semi_major_axis / 1000.0          # m → km
    ecc    = tle.eccentricity
    i_deg  = math.degrees(tle.inclination)          # rad → deg

    return {
        'catnum':     tle.norad_id,
        'name':       name,
        'intldes':    intldes,
        'country':    '',                           # not available from TLE
        'launch':     _launch_date_from_intldes(intldes),
        'decay':      None,
        'obj_type':   _infer_obj_type(intldes, name),
        'a_km':       a_km,
        'ecc':        ecc,
        'incl_deg':   i_deg,
        'perigee_km': a_km * (1.0 - ecc) - _RE_KM,
        'apogee_km':  a_km * (1.0 + ecc) - _RE_KM,
    }


def parse_tle_text(text: str) -> List[Dict]:
    """Parse TLE text (2-line or 3-line format) into a list of record dicts.

    Wraps tle.parse_tle_batch; warnings for malformed entries go to stderr.
    """
    tles = _parse_tle_batch(text)
    return [_tle_to_rec(t) for t in tles]


def parse_tle_file(path: str) -> List[Dict]:
    """Read a TLE file and return a list of record dicts."""
    with open(path) as fh:
        return parse_tle_text(fh.read())


# ---------------------------------------------------------------------------
# Size and RCS computation
# ---------------------------------------------------------------------------

def compute_size(diameter_m: float) -> Dict:
    """Return a size record for a sphere with the given diameter (metres)."""
    r = diameter_m / 2.0
    return {
        'size_type': SIZE_NASA_SEM,
        'radius_m':  r,
        'csa_m2':    math.pi * r * r,
    }


def compute_rcs(radius_m: float, bands: List[str]) -> Dict[str, float]:
    """Compute NASA SEM RCS (m²) at each named radar band for *radius_m*."""
    return {b: avgrcs(radius_m, radar_band_center(b)) for b in bands}


# ---------------------------------------------------------------------------
# Catalog record assembly
# ---------------------------------------------------------------------------

def make_record(tle_rec: Dict,
                diameter_m: Optional[float],
                default_radius_m: float = -1.0,
                synthetic_rcs: bool = True,
                bands: Optional[List[str]] = None) -> Dict:
    """Assemble a catalog record from a parsed TLE dict and a size estimate.

    Args:
        tle_rec:          Output of parse_tle_file() / parse_tle_text() for
                          one object.
        diameter_m:       Physical diameter in metres, or None if unknown.
        default_radius_m: Fallback radius when *diameter_m* is None.
        synthetic_rcs:    Compute RCS from size via NASA SEM when True.
        bands:            Radar bands to include; defaults to v1 (UHF/L/S/X/C).

    Returns:
        Dict suitable for write_satcat().
    """
    if bands is None:
        bands = _BANDS_V1

    if diameter_m is not None and diameter_m > 0.0:
        size = compute_size(diameter_m)
    elif default_radius_m > 0.0:
        size = {
            'size_type': SIZE_DEFAULT,
            'radius_m':  default_radius_m,
            'csa_m2':    math.pi * default_radius_m ** 2,
        }
    else:
        size = {'size_type': SIZE_NONE, 'radius_m': None, 'csa_m2': None}

    if synthetic_rcs and size['radius_m'] is not None:
        rcs = compute_rcs(size['radius_m'], bands)
    else:
        rcs = {b: None for b in bands}

    return {**tle_rec, 'size': size, 'rcs': rcs}


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _fmt(value, spec=None) -> str:
    """Format *value* for CSV; returns 'NULL' for None."""
    if value is None:
        return 'NULL'
    return format(value, spec) if spec else str(value)


def write_satcat(records: List[Dict], output_path: str, fmt: str = 'v1') -> None:
    """Write catalog records to *output_path*.

    Args:
        records:     List of dicts from make_record().
        output_path: Destination file path.
        fmt:         'v1' — mk_satcat_file layout (with C-band, no orbital
                     elements); 'v2' — mk_satcat_file2 layout (a/e/i, no C).
    """
    if fmt not in ('v1', 'v2'):
        raise ValueError(f"fmt must be 'v1' or 'v2', got {fmt!r}")

    bands = _BANDS_V1 if fmt == 'v1' else _BANDS_V2

    if fmt == 'v1':
        header = ('#CatalogNumber, Name, IntlDes, Country, Launch, Decay, '
                  'ObjType, SizeType, Size (m), CSA (m^2), '
                  'UHF RCS, L Band RCS, S Band RCS, X Band RCS, C Band RCS')
    else:
        header = ('#CatalogNumber, Name, IntlDes, Country, Launch, Decay, '
                  'a, e, i, ObjType, SizeType, Size (m), CSA (m^2), '
                  'UHF RCS, L Band RCS, S Band RCS, X Band RCS')

    with open(output_path, 'w', newline='') as fh:
        fh.write(header + '\n')
        for rec in records:
            _write_record(fh, rec, fmt, bands)


def _write_record(fh, rec: Dict, fmt: str, bands: List[str]) -> None:
    size = rec.get('size', {})
    rcs  = rec.get('rcs', {})

    parts = [
        str(rec['catnum']),
        rec['name'],
        rec['intldes'],
        rec['country'],
        rec['launch'],
        rec['decay'] if rec['decay'] else 'NULL',
    ]

    if fmt == 'v2':
        parts += [
            f"{rec['a_km']:.6f}",
            f"{rec['ecc']:.7f}",
            f"{rec['incl_deg']:.4f}",
        ]

    parts.append(rec['obj_type'])
    parts += [
        size.get('size_type', SIZE_NONE),
        _fmt(size.get('radius_m'), '.6g'),
        _fmt(size.get('csa_m2'),   '.6g'),
    ]
    parts += [_fmt(rcs.get(b), '.6g') for b in bands]

    fh.write(', '.join(parts) + '\n')


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _load_diameter_map(path: str) -> Dict[int, float]:
    """Load a two-column CSV mapping catalog numbers → diameters (metres)."""
    dmap = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            cols = [c.strip() for c in line.split(',')]
            if len(cols) < 2:
                continue
            try:
                dmap[int(cols[0])] = float(cols[1])
            except ValueError:
                pass
    return dmap


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='make_satcat',
        description='Create LOST satellite catalog files from TLE data using '
                    'the NASA Size Estimation Model.')

    p.add_argument('-t', '--tle', required=True, metavar='FILE',
                   help='TLE file (2-line or 3-line format)')
    p.add_argument('-o', '--output', required=True, metavar='FILE',
                   help='Output CSV filename')

    size_grp = p.add_mutually_exclusive_group()
    size_grp.add_argument('-d', '--diameter', type=float, metavar='METRES',
                          help='Diameter (m) applied to all objects')
    size_grp.add_argument('-D', '--diameter-file', metavar='CSV',
                          help='CSV file: catnum, diameter_m  (per-object sizes)')

    p.add_argument('-r', '--default-radius', type=float, default=-1.0,
                   metavar='METRES',
                   help='Fallback radius (m) when no size data is available')
    p.add_argument('--synthetic-rcs', '--synthetic_rcs', action='store_true',
                   default=True,
                   help='Use NASA SEM to fill missing RCS values (default: on)')
    p.add_argument('--no-synthetic-rcs', dest='synthetic_rcs',
                   action='store_false',
                   help='Suppress NASA SEM synthetic RCS computation')
    p.add_argument('--format', choices=['v1', 'v2'], default='v1',
                   help="'v1' = mk_satcat_file (C-band, no orbital elements); "
                        "'v2' = mk_satcat_file2 (a/e/i, no C-band)  [default: v1]")
    p.add_argument('-c', '--cfile', metavar='FILE',
                   help='File of catalog numbers (or TLE pairs) to filter input')
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    try:
        records_raw = parse_tle_file(args.tle)
    except OSError as exc:
        print(f'ERROR: Cannot read TLE file: {exc}', file=sys.stderr)
        return 1

    if not records_raw:
        print('ERROR: No valid TLE records parsed.', file=sys.stderr)
        return 1

    # Optional catalog-number filter
    if args.cfile:
        try:
            with open(args.cfile) as fh:
                text = fh.read()
            # Accept TLE pairs or plain catalog numbers
            clines = [ln.strip() for ln in text.splitlines()
                      if ln.strip() and not ln.startswith('#')]
            if clines and clines[0].startswith('1 '):
                allowed = {r['catnum'] for r in parse_tle_text(text)}
            else:
                allowed = set()
                for ln in clines:
                    try:
                        allowed.add(int(ln))
                    except ValueError:
                        print(f'WARNING: Cannot parse catalog number: {ln!r}',
                              file=sys.stderr)
        except OSError as exc:
            print(f'ERROR: Cannot read catalog file: {exc}', file=sys.stderr)
            return 1
        records_raw = [r for r in records_raw if r['catnum'] in allowed]

    # Diameter map
    diam_map = {}
    if args.diameter_file:
        try:
            diam_map = _load_diameter_map(args.diameter_file)
        except OSError as exc:
            print(f'ERROR: Cannot read diameter file: {exc}', file=sys.stderr)
            return 1

    bands = _BANDS_V1 if args.format == 'v1' else _BANDS_V2
    catalog = [
        make_record(
            raw,
            diameter_m=diam_map.get(raw['catnum'], args.diameter),
            default_radius_m=args.default_radius,
            synthetic_rcs=args.synthetic_rcs,
            bands=bands,
        )
        for raw in records_raw
    ]

    try:
        write_satcat(catalog, args.output, fmt=args.format)
    except OSError as exc:
        print(f'ERROR: Cannot write output file: {exc}', file=sys.stderr)
        return 1

    print(f'Wrote {len(catalog)} records to {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

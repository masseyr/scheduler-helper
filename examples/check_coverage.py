"""
check_coverage.py -- Check ground-sensor visibility for a satellite ephemeris.

Usage
-----
python examples/check_coverage.py \
    --ephemeris  path/to/ephemeris.csv \
    --sensors    path/to/ground_sensors.csv \
    [--object    SAT-01] \
    [--out       path/to/output.csv]

Ephemeris CSV  (ECI J2000, km):
    time_utc,x_km,y_km,z_km[,vx_kms,vy_kms,vz_kms]

Ground sensors CSV:
    name,lat_deg,lon_deg,alt_km,max_range_km[,min_elevation_deg]

    alt_km and min_elevation_deg may be omitted / left blank (defaults: 0.0 and 5.0).

Example sensors CSV:
    name,lat_deg,lon_deg,alt_km,max_range_km,min_elevation_deg
    SiteAlpha,28.5,-80.6,0.010,2000,5
    SiteBravo,51.5,-0.12,0.046,1500,10
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow running directly from the examples/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tasking_helper.utils.coverage import (
    GroundSensor,
    AccessInterval,
    compute_access_table,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ephemeris(path: str):
    """
    Load ephemeris CSV -> (times_utc, pos_eci_km).

    Required columns : time_utc, x_km, y_km, z_km
    Optional columns : vx_kms, vy_kms, vz_kms  (ignored here)

    time_utc may be ISO-8601 with or without trailing 'Z'.
    """
    times = []
    pos   = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row["time_utc"].strip().rstrip("Z")
            t = datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
            times.append(t)
            pos.append([float(row["x_km"]), float(row["y_km"]), float(row["z_km"])])
    if not times:
        raise ValueError(f"Ephemeris file is empty: {path}")
    return times, np.array(pos)


def load_sensors(path: str) -> list:
    """
    Load ground sensors from CSV.

    Required columns : name, lat_deg, lon_deg, max_range_km
    Optional columns : alt_km (default 0.0), min_elevation_deg (default 5.0)
    """
    sensors = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            alt_km  = float(row.get("alt_km",            0.0) or 0.0)
            min_el  = float(row.get("min_elevation_deg", 5.0) or 5.0)
            sensors.append(GroundSensor(
                name              = row["name"].strip(),
                lat_deg           = float(row["lat_deg"]),
                lon_deg           = float(row["lon_deg"]),
                alt_km            = alt_km,
                max_range_km      = float(row["max_range_km"]),
                min_elevation_deg = min_el,
            ))
    if not sensors:
        raise ValueError(f"Sensors file is empty: {path}")
    return sensors


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_SEP = "-" * 95

def print_summary(intervals: list, sensors: list) -> None:
    print()
    print("=== Ground Sensor Coverage Report ===")
    print()
    for s in sensors:
        print(s.summary())
    print()

    if not intervals:
        print("  No access intervals found.")
        return

    hdr = (
        f"{'Sensor':<20} {'Object':<12} "
        f"{'Start (UTC)':<22} {'End (UTC)':<22} "
        f"{'Dur[s]':>7} {'MinRng[km]':>11} {'MaxEl[deg]':>11}"
    )
    print(hdr)
    print(_SEP)
    for iv in intervals:
        print(
            f"{iv.sensor_name:<20} {iv.object_id:<12} "
            f"{iv.start_time.strftime('%Y-%m-%dT%H:%M:%SZ'):<22} "
            f"{iv.end_time.strftime('%Y-%m-%dT%H:%M:%SZ'):<22} "
            f"{iv.duration_s:>7.0f} "
            f"{iv.min_range_km:>11.1f} "
            f"{iv.max_elevation_deg:>11.1f}"
        )
    print()
    total_s = sum(iv.duration_s for iv in intervals)
    print(f"  {len(intervals)} access interval(s)  |  "
          f"total coverage = {total_s:.0f} s  ({total_s/60:.1f} min)")


def save_csv(intervals: list, path: str) -> None:
    if not intervals:
        print(f"No intervals to write -- {path} not created.")
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(intervals[0].to_dict().keys()))
        writer.writeheader()
        writer.writerows(iv.to_dict() for iv in intervals)
    print(f"Saved {len(intervals)} row(s) -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check ground-sensor visibility against a satellite ephemeris.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ephemeris", required=True,
                        help="Path to ephemeris CSV (ECI J2000, km)")
    parser.add_argument("--sensors",   required=True,
                        help="Path to ground sensors CSV")
    parser.add_argument("--object",    default="object",
                        help="Object label used in output (default: 'object')")
    parser.add_argument("--out",       default=None,
                        help="Output CSV path for access intervals (optional)")
    args = parser.parse_args()

    # -- load ---------------------------------------------------------------
    print(f"Loading ephemeris : {args.ephemeris}")
    times, pos_eci = load_ephemeris(args.ephemeris)
    dt_s = (times[1] - times[0]).total_seconds() if len(times) > 1 else 0.0
    print(f"  {len(times)} epochs  dt={dt_s:.1f}s  "
          f"[{times[0].strftime('%Y-%m-%dT%H:%M:%SZ')} -- "
          f"{times[-1].strftime('%Y-%m-%dT%H:%M:%SZ')}]")

    print(f"Loading sensors   : {args.sensors}")
    sensors = load_sensors(args.sensors)
    print(f"  {len(sensors)} sensor(s): {', '.join(s.name for s in sensors)}")

    # -- compute ------------------------------------------------------------
    print("Computing access intervals ...")
    intervals = compute_access_table(times, pos_eci, sensors, object_id=args.object)

    # -- report -------------------------------------------------------------
    print_summary(intervals, sensors)

    if args.out:
        save_csv(intervals, args.out)


if __name__ == "__main__":
    main()

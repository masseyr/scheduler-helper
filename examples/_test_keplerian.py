"""Quick validation of keplerian.py -- run with python examples/_test_keplerian.py."""
import math
import numpy as np
from datetime import datetime, timezone

from tasking_helper.utils.keplerian import (
    lla_to_ecr, ecr_to_lla, lla_to_eci, eci_to_lla,
    eci_to_ecr, ecr_to_eci,
    state_to_keplerian, keplerian_to_state, solve_kepler,
    keplerian_to_lla, lla_to_keplerian,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(label, err_m, tol_m=1.0):
    ok = err_m < tol_m
    print(f"  {'[OK]' if ok else '[FAIL]'} {label:55s}  err={err_m:.4f} m  (tol={tol_m} m)")
    return ok


# -- Kepler's equation ---------------------------------------------------------
print("=== Kepler's equation  M = E - e*sin(E) ===")
cases = [(0, 0.0), (45, 0.01), (180, 0.3), (270, 0.7), (90, 0.95), (1, 0.999)]
for M, e in cases:
    E     = solve_kepler(M, e)
    M_rec = math.degrees(math.radians(E) - e * math.sin(math.radians(E))) % 360.0
    residual = abs(M_rec - M % 360.0) * 1e9   # converted to nano-degrees for display
    ok = residual < 1e-3
    print(f"  {'[OK]' if ok else '[FAIL]'} M={M:6.1f}  e={e:.3f}  E={E:10.5f}  "
          f"residual={abs(M_rec-M%360):.2e} deg")

# -- LLA <-> ECR --------------------------------------------------------------
print()
print("=== LLA <-> ECR round-trip (WGS-84) ===")
lla_cases = [
    ("Null Island",       0.0,   0.0,   0.0),
    ("London",           51.5,  -0.1,   0.1),
    ("Sydney",          -33.9, 151.2,   0.5),
    ("North pole",       89.9,  45.0,   0.0),
    ("GEO subsatellite", 0.0,   0.0, 35786.0),
    ("500 km LEO",       28.5,  80.6,  500.0),
]
for name, lat, lon, alt in lla_cases:
    ecr          = lla_to_ecr(lat, lon, alt)
    lat2,lon2,a2 = ecr_to_lla(ecr)
    d_lat  = abs(lat2 - lat) * 111.0 * 1e3   # rough m
    d_lon  = abs(lon2 - lon) * 111.0 * 1e3
    d_alt  = abs(a2  - alt ) * 1e3
    err    = math.sqrt(d_lat**2 + d_lon**2 + d_alt**2)
    check(name, err, tol_m=0.001)

# -- ECI <-> ECR --------------------------------------------------------------
print()
print("=== ECI <-> ECR round-trip ===")
t = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
for r_eci in [np.array([6500.0, 1000.0, 2500.0]),
              np.array([-3000.0, 5000.0, -4000.0]),
              np.array([42164.0, 0.0, 0.0])]:
    r_ecr  = eci_to_ecr(r_eci, t)
    r_back = ecr_to_eci(r_ecr, t)
    err    = np.linalg.norm(r_back - r_eci) * 1e3
    check(f"r_eci={r_eci}", err, tol_m=0.001)

# -- state_to_keplerian <-> keplerian_to_state ---------------------------------
print()
print("=== Orbital elements round-trip: keplerian_to_state -> state_to_keplerian ===")

orbit_cases = [
    ("Circular LEO (ISS-like)",
     dict(a_km=6778.137, ecc=0.001,  incl_deg=51.6,  raan_deg=23.4,  argp_deg=45.0,  M_deg=315.0)),
    ("Molniya (e=0.74, i=63.4)",
     dict(a_km=26560.0,  ecc=0.74,   incl_deg=63.4,  raan_deg=0.0,   argp_deg=270.0, M_deg=45.0)),
    ("Sun-synchronous SSO",
     dict(a_km=7178.137, ecc=0.001,  incl_deg=98.2,  raan_deg=90.0,  argp_deg=0.0,   M_deg=90.0)),
    ("GEO (near-equatorial)",
     dict(a_km=42164.0,  ecc=0.0001, incl_deg=0.1,   raan_deg=0.0,   argp_deg=0.0,   M_deg=120.0)),
    ("Retrograde (i=150)",
     dict(a_km=7000.0,   ecc=0.05,   incl_deg=150.0, raan_deg=180.0, argp_deg=60.0,  M_deg=200.0)),
    ("Highly eccentric (e=0.9)",
     dict(a_km=15000.0,  ecc=0.9,    incl_deg=28.5,  raan_deg=45.0,  argp_deg=90.0,  M_deg=30.0)),
    ("High-ecc near perigee",
     dict(a_km=15000.0,  ecc=0.9,    incl_deg=28.5,  raan_deg=45.0,  argp_deg=90.0,  M_deg=355.0)),
]

all_ok = True
for name, elems0 in orbit_cases:
    pos,  vel  = keplerian_to_state(**elems0)
    elems1     = state_to_keplerian(pos, vel)
    pos2, vel2 = keplerian_to_state(**elems1)
    p_err = np.linalg.norm(pos2 - pos) * 1e3
    v_err = np.linalg.norm(vel2 - vel) * 1e6
    ok    = p_err < 1.0 and v_err < 1.0
    print(f"  {'[OK]' if ok else '[FAIL]'} {name:40s}  "
          f"pos={p_err:.4f} m  vel={v_err:.4f} um/s")
    if not ok:
        all_ok = False

# -- keplerian_to_lla pipeline -------------------------------------------------
print()
print("=== keplerian_to_lla -> lla_to_keplerian pipeline ===")
t = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
for name, elems0 in orbit_cases:
    pos, vel = keplerian_to_state(**elems0)
    lat, lon, alt = keplerian_to_lla(**elems0, t=t)
    elems_rec = lla_to_keplerian(lat, lon, alt, vel, t)
    pos2, vel2 = keplerian_to_state(**elems_rec)
    p_err = np.linalg.norm(pos2 - pos) * 1e3
    ok    = p_err < 1.0
    print(f"  {'[OK]' if ok else '[FAIL]'} {name:40s}  pos={p_err:.4f} m  "
          f"lat={lat:.2f} lon={lon:.2f} alt={alt:.1f} km")
    if not ok:
        all_ok = False

print()
print("All checks complete.")

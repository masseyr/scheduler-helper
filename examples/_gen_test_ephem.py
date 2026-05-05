"""
Generate a synthetic ephemeris for testing ephemeris_to_tle.py.

Uses the ISS TLE as the source, propagates with SGP4, and writes 5 hours
of state vectors at 60-second cadence.  This is a proper round-trip test:
the TLE fitter should recover TLE elements close to the original.
"""
from datetime import datetime, timedelta, timezone
from sgp4.api import Satrec

# ISS TLE (2025 approximate)
LINE1 = "1 25544U 98067A   25152.50000000  .00005000  00000-0  90000-4 0  9990"
LINE2 = "2 25544  51.6400  23.4000 0004000  45.0000 315.0000 15.50000000000000"

sat = Satrec.twoline2rv(LINE1, LINE2)
T0  = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

import math
JD0 = (367 * T0.year
       - int(7 * (T0.year + int((T0.month + 9) / 12)) / 4)
       + int(275 * T0.month / 9)
       + T0.day + 1721013.5
       + (T0.hour + T0.minute / 60 + T0.second / 3600) / 24)

rows = []
for k in range(301):          # 5 hours at 60-s cadence = 301 points
    jd  = JD0 + k * 60.0 / 86400.0
    jd_i = math.floor(jd)
    jd_f = jd - jd_i
    err, r, v = sat.sgp4(jd_i, jd_f)
    if err != 0:
        print(f"SGP4 error at step {k}: code {err}")
        continue
    t  = T0 + timedelta(seconds=60 * k)
    ts = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows.append(f"{ts},{r[0]:.6f},{r[1]:.6f},{r[2]:.6f},{v[0]:.9f},{v[1]:.9f},{v[2]:.9f}")

out = "examples/_test_ephem.csv"
with open(out, "w") as f:
    f.write("time_utc,x_km,y_km,z_km,vx_kms,vy_kms,vz_kms\n")
    f.write("\n".join(rows) + "\n")
print(f"Wrote {len(rows)} rows to {out}")
print(f"Source TLE:")
print(f"  {LINE1}")
print(f"  {LINE2}")

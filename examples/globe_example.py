"""
Globe 3D example — run with:
    python examples/globe_example.py
Opens an interactive HTML globe in your browser showing:
  - WGS-84 Earth sphere with lat/lon graticule
  - A 15-degree global target grid
  - Several named ground targets (cities)
  - ISS ground track over one orbit
  - Sentinel-2A ground track over one orbit
  - Current satellite positions at epoch
"""
import sys
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tasking_helper.utils.tle import parse_tle
from tasking_helper.utils.satellite import Satellite
from tasking_helper.utils.jdate import epoch_to_jd
from tasking_helper.viz import Globe3D, Target

# ── TLEs ─────────────────────────────────────────────────────────────────────
_ISS = parse_tle(
    "ISS (ZARYA)",
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
_S2A = parse_tle(
    "SENTINEL-2A",
    "1 40697U 15028A   24082.50000000  .00000654  00000-0  34512-4 0  9991",
    "2 40697  98.5680 115.4420 0001030  90.5600 269.5700 14.30818200470251",
)

iss  = Satellite.from_tle(_ISS)
s2a  = Satellite.from_tle(_S2A)

# Use jdate's epoch_to_jd for the correct absolute JD (tle.epoch_jd is 1 day high)
iss_epoch = epoch_to_jd(_ISS.epoch_year, _ISS.epoch_day)
s2a_epoch = epoch_to_jd(_S2A.epoch_year, _S2A.epoch_day)

# ── Named ground targets ──────────────────────────────────────────────────────
cities = [
    Target(lat=51.5,   lon=-0.1,   name="London",        priority=2),
    Target(lat=40.7,   lon=-74.0,  name="New York",       priority=2),
    Target(lat=35.7,   lon=139.7,  name="Tokyo",          priority=2),
    Target(lat=-33.9,  lon=151.2,  name="Sydney",         priority=3),
    Target(lat=48.9,   lon=2.3,    name="Paris",          priority=3),
    Target(lat=55.8,   lon=37.6,   name="Moscow",         priority=4),
    Target(lat=-23.5,  lon=-46.6,  name="Sao Paulo",      priority=4),
    Target(lat=28.6,   lon=77.2,   name="New Delhi",      priority=5),
    Target(lat=1.4,    lon=103.8,  name="Singapore",      priority=5),
    Target(lat=-1.3,   lon=36.8,   name="Nairobi",        priority=6),
    Target(lat=19.4,   lon=-99.1,  name="Mexico City",    priority=6),
    Target(lat=64.1,   lon=-21.9,  name="Reykjavik",      priority=7),
    Target(lat=-54.8,  lon=-68.3,  name="Ushuaia",        priority=8),
    Target(lat=90.0,   lon=0.0,    name="North Pole",     priority=9),
    Target(lat=-90.0,  lon=0.0,    name="South Pole",     priority=9),
]

# ── Build globe ───────────────────────────────────────────────────────────────
globe = (
    Globe3D(title="3-D Target Globe — ISS & Sentinel-2A", graticule_step=30)
    # 15-degree global target grid
    .add_target_grid(
        lat_range=(-75, 75),
        lon_range=(-180, 180),
        step=15,
        priority=5,
        group_name="15-deg Global Grid",
    )
    # Named city targets (coloured by priority)
    .add_targets(cities, group_name="City Targets")
    # ISS: one full orbit + 10 min
    .add_ground_track(iss, jd_start=iss_epoch, duration_min=105, step_sec=30,
                      color="#00e5ff")
    # Sentinel-2A: one full orbit
    .add_ground_track(s2a, jd_start=s2a_epoch, duration_min=100, step_sec=30,
                      color="#ff6d00")
    # Current satellite positions
    .add_satellite_position(iss, jd=iss_epoch, color="#00e5ff")
    .add_satellite_position(s2a, jd=s2a_epoch, color="#ff6d00")
)

# Show interactively in browser
globe.show()

# Also save as standalone HTML
out = Path(__file__).parent / "globe_output.html"
globe.save(out)
print(f"\nSaved to {out}")

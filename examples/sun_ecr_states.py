"""
Sun ECR state vectors — JPL DE432s vs VSOP87 comparison.

Run with:
    python examples/sun_ecr_states.py

Prints three tables for each epoch:
  1. JPL    — position/velocity from sun_jpl.py  (DE432s kernel, high accuracy)
  2. VSOP87 — position/velocity from sun_eci.py  (truncated VSOP87, ~2–4 arcsec)
  3. Diff   — |pos_jpl - pos_vsop| and |vel_jpl - vel_vsop|

Uses the same epochs as moon_ecr_states.py.
de432s.bsp covers 1949-12-14 to 2050-01-02; keep all epochs within this range.

Note on ECR velocity: at ~1 AU, ω_E × r dominates (~10 900 km/s), so |v_ecr|
reflects Earth's daily rotation sweeping the Sun. |v_eci| (~30 km/s) is the
physically meaningful orbital speed.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

from tasking_helper.utils.sun_jpl import sun_state_eci as jpl_state_eci
from tasking_helper.utils.sun_jpl import sun_state_ecr as jpl_state_ecr
from tasking_helper.utils.sun_jpl import setup
from tasking_helper.utils.sun_eci import sun_state_eci as vsop_state_eci
from tasking_helper.utils.sun_eci import sun_state_ecr as vsop_state_ecr

# ── Epochs (same as moon_ecr_states.py) ──────────────────────────────────────

TIMES = [
    datetime(2025, 1,  1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 3, 21, 6, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 6, 21, 12, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 9, 23, 18, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 12, 21, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2030, 1,  1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2030, 3, 21, 6, 0, 0, tzinfo=timezone.utc),
    datetime(2030, 6, 21, 12, 0, 0, tzinfo=timezone.utc),
    datetime(2030, 9, 23, 18, 0, 0, tzinfo=timezone.utc),
    datetime(2030, 12, 21, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2035, 1,  1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2035, 3, 21, 6, 0, 0, tzinfo=timezone.utc),
    datetime(2035, 6, 21, 12, 0, 0, tzinfo=timezone.utc),
    datetime(2035, 9, 23, 18, 0, 0, tzinfo=timezone.utc),
    datetime(2035, 12, 21, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2040, 1,  1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2040, 3, 21, 6, 0, 0, tzinfo=timezone.utc),
    datetime(2040, 6, 21, 12, 0, 0, tzinfo=timezone.utc),
    datetime(2040, 9, 23, 18, 0, 0, tzinfo=timezone.utc),
    datetime(2040, 12, 21, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2045, 1,  1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2045, 3, 21, 6, 0, 0, tzinfo=timezone.utc),
    datetime(2045, 6, 21, 12, 0, 0, tzinfo=timezone.utc),
    datetime(2045, 9, 23, 18, 0, 0, tzinfo=timezone.utc),
    datetime(2045, 12, 21, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2050, 1,  1, 0, 0, 0, tzinfo=timezone.utc),
]

# Option B: uniform grid  (uncomment and set parameters)
# _START = datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
# _STEP_S = 86400
# _N = 365
# TIMES = [_START + timedelta(seconds=_STEP_S * i) for i in range(_N)]

_AU = 149_597_870.7

# ── Helpers ───────────────────────────────────────────────────────────────────

_WE = 26   # epoch column width
_WP = 16   # position component width
_WV = 15   # velocity component width
_WN = 12   # norm column width


def _pos_hdr() -> str:
    return (
        f"{'Epoch (UTC)':<{_WE}}  "
        f"{'X [km]':>{_WP}}  {'Y [km]':>{_WP}}  {'Z [km]':>{_WP}}  "
        f"{'|r| [AU]':>{_WN}}  {'|v_eci| [km/s]':>{_WN}}"
    )


def _pos_row(t: datetime, pos_ecr: np.ndarray, vel_eci: np.ndarray) -> str:
    return (
        f"{t.strftime('%Y-%m-%d %H:%M:%S UTC'):<{_WE}}  "
        f"{pos_ecr[0]:{_WP}.3f}  {pos_ecr[1]:{_WP}.3f}  {pos_ecr[2]:{_WP}.3f}  "
        f"{np.linalg.norm(pos_ecr) / _AU:{_WN}.6f}  "
        f"{np.linalg.norm(vel_eci):{_WN}.6f}"
    )


def _diff_hdr() -> str:
    return (
        f"{'Epoch (UTC)':<{_WE}}  "
        f"{'|dr_ecr| [km]':>{_WN}}  {'|dv_eci| [m/s]':>{_WN}}  {'|dr| [arcsec]':>{_WN}}"
    )


def _diff_row(t: datetime, p_jpl: np.ndarray, p_vsop: np.ndarray,
              v_jpl: np.ndarray, v_vsop: np.ndarray) -> str:
    dr = np.linalg.norm(p_jpl - p_vsop)
    dv = np.linalg.norm(v_jpl - v_vsop) * 1000.0       # km/s -> m/s
    # angular separation at actual distance
    dist = np.linalg.norm(p_jpl)
    ang  = dr / dist * (180 * 3600 / 3.14159265)        # arcsec
    return (
        f"{t.strftime('%Y-%m-%d %H:%M:%S UTC'):<{_WE}}  "
        f"{dr:{_WN}.2f}  {dv:{_WN}.4f}  {ang:{_WN}.2f}"
    )


def _print_table(title: str, hdr: str, rows: list[str]) -> None:
    sep = "-" * len(hdr)
    print(title)
    print(hdr)
    print(sep)
    for r in rows:
        print(r)
    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    bsp = setup()
    print(f"Kernel (JPL) : {bsp}")
    print(f"Epochs       : {len(TIMES)}")
    print()

    # Evaluate both methods at every epoch
    jpl_pos_ecr_all  = np.empty((3, len(TIMES)))
    jpl_vel_eci_all  = np.empty((3, len(TIMES)))
    vsop_pos_ecr_all = np.empty((3, len(TIMES)))
    vsop_vel_eci_all = np.empty((3, len(TIMES)))

    # JPL batch for ECI velocity; ECR batch for position
    jpl_pos_eci_all, jpl_vel_eci_all = jpl_state_eci(TIMES)
    jpl_pos_ecr_batch, _ = jpl_state_ecr(TIMES)
    jpl_pos_ecr_all = jpl_pos_ecr_batch

    for i, t in enumerate(TIMES):
        vsop_pos_ecr_all[:, i], _ = vsop_state_ecr(t)
        vsop_pos_eci, vsop_vel = vsop_state_eci(t)
        vsop_vel_eci_all[:, i] = vsop_vel

    # Build row strings
    jpl_rows, vsop_rows, diff_rows = [], [], []
    for i, t in enumerate(TIMES):
        jpl_rows.append(_pos_row(t, jpl_pos_ecr_all[:, i], jpl_vel_eci_all[:, i]))
        vsop_rows.append(_pos_row(t, vsop_pos_ecr_all[:, i], vsop_vel_eci_all[:, i]))
        diff_rows.append(_diff_row(t,
                                   jpl_pos_ecr_all[:, i],  vsop_pos_ecr_all[:, i],
                                   jpl_vel_eci_all[:, i],  vsop_vel_eci_all[:, i]))

    hdr = _pos_hdr()
    _print_table("=== JPL DE432s (sun_jpl.py) ===", hdr, jpl_rows)
    _print_table("=== VSOP87 truncated (sun_eci.py) ===", hdr, vsop_rows)

    diff_hdr = _diff_hdr()
    sep = "-" * len(diff_hdr)
    dr_vals = np.array([np.linalg.norm(jpl_pos_ecr_all[:, i] - vsop_pos_ecr_all[:, i])
                        for i in range(len(TIMES))])
    dv_vals = np.array([np.linalg.norm(jpl_vel_eci_all[:, i] - vsop_vel_eci_all[:, i]) * 1000
                        for i in range(len(TIMES))])

    print("=== Difference: JPL - VSOP87 ===")
    print(diff_hdr)
    print(sep)
    for r in diff_rows:
        print(r)
    print(sep)
    print(f"  mean |dr|: {dr_vals.mean():.0f} km    max |dr|: {dr_vals.max():.0f} km")
    print(f"  mean |dv|: {dv_vals.mean():.4f} m/s  max |dv|: {dv_vals.max():.4f} m/s")


if __name__ == "__main__":
    main()

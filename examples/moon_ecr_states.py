"""
Moon ECR state vectors — JPL DE432s vs ELP2000/82 comparison.

Run with:
    python examples/moon_ecr_states.py

Prints three tables for each epoch:
  1. JPL  — position/velocity from moon_jpl.py  (DE432s kernel, high accuracy)
  2. ELP  — position/velocity from moon_eci.py  (Meeus ELP2000/82 + IAU 1980 nutation)
  3. Diff — |pos_jpl - pos_elp| and |vel_jpl - vel_elp|

Edit TIMES below to choose your epochs.
de432s.bsp covers 1949-12-14 to 2050-01-02; keep all epochs within this range.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

from tasking_helper.utils.moon_jpl import moon_state_ecr as jpl_state_ecr, setup
from tasking_helper.utils.moon_eci import moon_state_ecr as elp_state_ecr

# ── Configure epochs here ─────────────────────────────────────────────────────

# Option A: explicit list of datetimes
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
# _START = datetime(2030, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
# _STEP_S = 3600          # seconds between epochs
# _N = 24                 # number of epochs
# TIMES = [_START + timedelta(seconds=_STEP_S * i) for i in range(_N)]


# ── Helpers ───────────────────────────────────────────────────────────────────

_W_EPOCH = 26
_W_XYZ   = 14
_W_VEL   = 11
_W_NORM  = 12


def _state_hdr() -> str:
    return (
        f"{'Epoch (UTC)':<{_W_EPOCH}}  "
        f"{'X [km]':>{_W_XYZ}}  {'Y [km]':>{_W_XYZ}}  {'Z [km]':>{_W_XYZ}}  "
        f"{'Vx [km/s]':>{_W_VEL}}  {'Vy [km/s]':>{_W_VEL}}  {'Vz [km/s]':>{_W_VEL}}  "
        f"{'|r| [km]':>{_W_NORM}}  {'|v| [km/s]':>{_W_NORM}}"
    )


def _state_row(t: datetime, pos: np.ndarray, vel: np.ndarray) -> str:
    return (
        f"{t.strftime('%Y-%m-%d %H:%M:%S UTC'):<{_W_EPOCH}}  "
        f"{pos[0]:{_W_XYZ}.3f}  {pos[1]:{_W_XYZ}.3f}  {pos[2]:{_W_XYZ}.3f}  "
        f"{vel[0]:{_W_VEL}.6f}  {vel[1]:{_W_VEL}.6f}  {vel[2]:{_W_VEL}.6f}  "
        f"{np.linalg.norm(pos):{_W_NORM}.3f}  {np.linalg.norm(vel):{_W_NORM}.6f}"
    )


def _diff_hdr() -> str:
    return (
        f"{'Epoch (UTC)':<{_W_EPOCH}}  "
        f"{'|dr| [km]':>{_W_NORM}}  {'|dv| [m/s]':>{_W_NORM}}"
    )


def _diff_row(t: datetime, pos_jpl: np.ndarray, pos_elp: np.ndarray,
              vel_jpl: np.ndarray, vel_elp: np.ndarray) -> str:
    dr = np.linalg.norm(pos_jpl - pos_elp)
    dv = np.linalg.norm(vel_jpl - vel_elp) * 1000.0  # km/s -> m/s
    return (
        f"{t.strftime('%Y-%m-%d %H:%M:%S UTC'):<{_W_EPOCH}}  "
        f"{dr:{_W_NORM}.2f}  {dv:{_W_NORM}.4f}"
    )


def _print_table(title: str, hdr: str, rows: list[str]) -> None:
    sep = "-" * len(hdr)
    print(title)
    print(hdr)
    print(sep)
    for row in rows:
        print(row)
    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    bsp = setup()
    print(f"Kernel (JPL) : {bsp}")
    print(f"Epochs       : {len(TIMES)}")
    print()

    # JPL: batch call returns (3, N) arrays
    jpl_pos_all, jpl_vel_all = jpl_state_ecr(TIMES)

    # ELP2000: scalar-only API, loop over epochs
    elp_pos_all = np.empty((3, len(TIMES)))
    elp_vel_all = np.empty((3, len(TIMES)))
    for i, t in enumerate(TIMES):
        elp_pos_all[:, i], elp_vel_all[:, i] = elp_state_ecr(t)

    # Build row strings
    jpl_rows, elp_rows, diff_rows = [], [], []
    for i, t in enumerate(TIMES):
        jpl_rows.append(_state_row(t, jpl_pos_all[:, i], jpl_vel_all[:, i]))
        elp_rows.append(_state_row(t, elp_pos_all[:, i], elp_vel_all[:, i]))
        diff_rows.append(_diff_row(t,
                                   jpl_pos_all[:, i], elp_pos_all[:, i],
                                   jpl_vel_all[:, i], elp_vel_all[:, i]))

    hdr = _state_hdr()
    _print_table("=== JPL DE432s (moon_jpl.py) ===", hdr, jpl_rows)
    _print_table("=== ELP2000/82 + nutation (moon_eci.py) ===", hdr, elp_rows)

    diff_hdr = _diff_hdr()
    diff_sep  = "-" * len(diff_hdr)
    dr_vals = np.array([np.linalg.norm(jpl_pos_all[:, i] - elp_pos_all[:, i])
                        for i in range(len(TIMES))])
    dv_vals = np.array([np.linalg.norm(jpl_vel_all[:, i] - elp_vel_all[:, i]) * 1000
                        for i in range(len(TIMES))])

    print("=== Difference: JPL - ELP2000 ===")
    print(diff_hdr)
    print(diff_sep)
    for row in diff_rows:
        print(row)
    print(diff_sep)
    print(f"  mean |dr|: {dr_vals.mean():.1f} km    "
          f"max |dr|: {dr_vals.max():.1f} km")
    print(f"  mean |dv|: {dv_vals.mean():.4f} m/s  "
          f"max |dv|: {dv_vals.max():.4f} m/s")


if __name__ == "__main__":
    main()

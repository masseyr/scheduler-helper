"""
tasking_helper/gui/analysis_gui.py
═══════════════════════════════════
PyQt6 GUI — EO ground-station sensor analysis.

Propagates a Keplerian orbit, computes visibility windows, SNR, and visual
magnitude for an electro-optical ground sensor tracking a space object.

Run standalone:
    python -m tasking_helper.gui.analysis_gui
    python src/tasking_helper/gui/analysis_gui.py

Requires: PyQt6, matplotlib, numpy
"""

from __future__ import annotations

import sys
import math
from typing import Any

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QPalette, QColor, QLinearGradient, QPainter, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QLabel, QPushButton, QProgressBar,
    QScrollArea, QSplitter, QStatusBar, QMessageBox,
    QSizePolicy,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# ─── parameter groups ─────────────────────────────────────────────────────────
# Each parameter: (key, label, default, lo, hi, decimals, step, unit, tooltip)
_PARAM_GROUPS: list[tuple[str, list]] = [
    ("Orbit (observed object)", [
        ("alt_km",   "Altitude",          550.0,  160.0, 36000.0, 1,  10.0, "km",
         "Mean orbital altitude above Earth's surface"),
        ("incl_deg", "Inclination",        97.6,    0.0,   180.0, 2,   1.0, "deg",
         "Orbital inclination"),
        ("ecc",      "Eccentricity",      0.001,    0.0,    0.99, 4, 0.001, "",
         "Orbital eccentricity (0 = circular)"),
        ("raan_deg", "RAAN",               0.0,    0.0,   360.0, 2,   1.0, "deg",
         "Right ascension of the ascending node"),
        ("argp_deg", "Arg. of Perigee",    0.0,    0.0,   360.0, 2,   1.0, "deg",
         "Argument of perigee"),
        ("m0_deg",   "Mean Anomaly",       0.0,    0.0,   360.0, 2,   1.0, "deg",
         "Mean anomaly at epoch"),
    ]),
    ("Target", [
        ("target_diam_m", "Diameter",     1.0,   0.01,  100.0, 2,  0.10, "m",
         "Effective cross-section diameter of the target"),
        ("albedo",        "Albedo",       0.30,   0.0,    1.0,  3,  0.05, "",
         "Diffuse (Lambertian) reflectance, 0–1"),
    ]),
    ("Observer / Location", [
        ("obs_lat_deg", "Latitude",       51.5,  -90.0,   90.0, 2,  1.0, "deg",
         "Observer geodetic latitude"),
        ("obs_lon_deg", "Longitude",      -0.1, -180.0,  180.0, 2,  1.0, "deg",
         "Observer geodetic longitude"),
        ("obs_alt_m",   "Altitude",       50.0,   0.0,  5000.0, 1, 10.0, "m",
         "Observer altitude above WGS-84 ellipsoid"),
        ("min_el_deg",  "Min. Elevation", 10.0,   0.0,   90.0,  1,  1.0, "deg",
         "Minimum elevation angle for a visible pass"),
    ]),
    ("Sensor", [
        ("aperture_m",   "Aperture",      0.50,  0.01,    5.0, 3, 0.05, "m",
         "Lens / mirror clear aperture diameter"),
        ("focal_len_mm", "Focal Length",  2000.0, 50.0, 20000.0, 0, 100.0, "mm",
         "Effective focal length"),
        ("pixel_um",     "Pixel Pitch",   6.5,   1.0,   50.0, 2,  0.5, "µm",
         "Detector pixel pitch"),
        ("exposure_s",   "Exposure",      0.01, 1e-5,   10.0, 4, 0.005, "s",
         "Single-frame integration time"),
        ("qe",           "Quantum Eff.",  0.80,  0.05,   1.0, 2, 0.05, "",
         "Detector quantum efficiency"),
        ("read_noise_e", "Read Noise",    5.0,   0.0,  200.0, 1,  1.0, "e⁻",
         "RMS read noise per pixel"),
        ("snr_threshold","SNR Threshold", 5.0,   1.0,  100.0, 1,  0.5, "",
         "Minimum SNR required for detection"),
        ("vizmag_limit", "Vmag Limit",   14.0,   5.0,   25.0, 1,  0.5, "mag",
         "Faint limiting visual magnitude of the sensor"),
        ("loop_gain_db", "Loop Gain",    20.0,   0.0,   60.0, 1,  1.0, "dB",
         "Optical / electronic loop gain applied to signal"),
    ]),
    ("Simulation", [
        ("duration_min", "Duration",     120.0,  10.0, 1440.0, 0,  10.0, "min",
         "Total simulation duration"),
        ("step_sec",     "Time Step",     10.0,   1.0,   60.0, 1,   1.0, "s",
         "Propagation step size"),
    ]),
]

# ─── physics / orbital mechanics ─────────────────────────────────────────────

_MU_KM3_S2 = 398_600.4418       # Earth GM [km³/s²]
_R_E_KM    = 6_378.137          # Earth equatorial radius [km]
_E2_WGS84  = 0.006_694_379_990_14
_F_SUN     = 1_361.0            # Solar irradiance [W/m²]
_H_PLANCK  = 6.626e-34          # J·s
_C_LIGHT   = 3.0e8              # m/s
_LAM_VIS   = 550e-9             # reference wavelength (green) [m]
_F_VEGA_V  = 3.64e-9            # V-band Vega flux [W/m²]   (zero-point)


def _kepler_solve(M: float, e: float, tol: float = 1e-12) -> float:
    """Newton–Raphson solution of Kepler's equation M = E − e·sin E."""
    E = M + e * math.sin(M) * (1.0 + e * math.cos(M))
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def _keplerian_pos_eci(
    a: float, e: float, i: float, raan: float, argp: float, M: float
) -> np.ndarray:
    """
    Return ECI position [km] from Keplerian elements.

    All angles in radians; a in km.
    """
    E  = _kepler_solve(M, e)
    nu = 2.0 * math.atan2(
        math.sqrt(1 + e) * math.sin(E / 2),
        math.sqrt(1 - e) * math.cos(E / 2),
    )
    r = a * (1.0 - e * math.cos(E))

    xo = r * math.cos(nu)
    yo = r * math.sin(nu)

    ci, si = math.cos(i),    math.sin(i)
    cr, sr = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)

    x = (cr * cw - sr * sw * ci) * xo + (-cr * sw - sr * cw * ci) * yo
    y = (sr * cw + cr * sw * ci) * xo + (-sr * sw + cr * cw * ci) * yo
    z = (sw * si)                * xo + (cw * si)                 * yo
    return np.array([x, y, z])


def _gmst_rad(t_sec: float) -> float:
    """Approximate Greenwich Mean Sidereal Time [rad] at t seconds after J2000."""
    return math.radians((280.460_618_37 + 360.985_647_24 * t_sec / 86_400.0) % 360.0)


def _lla_to_eci(lat_deg: float, lon_deg: float, alt_m: float, t_sec: float) -> np.ndarray:
    """Geodetic lat/lon/alt → ECI [km] at time t_sec past J2000."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    alt_km = alt_m / 1_000.0

    N = _R_E_KM / math.sqrt(1.0 - _E2_WGS84 * math.sin(lat) ** 2)
    xe = (N + alt_km) * math.cos(lat) * math.cos(lon)
    ye = (N + alt_km) * math.cos(lat) * math.sin(lon)
    ze = (N * (1.0 - _E2_WGS84) + alt_km) * math.sin(lat)

    th = _gmst_rad(t_sec)
    x = xe * math.cos(th) - ye * math.sin(th)
    y = xe * math.sin(th) + ye * math.cos(th)
    return np.array([x, y, ze])


def _elevation_range(obs_eci: np.ndarray, sat_eci: np.ndarray,
                     lat_deg: float, lon_deg: float, t_sec: float
                     ) -> tuple[float, float]:
    """Return (elevation_deg, range_km) of satellite from observer."""
    diff      = sat_eci - obs_eci
    range_km  = float(np.linalg.norm(diff))

    lat  = math.radians(lat_deg)
    th   = _gmst_rad(t_sec) + math.radians(lon_deg)
    up   = np.array([math.cos(lat) * math.cos(th),
                     math.cos(lat) * math.sin(th),
                     math.sin(lat)])
    sin_el = float(np.dot(diff / range_km, up))
    el_deg = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    return el_deg, range_km


def _sun_dir_eci(t_sec: float) -> np.ndarray:
    """Approximate unit vector from Earth to Sun in ECI frame."""
    T   = t_sec / (86_400.0 * 365.25)
    L   = math.radians(280.460 + 36_000.771 * T)
    M   = math.radians(357.528 + 35_999.050 * T)
    lam = L + math.radians(1.915) * math.sin(M) + math.radians(0.020) * math.sin(2 * M)
    eps = math.radians(23.439 - 0.000_000_4 * T * 36_525)
    return np.array([math.cos(lam),
                     math.sin(lam) * math.cos(eps),
                     math.sin(lam) * math.sin(eps)])


def _is_illuminated(sat_eci: np.ndarray, sun_dir: np.ndarray) -> bool:
    """True if satellite is outside Earth's geometric shadow (cylinder approx)."""
    proj = float(np.dot(sat_eci, sun_dir))
    if proj > 0.0:
        return True
    perp = np.linalg.norm(sat_eci - proj * sun_dir)
    return perp > _R_E_KM


def _compute_snr(
    range_km: float,
    target_diam_m: float,
    albedo: float,
    aperture_m: float,
    exposure_s: float,
    qe: float,
    read_noise_e: float,
    loop_gain_db: float,
) -> float:
    """Signal-to-noise ratio for reflected solar light (Lambertian reflector)."""
    range_m     = range_km * 1_000.0
    target_area = math.pi * (target_diam_m / 2.0) ** 2
    aperture_area = math.pi * (aperture_m / 2.0) ** 2

    # Flux at sensor from Lambertian sphere
    flux = _F_SUN * albedo * target_area / (math.pi * range_m ** 2)

    # Loop (electronic/optical) gain
    gain = 10.0 ** (loop_gain_db / 10.0)

    # Signal electrons
    photon_rate = flux * _LAM_VIS / (_H_PLANCK * _C_LIGHT)
    sig_e = photon_rate * aperture_area * qe * exposure_s * gain

    noise = math.sqrt(max(sig_e, 0.0) + read_noise_e ** 2)
    return sig_e / noise if noise > 0 else 0.0


def _compute_vizmag(range_km: float, target_diam_m: float, albedo: float) -> float:
    """Approximate visual magnitude of reflected-light target."""
    range_m     = range_km * 1_000.0
    target_area = math.pi * (target_diam_m / 2.0) ** 2
    flux = _F_SUN * albedo * target_area / (math.pi * range_m ** 2)
    if flux <= 0:
        return 30.0
    return -2.5 * math.log10(flux / _F_VEGA_V)


# ─── analysis worker ──────────────────────────────────────────────────────────

class AnalysisWorker(QObject):
    """Runs the orbit propagation and analysis in a background QThread."""

    progress = pyqtSignal(int, str)   # (percent 0-100, status message)
    finished = pyqtSignal(dict)        # results dict
    error    = pyqtSignal(str)

    def __init__(self, params: dict[str, float]) -> None:
        super().__init__()
        self.params     = params
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            results = self._analyse()
            if not self._cancelled:
                self.finished.emit(results)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    # ── core analysis ────────────────────────────────────────────────────────

    def _analyse(self) -> dict:
        p = self.params
        self.progress.emit(0, "Setting up orbit…")

        # Orbital parameters
        a_km   = _R_E_KM + p["alt_km"]                  # semi-major axis [km]
        e      = p["ecc"]
        i      = math.radians(p["incl_deg"])
        raan   = math.radians(p["raan_deg"])
        argp   = math.radians(p["argp_deg"])
        M0     = math.radians(p["m0_deg"])

        # Mean motion [rad/s]
        n = math.sqrt(_MU_KM3_S2 / a_km ** 3)

        duration_s = p["duration_min"] * 60.0
        step_s     = p["step_sec"]
        n_steps    = int(duration_s / step_s) + 1
        times      = np.arange(n_steps) * step_s

        # Result arrays
        elevations = np.full(n_steps, np.nan)
        ranges     = np.full(n_steps, np.nan)
        snrs       = np.full(n_steps, np.nan)
        vmags      = np.full(n_steps, np.nan)
        visible    = np.zeros(n_steps, dtype=bool)
        lats       = np.full(n_steps, np.nan)
        lons_gt    = np.full(n_steps, np.nan)
        illuminated = np.zeros(n_steps, dtype=bool)

        progress_every = max(1, n_steps // 100)

        for idx, t in enumerate(times):
            if self._cancelled:
                break

            if idx % progress_every == 0:
                pct = int(idx / n_steps * 95)
                self.progress.emit(pct, f"Propagating… step {idx}/{n_steps}")

            M = M0 + n * t
            sat_eci = _keplerian_pos_eci(a_km, e, i, raan, argp, M)
            obs_eci = _lla_to_eci(p["obs_lat_deg"], p["obs_lon_deg"],
                                   p["obs_alt_m"], t)

            el, rng = _elevation_range(obs_eci, sat_eci,
                                        p["obs_lat_deg"], p["obs_lon_deg"], t)
            elevations[idx] = el
            ranges[idx]     = rng

            sun_dir = _sun_dir_eci(t)
            illum   = _is_illuminated(sat_eci, sun_dir)
            illuminated[idx] = illum

            if el >= p["min_el_deg"] and illum:
                snr = _compute_snr(
                    rng, p["target_diam_m"], p["albedo"],
                    p["aperture_m"], p["exposure_s"], p["qe"],
                    p["read_noise_e"], p["loop_gain_db"],
                )
                vm  = _compute_vizmag(rng, p["target_diam_m"], p["albedo"])
                snrs[idx]  = snr
                vmags[idx] = vm
                visible[idx] = (snr >= p["snr_threshold"]
                                and vm <= p["vizmag_limit"])

            # Ground track sub-satellite point (ECEF → lat/lon)
            r_mag = np.linalg.norm(sat_eci)
            th = _gmst_rad(t)
            xe =  sat_eci[0] * math.cos(th) + sat_eci[1] * math.sin(th)
            ye = -sat_eci[0] * math.sin(th) + sat_eci[1] * math.cos(th)
            ze =  sat_eci[2]
            lats[idx]    = math.degrees(math.asin(ze / r_mag))
            lons_gt[idx] = math.degrees(math.atan2(ye, xe))

        self.progress.emit(97, "Collating results…")
        times_min = times / 60.0

        return dict(
            times_min   = times_min,
            elevations  = elevations,
            ranges      = ranges,
            snrs        = snrs,
            vmags       = vmags,
            visible     = visible,
            illuminated = illuminated,
            lats        = lats,
            lons_gt     = lons_gt,
            params      = p,
        )


# ─── parameter panel ──────────────────────────────────────────────────────────

class _UnitLabel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setMinimumWidth(36)
        self.setStyleSheet("color: #888; font-size: 11px;")


class ParameterPanel(QScrollArea):
    """Scrollable panel with grouped parameter spinboxes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(340)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        self._spinboxes: dict[str, QDoubleSpinBox] = {}

        for group_name, params in _PARAM_GROUPS:
            box = QGroupBox(group_name)
            form = QFormLayout(box)
            form.setSpacing(4)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

            for key, label, default, lo, hi, dec, step, unit, tip in params:
                sb = QDoubleSpinBox()
                sb.setRange(lo, hi)
                sb.setDecimals(dec)
                sb.setSingleStep(step)
                sb.setValue(default)
                sb.setToolTip(tip)
                sb.setMinimumWidth(110)

                row = QWidget()
                hl  = QHBoxLayout(row)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(sb)
                if unit:
                    hl.addWidget(_UnitLabel(unit))
                hl.addStretch()

                form.addRow(label, row)
                self._spinboxes[key] = sb

            layout.addWidget(box)

        layout.addStretch()
        self.setWidget(inner)

    def get_params(self) -> dict[str, float]:
        return {k: sb.value() for k, sb in self._spinboxes.items()}

    def set_params(self, params: dict[str, float]) -> None:
        for k, v in params.items():
            if k in self._spinboxes:
                self._spinboxes[k].setValue(v)


# ─── results panel ────────────────────────────────────────────────────────────

_PASS_COLOR    = "#2196F3"   # blue  — satellite above min elevation
_DETECT_COLOR  = "#4CAF50"   # green — detected (SNR + vmag OK)
_SNR_COLOR     = "#FF9800"   # orange
_VMAG_COLOR    = "#9C27B0"   # purple

class ResultsPanel(QWidget):
    """Matplotlib canvas showing analysis results in four subplots + ground track."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._fig = Figure(figsize=(10, 8), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._axes: list = []
        self._placeholder()

    def _placeholder(self) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, "Run the analysis to see results",
                ha="center", va="center", fontsize=14, color="#aaa",
                transform=ax.transAxes)
        ax.axis("off")
        self._canvas.draw()

    def update(self, results: dict) -> None:  # type: ignore[override]
        self._fig.clear()
        self._draw(results)
        self._canvas.draw()

    def _draw(self, r: dict) -> None:
        t          = r["times_min"]
        elevations = r["elevations"]
        ranges     = r["ranges"]
        snrs       = r["snrs"]
        vmags      = r["vmags"]
        visible    = r["visible"]
        lats       = r["lats"]
        lons_gt    = r["lons_gt"]
        p          = r["params"]

        gs = self._fig.add_gridspec(3, 2, hspace=0.45, wspace=0.32,
                                     left=0.08, right=0.97,
                                     top=0.95, bottom=0.06)
        ax_el   = self._fig.add_subplot(gs[0, 0])
        ax_rng  = self._fig.add_subplot(gs[0, 1])
        ax_snr  = self._fig.add_subplot(gs[1, 0])
        ax_vmag = self._fig.add_subplot(gs[1, 1])
        ax_gt   = self._fig.add_subplot(gs[2, :])

        # ── elevation ────────────────────────────────────────────────────────
        ax_el.plot(t, elevations, color="#555", lw=1, label="Elevation")
        ax_el.axhline(p["min_el_deg"], color=_PASS_COLOR, ls="--", lw=1,
                      label=f"Min el. ({p['min_el_deg']:.0f}°)")
        _shade_visible(ax_el, t, elevations >= p["min_el_deg"],
                       color=_PASS_COLOR, alpha=0.12)
        ax_el.set_xlabel("Time (min)"); ax_el.set_ylabel("Elevation (deg)")
        ax_el.set_title("Elevation angle"); ax_el.legend(fontsize=8)
        ax_el.set_ylim(-90, 90); ax_el.grid(True, alpha=0.3)

        # ── range ────────────────────────────────────────────────────────────
        ax_rng.plot(t, ranges, color=_PASS_COLOR, lw=1.2)
        ax_rng.set_xlabel("Time (min)"); ax_rng.set_ylabel("Range (km)")
        ax_rng.set_title("Slant range"); ax_rng.grid(True, alpha=0.3)

        # ── SNR ──────────────────────────────────────────────────────────────
        snr_vals = np.where(np.isnan(snrs), 0.0, snrs)
        ax_snr.semilogy(t, np.maximum(snr_vals, 1e-3),
                        color=_SNR_COLOR, lw=1.2, label="SNR")
        ax_snr.axhline(p["snr_threshold"], color="red", ls="--", lw=1,
                       label=f"Threshold ({p['snr_threshold']:.1f})")
        _shade_visible(ax_snr, t, visible, color=_DETECT_COLOR, alpha=0.20)
        ax_snr.set_xlabel("Time (min)"); ax_snr.set_ylabel("SNR")
        ax_snr.set_title("Signal-to-noise ratio"); ax_snr.legend(fontsize=8)
        ax_snr.grid(True, alpha=0.3, which="both")

        # ── visual magnitude ─────────────────────────────────────────────────
        vmag_vals = np.where(np.isnan(vmags), 30.0, vmags)
        ax_vmag.plot(t, vmag_vals, color=_VMAG_COLOR, lw=1.2, label="Vmag")
        ax_vmag.axhline(p["vizmag_limit"], color="red", ls="--", lw=1,
                        label=f"Limit ({p['vizmag_limit']:.1f})")
        ax_vmag.invert_yaxis()
        _shade_visible(ax_vmag, t, visible, color=_DETECT_COLOR, alpha=0.20)
        ax_vmag.set_xlabel("Time (min)"); ax_vmag.set_ylabel("Visual magnitude")
        ax_vmag.set_title("Visual magnitude (brighter ↑)")
        ax_vmag.legend(fontsize=8); ax_vmag.grid(True, alpha=0.3)

        # ── ground track ─────────────────────────────────────────────────────
        ax_gt.set_facecolor("#e8f0fe")
        ax_gt.plot(lons_gt, lats, color="#555", lw=0.8, alpha=0.6, zorder=2)
        # Highlight detected portions
        if visible.any():
            ax_gt.scatter(lons_gt[visible], lats[visible],
                          c=_DETECT_COLOR, s=6, zorder=4, label="Detected")
        # Observer location
        ax_gt.scatter([p["obs_lon_deg"]], [p["obs_lat_deg"]],
                      marker="^", s=80, c="red", zorder=5, label="Observer")
        ax_gt.set_xlim(-180, 180); ax_gt.set_ylim(-90, 90)
        ax_gt.set_xlabel("Longitude (deg)"); ax_gt.set_ylabel("Latitude (deg)")
        ax_gt.set_title("Ground track"); ax_gt.legend(fontsize=8)
        ax_gt.grid(True, alpha=0.3)
        _draw_coastlines_approx(ax_gt)

        # ── detection summary in title ────────────────────────────────────────
        n_passes  = _count_passes(elevations >= p["min_el_deg"])
        det_frac  = float(visible.sum()) / len(visible) * 100 if len(visible) else 0
        self._fig.suptitle(
            f"Passes above {p['min_el_deg']:.0f}°: {n_passes}   |   "
            f"Detection fraction: {det_frac:.1f}%   |   "
            f"Duration: {p['duration_min']:.0f} min",
            fontsize=10, y=0.998,
        )


def _shade_visible(ax, t: np.ndarray, mask: np.ndarray,
                   color: str, alpha: float) -> None:
    """Shade time intervals where mask is True."""
    in_block = False
    t0 = 0.0
    for i, m in enumerate(mask):
        if m and not in_block:
            t0 = t[i]; in_block = True
        elif not m and in_block:
            ax.axvspan(t0, t[i], color=color, alpha=alpha, lw=0)
            in_block = False
    if in_block:
        ax.axvspan(t0, t[-1], color=color, alpha=alpha, lw=0)


def _count_passes(above: np.ndarray) -> int:
    """Count contiguous True blocks."""
    return int(np.sum(np.diff(above.astype(int)) == 1))


def _draw_coastlines_approx(ax) -> None:
    """Draw a rough world coastline outline using simplified polygons."""
    try:
        # Use cartopy or shapely if available — skip silently if not
        import importlib
        if importlib.util.find_spec("cartopy") is not None:
            import cartopy.crs as ccrs  # type: ignore
            import cartopy.feature as cfeature  # type: ignore
            # Can't add cartopy to an existing Axes this way — skip
            pass
    except Exception:
        pass
    # Minimal continents: draw a box outline so the plot isn't blank
    for lon0, lat0, lon1, lat1 in [
        (-125, 25, -65, 50),  # North America
        (-80, -55, -35, 10),  # South America
        (-10, 35, 40, 70),    # Europe
        (25, -35, 52, 38),    # Africa
        (60, 10, 140, 55),    # Asia
        (113, -45, 155, -10), # Australia
    ]:
        ax.plot([lon0, lon1, lon1, lon0, lon0],
                [lat0, lat0, lat1, lat1, lat0],
                color="#bbb", lw=0.5, zorder=1)


# ─── banner ───────────────────────────────────────────────────────────────────

class BannerWidget(QWidget):
    """
    Fixed-height header banner: dark-blue gradient background, title on the
    left and a subtitle / version tag on the right.
    """

    _BG_LEFT  = QColor("#0D47A1")
    _BG_RIGHT = QColor("#1565C0")
    _ACCENT   = QColor("#42A5F5")

    def __init__(
        self,
        title: str = "EO Ground Station Analysis",
        subtitle: str = "Keplerian orbit propagation · SNR · Visual magnitude",
        version: str = "v0.1",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._title    = title
        self._subtitle = subtitle
        self._version  = version
        self.setFixedHeight(58)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Gradient background
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, self._BG_LEFT)
        grad.setColorAt(1.0, self._BG_RIGHT)
        p.fillRect(self.rect(), QBrush(grad))

        # Accent bar on the left edge
        p.fillRect(0, 0, 5, self.height(), self._ACCENT)

        # Title
        title_font = QFont("Segoe UI", 15, QFont.Weight.Bold)
        p.setFont(title_font)
        p.setPen(QColor("#FFFFFF"))
        p.drawText(18, 0, self.width() - 120, self.height(),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self._title)

        # Subtitle (below title, smaller)
        sub_font = QFont("Segoe UI", 8)
        p.setFont(sub_font)
        p.setPen(QColor("#90CAF9"))
        p.drawText(20, 26, self.width() - 130, self.height() - 26,
                   Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                   self._subtitle)

        # Version tag — right-aligned
        ver_font = QFont("Consolas", 8)
        p.setFont(ver_font)
        p.setPen(QColor("#64B5F6"))
        p.drawText(0, 0, self.width() - 10, self.height(),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   self._version)

        p.end()


# ─── main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EO Ground Station Sensor Analysis")
        self.resize(1200, 780)

        self._worker: AnalysisWorker | None = None
        self._thread: QThread | None = None

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── banner ───────────────────────────────────────────────────────────
        root.addWidget(BannerWidget())

        # ── tab widget ───────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        # Tab 0: Parameters
        self._params_panel = ParameterPanel()
        self._tabs.addTab(self._params_panel, "Parameters")

        # Tab 1: Results
        self._results_panel = ResultsPanel()
        self._tabs.addTab(self._results_panel, "Results")

        # ── bottom bar: progress + buttons ───────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%")
        self._progress.setFixedHeight(22)

        self._run_btn = QPushButton("▶  Run Analysis")
        self._run_btn.setFixedHeight(30)
        self._run_btn.setDefault(True)
        self._run_btn.setStyleSheet(
            "QPushButton { background: #1976D2; color: white; font-weight: bold;"
            "  border-radius: 4px; padding: 0 16px; }"
            "QPushButton:hover  { background: #1565C0; }"
            "QPushButton:pressed{ background: #0D47A1; }"
            "QPushButton:disabled{ background: #aaa; }"
        )
        self._run_btn.clicked.connect(self._start_analysis)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(30)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_analysis)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        bar.addWidget(self._status_lbl)
        bar.addWidget(self._progress, stretch=1)
        bar.addWidget(self._run_btn)
        bar.addWidget(self._cancel_btn)
        root.addLayout(bar)

    # ── analysis lifecycle ───────────────────────────────────────────────────

    def _start_analysis(self) -> None:
        if self._thread and self._thread.isRunning():
            return

        params = self._params_panel.get_params()

        self._worker = AnalysisWorker(params)
        self._thread = QThread(self)

        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_done)

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)
        self._status_lbl.setText("Running…")

        self._thread.start()

    def _cancel_analysis(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._status_lbl.setText("Cancelled.")
        self._cancel_btn.setEnabled(False)

    # ── slots ────────────────────────────────────────────────────────────────

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self._status_lbl.setText(msg)

    def _on_finished(self, results: dict) -> None:
        self._progress.setValue(100)
        self._status_lbl.setText("Done — updating plots…")
        self._results_panel.update(results)
        self._tabs.setCurrentIndex(1)
        self._status_lbl.setText("Analysis complete.")

    def _on_error(self, msg: str) -> None:
        self._progress.setValue(0)
        self._status_lbl.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Analysis Error", msg)

    def _on_thread_done(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None
        self._thread = None

    # ── close event ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._cancel_analysis()
        super().closeEvent(event)


# ─── entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    app.setApplicationName("EO Analysis")
    app.setStyle("Fusion")

    # Slightly warmer palette
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window,      QColor("#f5f5f5"))
    pal.setColor(QPalette.ColorRole.WindowText,  QColor("#212121"))
    pal.setColor(QPalette.ColorRole.Base,        QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#fafafa"))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    # Allow running directly from repo root without installing
    import pathlib, sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
    raise SystemExit(main())

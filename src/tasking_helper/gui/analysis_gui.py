"""
tasking_helper/gui/analysis_gui.py
═══════════════════════════════════
PyQt6 GUI — EO ground-station sensor analysis.

Propagates a Keplerian orbit, computes visibility windows, SNR, and visual
magnitude for one or more electro-optical ground sensors tracking a space object.

Run standalone:
    python -m tasking_helper.gui.analysis_gui
    python src/tasking_helper/gui/analysis_gui.py

Requires: PyQt6, matplotlib, numpy
"""

from __future__ import annotations

import os
import sys
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QPalette, QColor, QLinearGradient, QPainter, QBrush
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QProgressBar,
    QScrollArea, QSizePolicy, QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# ─── scene / sensor parameter specs ──────────────────────────────────────────
# Each entry: (key, label, default, lo, hi, decimals, step, unit, tooltip)

_SCENE_PARAM_GROUPS: list[tuple[str, list]] = [
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
    ("Simulation", [
        ("duration_min", "Duration",     120.0,  10.0, 1440.0, 0,  10.0, "min",
         "Total simulation duration"),
        ("step_sec",     "Time Step",     10.0,   1.0,   60.0, 1,   1.0, "s",
         "Propagation step size"),
    ]),
]

# Each entry: (key, label, default, lo, hi, decimals, step, unit, tooltip)
_SENSOR_TYPE_DEFS: dict[str, dict] = {
    "Ground Optical": {
        "params": [
            ("aperture_m",    "Aperture",      0.50,  0.01,    5.0, 3, 0.05, "m",
             "Lens / mirror clear aperture diameter"),
            ("focal_len_mm",  "Focal Length",  2000.0, 50.0, 20000.0, 0, 100.0, "mm",
             "Effective focal length"),
            ("pixel_um",      "Pixel Pitch",    6.5,   1.0,   50.0, 2,  0.5, "µm",
             "Detector pixel pitch"),
            ("exposure_s",    "Exposure",       0.01,  1e-5,  10.0, 4, 0.005, "s",
             "Single-frame integration time"),
            ("qe",            "Quantum Eff.",   0.80,  0.05,   1.0, 2,  0.05, "",
             "Detector quantum efficiency"),
            ("read_noise_e",  "Read Noise",     5.0,   0.0,  200.0, 1,  1.0, "e⁻",
             "RMS read noise per pixel"),
            ("snr_threshold", "SNR Threshold",  5.0,   1.0,  100.0, 1,  0.5, "",
             "Minimum SNR required for detection"),
            ("vizmag_limit",  "Vmag Limit",    14.0,   5.0,   25.0, 1,  0.5, "mag",
             "Faint limiting visual magnitude of the sensor"),
            ("loop_gain_db",  "Loop Gain",     20.0,   0.0,   60.0, 1,  1.0, "dB",
             "Optical / electronic loop gain applied to signal"),
        ],
    },
    "Radar": {
        "params": [
            ("freq_ghz",      "Frequency",     10.0,   0.1,  100.0, 1,  1.0, "GHz",
             "Carrier frequency"),
            ("power_kw",      "Peak Power",   100.0,   0.1, 10000.0, 0, 10.0, "kW",
             "Peak transmit power"),
            ("ant_gain_db",   "Antenna Gain",  30.0,   0.0,   60.0, 1,  1.0, "dBi",
             "Transmit/receive antenna gain"),
            ("pulse_us",      "Pulse Width",    1.0,   0.001, 10000.0, 3, 0.1, "µs",
             "Transmitted pulse width"),
            ("prf_hz",        "PRF",         1000.0,   1.0,  1e6,  0, 100.0, "Hz",
             "Pulse repetition frequency"),
            ("noise_fig_db",  "Noise Figure",   3.0,   0.0,   20.0, 1,  0.5, "dB",
             "Receiver noise figure"),
            ("n_pulses",      "Pulses Integrated", 10.0, 1.0, 10000.0, 0, 1.0, "",
             "Number of coherently integrated pulses"),
            ("losses_db",     "System Losses",  3.0,   0.0,   30.0, 1,  0.5, "dB",
             "Total system losses"),
            ("snr_threshold", "SNR Threshold", 13.0,  -20.0,  40.0, 1,  0.5, "dB",
             "Minimum SNR (dB) required for detection"),
        ],
    },
    "Space Optical": {
        "params": [
            ("aperture_m",    "Aperture",      0.30,  0.01,    5.0, 3, 0.05, "m",
             "Lens / mirror clear aperture diameter"),
            ("focal_len_mm",  "Focal Length",  1500.0, 50.0, 20000.0, 0, 100.0, "mm",
             "Effective focal length"),
            ("pixel_um",      "Pixel Pitch",    7.0,   1.0,   50.0, 2,  0.5, "µm",
             "Detector pixel pitch"),
            ("exposure_s",    "Exposure",       0.001, 1e-5,   1.0, 4, 0.001, "s",
             "Single-frame integration time"),
            ("qe",            "Quantum Eff.",   0.75,  0.05,   1.0, 2,  0.05, "",
             "Detector quantum efficiency"),
            ("read_noise_e",  "Read Noise",    20.0,   0.0,  500.0, 0,  5.0, "e⁻",
             "RMS read noise per pixel"),
            ("snr_threshold", "SNR Threshold",  5.0,   1.0,  100.0, 1,  0.5, "",
             "Minimum SNR required for detection"),
            ("vizmag_limit",  "Vmag Limit",    14.0,   5.0,   25.0, 1,  0.5, "mag",
             "Faint limiting visual magnitude of the sensor"),
            ("loop_gain_db",  "Loop Gain",      0.0,   0.0,   40.0, 1,  1.0, "dB",
             "Optical / electronic loop gain applied to signal"),
            ("sensor_alt_km", "Sensor Altitude", 500.0, 200.0, 36000.0, 0, 10.0, "km",
             "Sensor spacecraft orbital altitude"),
            ("sensor_incl_deg", "Sensor Incl.", 98.0,  0.0,  180.0, 1,  1.0, "deg",
             "Sensor spacecraft orbital inclination"),
        ],
    },
}

_SENSOR_TYPE_NAMES = list(_SENSOR_TYPE_DEFS.keys())

_FOR_SHAPES = ["Open", "Rectangular", "Elliptical", "Custom"]

# Columns shown in the Sensors overview table
_SENSOR_TABLE_COLS: list[tuple[str, str]] = [
    ("name",          "Name"),
    ("sensor_type",   "Type"),
    ("snr_threshold", "SNR thr."),
]

# Per-sensor plot colours (cycles if more sensors than colours)
_SENSOR_COLORS = [
    "#2196F3", "#FF9800", "#4CAF50", "#E91E63",
    "#9C27B0", "#00BCD4", "#FF5722", "#795548",
]


def _default_sensor(name: str = "Sensor 1",
                    sensor_type: str = "Ground Optical") -> dict:
    d: dict = {"name": name, "sensor_type": sensor_type}
    for key, _, default, *_ in _SENSOR_TYPE_DEFS[sensor_type]["params"]:
        d[key] = float(default)
    # Field-of-regard defaults
    d["for_shape"]          = "Open"
    d["for_boresight_az"]   = 0.0
    d["for_boresight_el"]   = 90.0
    d["for_width_deg"]      = 10.0
    d["for_height_deg"]     = 10.0
    d["for_custom_points"]  = []
    return d


# ─── physics / orbital mechanics ─────────────────────────────────────────────

_MU_KM3_S2 = 398_600.4418
_R_E_KM    = 6_378.137
_E2_WGS84  = 0.006_694_379_990_14
_F_SUN     = 1_361.0
_H_PLANCK  = 6.626e-34
_C_LIGHT   = 3.0e8
_LAM_VIS   = 550e-9
_F_VEGA_V  = 3.64e-9


def _kepler_solve(M: float, e: float, tol: float = 1e-12) -> float:
    E = M + e * math.sin(M) * (1.0 + e * math.cos(M))
    for _ in range(50):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def _keplerian_pos_eci(
    a: float, e: float, i: float, raan: float, argp: float, M: float,
) -> np.ndarray:
    E  = _kepler_solve(M, e)
    nu = 2.0 * math.atan2(
        math.sqrt(1 + e) * math.sin(E / 2),
        math.sqrt(1 - e) * math.cos(E / 2),
    )
    r  = a * (1.0 - e * math.cos(E))
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
    return math.radians((280.460_618_37 + 360.985_647_24 * t_sec / 86_400.0) % 360.0)


def _lla_to_eci(lat_deg: float, lon_deg: float, alt_m: float, t_sec: float) -> np.ndarray:
    lat    = math.radians(lat_deg)
    lon    = math.radians(lon_deg)
    alt_km = alt_m / 1_000.0
    N  = _R_E_KM / math.sqrt(1.0 - _E2_WGS84 * math.sin(lat) ** 2)
    xe = (N + alt_km) * math.cos(lat) * math.cos(lon)
    ye = (N + alt_km) * math.cos(lat) * math.sin(lon)
    ze = (N * (1.0 - _E2_WGS84) + alt_km) * math.sin(lat)
    th = _gmst_rad(t_sec)
    return np.array([xe * math.cos(th) - ye * math.sin(th),
                     xe * math.sin(th) + ye * math.cos(th),
                     ze])


def _elevation_range(
    obs_eci: np.ndarray, sat_eci: np.ndarray,
    lat_deg: float, lon_deg: float, t_sec: float,
) -> tuple[float, float]:
    diff     = sat_eci - obs_eci
    range_km = float(np.linalg.norm(diff))
    lat  = math.radians(lat_deg)
    th   = _gmst_rad(t_sec) + math.radians(lon_deg)
    up   = np.array([math.cos(lat) * math.cos(th),
                     math.cos(lat) * math.sin(th),
                     math.sin(lat)])
    sin_el = float(np.dot(diff / range_km, up))
    el_deg = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    return el_deg, range_km


def _sun_dir_eci(t_sec: float) -> np.ndarray:
    T   = t_sec / (86_400.0 * 365.25)
    L   = math.radians(280.460 + 36_000.771 * T)
    M   = math.radians(357.528 + 35_999.050 * T)
    lam = L + math.radians(1.915) * math.sin(M) + math.radians(0.020) * math.sin(2 * M)
    eps = math.radians(23.439 - 0.000_000_4 * T * 36_525)
    return np.array([math.cos(lam),
                     math.sin(lam) * math.cos(eps),
                     math.sin(lam) * math.sin(eps)])


def _is_illuminated(sat_eci: np.ndarray, sun_dir: np.ndarray) -> bool:
    proj = float(np.dot(sat_eci, sun_dir))
    if proj > 0.0:
        return True
    return np.linalg.norm(sat_eci - proj * sun_dir) > _R_E_KM


def _compute_snr(
    range_km: float, target_diam_m: float, albedo: float,
    aperture_m: float, exposure_s: float, qe: float,
    read_noise_e: float, loop_gain_db: float,
) -> float:
    range_m       = range_km * 1_000.0
    target_area   = math.pi * (target_diam_m / 2.0) ** 2
    aperture_area = math.pi * (aperture_m    / 2.0) ** 2
    flux          = _F_SUN * albedo * target_area / (math.pi * range_m ** 2)
    gain          = 10.0 ** (loop_gain_db / 10.0)
    sig_e         = (flux * _LAM_VIS / (_H_PLANCK * _C_LIGHT)) * aperture_area * qe * exposure_s * gain
    noise         = math.sqrt(max(sig_e, 0.0) + read_noise_e ** 2)
    return sig_e / noise if noise > 0 else 0.0


def _compute_vizmag(range_km: float, target_diam_m: float, albedo: float) -> float:
    range_m     = range_km * 1_000.0
    target_area = math.pi * (target_diam_m / 2.0) ** 2
    flux = _F_SUN * albedo * target_area / (math.pi * range_m ** 2)
    return -2.5 * math.log10(flux / _F_VEGA_V) if flux > 0 else 30.0


_K_BOLTZ = 1.38e-23


def _compute_snr_radar(
    range_km: float, target_diam_m: float,
    freq_ghz: float, power_kw: float, ant_gain_db: float,
    pulse_us: float, noise_fig_db: float, n_pulses: float, losses_db: float,
) -> float:
    """Radar range equation; returns SNR in dB."""
    lam    = 3e8 / (freq_ghz * 1e9)
    Pt     = power_kw * 1e3
    G      = 10.0 ** (ant_gain_db  / 10.0)
    F      = 10.0 ** (noise_fig_db / 10.0)
    L      = 10.0 ** (losses_db    / 10.0)
    rcs    = math.pi * (target_diam_m / 2.0) ** 2
    R      = range_km * 1e3
    bw     = 1.0 / max(pulse_us * 1e-6, 1e-12)
    N      = max(1.0, n_pulses)
    snr_lin = (Pt * G ** 2 * lam ** 2 * rcs * N) / (
        (4.0 * math.pi) ** 3 * R ** 4 * _K_BOLTZ * 290.0 * bw * F * L
    )
    return 10.0 * math.log10(max(snr_lin, 1e-30))


# ─── analysis worker ──────────────────────────────────────────────────────────

class AnalysisWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, scene_params: dict, sensors: list[dict]) -> None:
        super().__init__()
        self.scene_params = scene_params
        self.sensors      = sensors
        self._cancelled   = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            results = self._analyse()
            if not self._cancelled:
                self.finished.emit(results)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def _analyse(self) -> dict:
        p = self.scene_params
        self.progress.emit(0, "Setting up orbit…")

        a_km = _R_E_KM + p["alt_km"]
        e    = p["ecc"]
        i    = math.radians(p["incl_deg"])
        raan = math.radians(p["raan_deg"])
        argp = math.radians(p["argp_deg"])
        M0   = math.radians(p["m0_deg"])
        n    = math.sqrt(_MU_KM3_S2 / a_km ** 3)

        duration_s = p["duration_min"] * 60.0
        step_s     = p["step_sec"]
        n_steps    = int(duration_s / step_s) + 1
        times      = np.arange(n_steps) * step_s

        # Shared geometry arrays (same for every sensor)
        elevations  = np.full(n_steps, np.nan)
        ranges      = np.full(n_steps, np.nan)
        lats        = np.full(n_steps, np.nan)
        lons_gt     = np.full(n_steps, np.nan)
        illuminated = np.zeros(n_steps, dtype=bool)

        progress_every = max(1, n_steps // 80)

        self.progress.emit(2, "Propagating orbit…")
        for idx, t in enumerate(times):
            if self._cancelled:
                break
            if idx % progress_every == 0:
                pct = 2 + int(idx / n_steps * 45)
                self.progress.emit(pct, f"Propagating orbit… {idx}/{n_steps}")

            M       = M0 + n * t
            sat_eci = _keplerian_pos_eci(a_km, e, i, raan, argp, M)
            obs_eci = _lla_to_eci(p["obs_lat_deg"], p["obs_lon_deg"], p["obs_alt_m"], t)
            el, rng = _elevation_range(obs_eci, sat_eci, p["obs_lat_deg"], p["obs_lon_deg"], t)

            elevations[idx] = el
            ranges[idx]     = rng
            illuminated[idx] = _is_illuminated(sat_eci, _sun_dir_eci(t))

            r_mag = np.linalg.norm(sat_eci)
            th    = _gmst_rad(t)
            xe    =  sat_eci[0] * math.cos(th) + sat_eci[1] * math.sin(th)
            ye    = -sat_eci[0] * math.sin(th) + sat_eci[1] * math.cos(th)
            lats[idx]    = math.degrees(math.asin(sat_eci[2] / r_mag))
            lons_gt[idx] = math.degrees(math.atan2(ye, xe))

        # Per-sensor analysis
        above_min = elevations >= p["min_el_deg"]
        sensor_results: list[dict] = []
        n_sensors = len(self.sensors)

        for s_idx, sensor in enumerate(self.sensors):
            if self._cancelled:
                break
            base_pct = 48 + int(s_idx / n_sensors * 47)
            self.progress.emit(base_pct, f"Analysing {sensor['name']}…")

            snrs    = np.full(n_steps, np.nan)
            vmags   = np.full(n_steps, np.nan)
            visible = np.zeros(n_steps, dtype=bool)
            stype   = sensor.get("sensor_type", "Ground Optical")
            is_radar = stype == "Radar"

            for idx in range(n_steps):
                if not above_min[idx]:
                    continue
                if not is_radar and not illuminated[idx]:
                    continue
                rng = float(ranges[idx])
                if is_radar:
                    snr = _compute_snr_radar(
                        rng, p["target_diam_m"],
                        sensor["freq_ghz"], sensor["power_kw"],
                        sensor["ant_gain_db"], sensor["pulse_us"],
                        sensor["noise_fig_db"], sensor["n_pulses"],
                        sensor["losses_db"],
                    )
                    snrs[idx]    = snr
                    visible[idx] = snr >= sensor["snr_threshold"]
                else:
                    snr = _compute_snr(
                        rng, p["target_diam_m"], p["albedo"],
                        sensor["aperture_m"], sensor["exposure_s"],
                        sensor["qe"], sensor["read_noise_e"],
                        sensor["loop_gain_db"],
                    )
                    vm = _compute_vizmag(rng, p["target_diam_m"], p["albedo"])
                    snrs[idx]    = snr
                    vmags[idx]   = vm
                    visible[idx] = (snr >= sensor["snr_threshold"]
                                    and vm <= sensor.get("vizmag_limit", 99.0))

            sensor_results.append(dict(
                name    = sensor["name"],
                snrs    = snrs,
                vmags   = vmags,
                visible = visible,
                params  = dict(sensor),
            ))

        self.progress.emit(97, "Collating results…")
        return dict(
            times_min   = times / 60.0,
            elevations  = elevations,
            ranges      = ranges,
            lats        = lats,
            lons_gt     = lons_gt,
            illuminated = illuminated,
            scene_params = p,
            sensors     = sensor_results,
        )


# ─── shared UI helpers ────────────────────────────────────────────────────────

class _UnitLabel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setMinimumWidth(36)
        self.setStyleSheet("color: #888; font-size: 11px;")


def _make_param_form(
    params: list[tuple],
    spinboxes: dict,
    changed_slot=None,
) -> QGroupBox | QWidget:
    """Build a QFormLayout from a params list, store spinboxes in *spinboxes* dict."""
    container = QWidget()
    form = QFormLayout(container)
    form.setSpacing(4)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
    form.setContentsMargins(0, 0, 0, 0)

    for key, label, default, lo, hi, dec, step, unit, tip in params:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(dec)
        sb.setSingleStep(step)
        sb.setValue(default)
        sb.setToolTip(tip)
        sb.setMinimumWidth(110)
        if changed_slot is not None:
            sb.valueChanged.connect(changed_slot)

        row = QWidget()
        hl  = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(sb)
        if unit:
            hl.addWidget(_UnitLabel(unit))
        hl.addStretch()

        form.addRow(label, row)
        spinboxes[key] = sb

    return container


# ─── scene / orbit parameter panel ───────────────────────────────────────────

# Mapping from _SCENE_PARAM_GROUPS display name → config file section name
_SCENE_CONFIG_SECTIONS: dict[str, str] = {
    g: g.split("(")[0].strip().replace(" / ", " ").rstrip()
    for g, _ in _SCENE_PARAM_GROUPS
}
# e.g. "Orbit (observed object)" → "Orbit"
#      "Observer / Location"     → "Observer  Location" → we clean up below
_SCENE_CONFIG_SECTIONS = {
    g: g.split("(")[0].strip().replace(" / ", "_")
    for g, _ in _SCENE_PARAM_GROUPS
}


class ParameterPanel(QWidget):
    """
    Scrollable panel for orbit / target / observer / simulation parameters.
    Includes Export / Import buttons to persist settings as a .config file.
    """

    _CONFIG_VERSION = "1"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── toolbar ──────────────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setFixedHeight(32)
        toolbar.setStyleSheet("background:#f0f4ff; border-bottom:1px solid #d0d8f0;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(6, 3, 6, 3)
        tb_layout.setSpacing(6)

        lbl = QLabel("Scene parameters")
        lbl.setStyleSheet("font-weight:bold; font-size:11px; color:#444;")

        self._export_btn = QPushButton("↑ Export .config")
        self._import_btn = QPushButton("↓ Import .config")
        for b in (self._export_btn, self._import_btn):
            b.setFixedHeight(24)
            b.setStyleSheet(_BTN_STYLE)

        self._export_btn.clicked.connect(self._export_config)
        self._import_btn.clicked.connect(self._import_config)

        tb_layout.addWidget(lbl)
        tb_layout.addStretch()
        tb_layout.addWidget(self._export_btn)
        tb_layout.addWidget(self._import_btn)
        root.addWidget(toolbar)

        # ── scrollable form ───────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(340)

        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        self._spinboxes: dict[str, QDoubleSpinBox] = {}

        for group_name, params in _SCENE_PARAM_GROUPS:
            box = QGroupBox(group_name)
            vl  = QVBoxLayout(box)
            vl.setContentsMargins(6, 4, 6, 4)
            vl.addWidget(_make_param_form(params, self._spinboxes))
            layout.addWidget(box)

        layout.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll)

    # ── data access ──────────────────────────────────────────────────────────

    def get_params(self) -> dict[str, float]:
        return {k: sb.value() for k, sb in self._spinboxes.items()}

    def set_params(self, params: dict[str, float]) -> None:
        for k, v in params.items():
            if k in self._spinboxes:
                sb = self._spinboxes[k]
                sb.setValue(max(sb.minimum(), min(sb.maximum(), v)))

    # ── config export ─────────────────────────────────────────────────────────

    def _export_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export scene config", "scene.config",
            "Config files (*.config);;All files (*)")
        if not path:
            return
        import configparser, datetime
        cfg = configparser.ConfigParser()
        cfg["Metadata"] = {
            "version":   self._CONFIG_VERSION,
            "exported":  datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "tasking_helper.gui",
        }
        for group_name, params in _SCENE_PARAM_GROUPS:
            section = _SCENE_CONFIG_SECTIONS[group_name]
            cfg[section] = {
                key: repr(self._spinboxes[key].value())
                for key, *_ in params
            }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                cfg.write(fh)
            print(f"Exported scene config → {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ── config import ─────────────────────────────────────────────────────────

    def _import_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import scene config", "",
            "Config files (*.config);;All files (*)")
        if not path:
            return
        import configparser
        cfg = configparser.ConfigParser()
        try:
            read_ok = cfg.read(path, encoding="utf-8")
            if not read_ok:
                raise OSError("File could not be read")
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return

        # Version check (warn, don't block)
        file_ver = cfg.get("Metadata", "version", fallback=None)
        if file_ver and file_ver != self._CONFIG_VERSION:
            QMessageBox.warning(
                self, "Version mismatch",
                f"Config version {file_ver!r} differs from expected "
                f"{self._CONFIG_VERSION!r}. Values will be loaded anyway.")

        loaded = errors = 0
        for group_name, params in _SCENE_PARAM_GROUPS:
            section = _SCENE_CONFIG_SECTIONS[group_name]
            if section not in cfg:
                continue
            for key, _, _, lo, hi, *_ in params:
                raw = cfg[section].get(key)
                if raw is None:
                    continue
                try:
                    v = float(raw)
                    self._spinboxes[key].setValue(
                        max(lo, min(hi, v)))
                    loaded += 1
                except ValueError:
                    errors += 1

        msg = f"Imported {loaded} parameter(s) from {path}"
        if errors:
            msg += f"  ({errors} value(s) skipped — unparseable)"
        print(msg)
        if errors:
            QMessageBox.warning(self, "Import warnings", msg)


_BTN_STYLE = (
    "QPushButton { border: 1px solid #bbb; border-radius: 3px; padding: 3px 10px; }"
    "QPushButton:hover   { background: #e3f2fd; }"
    "QPushButton:pressed { background: #bbdefb; }"
    "QPushButton:disabled{ color: #aaa; }"
)


# ─── field-of-regard editor ───────────────────────────────────────────────────

class FieldOfRegardEditor(QWidget):
    """
    FoR shape selector + boresight/dimension inputs + custom Az/El point list.
    Embeds inside TypedSensorEditor below the type-tabs.
    """

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._blocked = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        box = QGroupBox("Field of Regard")
        vbl = QVBoxLayout(box)
        vbl.setSpacing(4)
        vbl.setContentsMargins(6, 4, 6, 6)

        # ── shape selector ────────────────────────────────────────────────────
        shape_row = QWidget()
        srl = QHBoxLayout(shape_row)
        srl.setContentsMargins(0, 0, 0, 0)
        lbl_shape = QLabel("Shape:")
        lbl_shape.setMinimumWidth(90)
        lbl_shape.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._shape_combo = QComboBox()
        self._shape_combo.addItems(_FOR_SHAPES)
        self._shape_combo.setFixedWidth(130)
        srl.addWidget(lbl_shape)
        srl.addSpacing(6)
        srl.addWidget(self._shape_combo)
        srl.addStretch()
        vbl.addWidget(shape_row)

        # ── numeric fields (boresight + dims) ─────────────────────────────────
        self._for_form = QFormLayout()
        self._for_form.setSpacing(4)
        self._for_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._for_form.setContentsMargins(0, 2, 0, 2)

        def _sb(lo: float, hi: float, dec: int, step: float, val: float) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setDecimals(dec)
            sb.setSingleStep(step)
            sb.setValue(val)
            sb.setMinimumWidth(90)
            sb.valueChanged.connect(self._on_value_changed)
            return sb

        self._bs_az  = _sb(-360.0, 360.0, 1, 1.0,  0.0)
        self._bs_el  = _sb( -90.0,  90.0, 1, 1.0, 90.0)
        self._width  = _sb(   0.0, 360.0, 2, 0.5, 10.0)
        self._height = _sb(   0.0, 180.0, 2, 0.5, 10.0)

        def _unit_row(sb: QDoubleSpinBox, unit: str) -> QWidget:
            w = QWidget()
            hl = QHBoxLayout(w)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(sb)
            hl.addWidget(_UnitLabel(unit))
            hl.addStretch()
            return w

        self._bs_az_row  = _unit_row(self._bs_az,  "deg")
        self._bs_el_row  = _unit_row(self._bs_el,  "deg")
        self._width_row  = _unit_row(self._width,  "deg")
        self._height_row = _unit_row(self._height, "deg")

        self._for_form.addRow("Boresight Az:", self._bs_az_row)
        self._for_form.addRow("Boresight El:", self._bs_el_row)
        self._for_form.addRow("Width:",        self._width_row)
        self._for_form.addRow("Height:",       self._height_row)
        vbl.addLayout(self._for_form)

        # ── custom Az/El point table ──────────────────────────────────────────
        self._custom_widget = QWidget()
        cvl = QVBoxLayout(self._custom_widget)
        cvl.setContentsMargins(0, 4, 0, 0)
        cvl.setSpacing(4)

        pt_lbl = QLabel("Az / El coordinate pairs (deg):")
        pt_lbl.setStyleSheet("font-size:11px; color:#555;")
        cvl.addWidget(pt_lbl)

        self._pt_table = QTableWidget(0, 2)
        self._pt_table.setHorizontalHeaderLabels(["Az (deg)", "El (deg)"])
        self._pt_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._pt_table.verticalHeader().setVisible(False)
        self._pt_table.setMinimumHeight(100)
        self._pt_table.setMaximumHeight(200)
        self._pt_table.setAlternatingRowColors(True)
        self._pt_table.itemChanged.connect(self._on_value_changed)
        cvl.addWidget(self._pt_table)

        pt_btns = QHBoxLayout()
        self._add_pt_btn = QPushButton("+ Add Point")
        self._rem_pt_btn = QPushButton("− Remove")
        for b in (self._add_pt_btn, self._rem_pt_btn):
            b.setFixedHeight(24)
            b.setStyleSheet(_BTN_STYLE)
            pt_btns.addWidget(b)
        pt_btns.addStretch()
        self._add_pt_btn.clicked.connect(self._add_point)
        self._rem_pt_btn.clicked.connect(self._remove_point)
        cvl.addLayout(pt_btns)

        vbl.addWidget(self._custom_widget)
        root.addWidget(box)

        # wire shape change after all widgets exist
        self._shape_combo.currentTextChanged.connect(self._on_shape_changed)
        self._on_shape_changed(_FOR_SHAPES[0])

    # ── internal ─────────────────────────────────────────────────────────────

    def _on_value_changed(self) -> None:
        if not self._blocked:
            self.changed.emit()

    def _on_shape_changed(self, shape: str) -> None:
        show_bs   = shape != "Open"
        show_dims = shape in ("Rectangular", "Elliptical")
        show_cust = shape == "Custom"

        for row_w in (self._bs_az_row, self._bs_el_row):
            row_w.setVisible(show_bs)
            lbl = self._for_form.labelForField(row_w)
            if lbl:
                lbl.setVisible(show_bs)

        for row_w in (self._width_row, self._height_row):
            row_w.setVisible(show_dims)
            lbl = self._for_form.labelForField(row_w)
            if lbl:
                lbl.setVisible(show_dims)

        self._custom_widget.setVisible(show_cust)
        if not self._blocked:
            self.changed.emit()

    def _add_point(self) -> None:
        r = self._pt_table.rowCount()
        self._pt_table.insertRow(r)
        self._pt_table.setItem(r, 0, QTableWidgetItem("0.0"))
        self._pt_table.setItem(r, 1, QTableWidgetItem("0.0"))

    def _remove_point(self) -> None:
        rows = sorted({idx.row() for idx in self._pt_table.selectedItems()},
                      reverse=True)
        if not rows:
            rc = self._pt_table.rowCount()
            if rc > 0:
                rows = [rc - 1]
        for r in rows:
            self._pt_table.removeRow(r)
        self._on_value_changed()

    def _get_custom_points(self) -> list[list[float]]:
        pts: list[list[float]] = []
        for r in range(self._pt_table.rowCount()):
            try:
                az = float((self._pt_table.item(r, 0) or QTableWidgetItem("0")).text())
                el = float((self._pt_table.item(r, 1) or QTableWidgetItem("0")).text())
                pts.append([az, el])
            except ValueError:
                pass
        return pts

    # ── public API ────────────────────────────────────────────────────────────

    def get_for(self) -> dict:
        return {
            "for_shape":         self._shape_combo.currentText(),
            "for_boresight_az":  self._bs_az.value(),
            "for_boresight_el":  self._bs_el.value(),
            "for_width_deg":     self._width.value(),
            "for_height_deg":    self._height.value(),
            "for_custom_points": self._get_custom_points(),
        }

    def set_for(self, sensor: dict) -> None:
        self._blocked = True
        shape = sensor.get("for_shape", "Open")
        if shape not in _FOR_SHAPES:
            shape = "Open"
        self._shape_combo.setCurrentText(shape)
        self._bs_az.setValue(float(sensor.get("for_boresight_az",  0.0)))
        self._bs_el.setValue(float(sensor.get("for_boresight_el", 90.0)))
        self._width.setValue(float(sensor.get("for_width_deg",    10.0)))
        self._height.setValue(float(sensor.get("for_height_deg",  10.0)))
        pts = sensor.get("for_custom_points", [])
        self._pt_table.blockSignals(True)
        self._pt_table.setRowCount(0)
        for az, el in (pts if isinstance(pts, list) else []):
            r = self._pt_table.rowCount()
            self._pt_table.insertRow(r)
            self._pt_table.setItem(r, 0, QTableWidgetItem(str(az)))
            self._pt_table.setItem(r, 1, QTableWidgetItem(str(el)))
        self._pt_table.blockSignals(False)
        self._blocked = False
        self._on_shape_changed(shape)


# ─── typed sensor editor (one tab per sensor type) ───────────────────────────

class TypedSensorEditor(QWidget):
    """
    Parameter editor with one tab per sensor type.
    Switching tabs changes the active sensor type.
    """

    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._blocked = False
        self._tab_spinboxes: dict[str, dict[str, QDoubleSpinBox]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._type_tabs = QTabWidget()

        for type_name, type_def in _SENSOR_TYPE_DEFS.items():
            spinboxes: dict[str, QDoubleSpinBox] = {}
            box = QGroupBox(f"{type_name} Parameters")
            vl  = QVBoxLayout(box)
            vl.setContentsMargins(6, 4, 6, 4)
            vl.addWidget(_make_param_form(
                type_def["params"], spinboxes, self._on_value_changed))

            inner = QWidget()
            il = QVBoxLayout(inner)
            il.setContentsMargins(4, 4, 4, 4)
            il.addWidget(box)
            il.addStretch()

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(inner)

            self._type_tabs.addTab(scroll, type_name)
            self._tab_spinboxes[type_name] = spinboxes

        self._type_tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._type_tabs)

        # Field-of-regard editor below the type tabs
        self._for_editor = FieldOfRegardEditor()
        self._for_editor.changed.connect(self._on_value_changed)
        layout.addWidget(self._for_editor)

    def _on_value_changed(self) -> None:
        if not self._blocked:
            self.changed.emit()

    def _on_tab_changed(self, _: int) -> None:
        if not self._blocked:
            self.changed.emit()

    def get_sensor_type(self) -> str:
        return self._type_tabs.tabText(self._type_tabs.currentIndex())

    def get_params(self) -> dict:
        stype = self.get_sensor_type()
        params = {k: sb.value() for k, sb in self._tab_spinboxes[stype].items()}
        params.update(self._for_editor.get_for())
        return params

    def set_sensor(self, sensor: dict) -> None:
        self._blocked = True
        stype = sensor.get("sensor_type", _SENSOR_TYPE_NAMES[0])
        for i in range(self._type_tabs.count()):
            if self._type_tabs.tabText(i) == stype:
                self._type_tabs.setCurrentIndex(i)
                break
        for k, sb in self._tab_spinboxes[stype].items():
            if k in sensor:
                sb.setValue(float(sensor[k]))
        self._blocked = False
        self._for_editor.set_for(sensor)


# ─── sensors tab ─────────────────────────────────────────────────────────────


class SensorsTab(QWidget):
    """
    Sensor management tab.

    Left pane  — QTableWidget overview (all sensors, key properties).
    Right pane — name field + full parameter editor for selected sensor.
    Add / Duplicate / Remove buttons above the table.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sensors: list[dict] = [_default_sensor("Sensor 1")]
        self._current: int = 0
        self._quiet: bool  = False          # suppress recursive updates

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # ── left pane ────────────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(480)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 2, 4)
        lv.setSpacing(4)

        # Action buttons
        btn_row = QHBoxLayout()
        self._add_btn    = QPushButton("+ Add")
        self._dup_btn    = QPushButton("⎘ Duplicate")
        self._del_btn    = QPushButton("− Remove")
        self._export_btn = QPushButton("↑ Export CSV")
        self._import_btn = QPushButton("↓ Import CSV")
        for b in (self._add_btn, self._dup_btn, self._del_btn,
                  self._export_btn, self._import_btn):
            b.setStyleSheet(_BTN_STYLE)
            b.setFixedHeight(26)
            btn_row.addWidget(b)
        btn_row.addStretch()

        self._add_btn.clicked.connect(self._add_sensor)
        self._dup_btn.clicked.connect(self._dup_sensor)
        self._del_btn.clicked.connect(self._del_sensor)
        self._export_btn.clicked.connect(self._export_csv)
        self._import_btn.clicked.connect(self._import_csv)

        # Table
        n_cols = len(_SENSOR_TABLE_COLS)
        self._table = QTableWidget(0, n_cols)
        self._table.setHorizontalHeaderLabels([c[1] for c in _SENSOR_TABLE_COLS])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, n_cols):
            self._table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.currentCellChanged.connect(
            lambda cur_row, *_: self._on_row_changed(cur_row))

        lv.addLayout(btn_row)
        lv.addWidget(self._table)

        # ── right pane ───────────────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(280)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(2, 4, 4, 4)
        rv.setSpacing(6)

        # Sensor name field
        name_box = QGroupBox("Sensor Identity")
        name_form = QFormLayout(name_box)
        name_form.setSpacing(4)
        name_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Sensor name…")
        self._name_edit.textEdited.connect(self._on_name_edited)
        name_form.addRow("Name:", self._name_edit)

        self._color_lbl = QLabel()
        self._color_lbl.setFixedHeight(16)
        self._color_lbl.setToolTip("Plot colour assigned to this sensor")
        name_form.addRow("Plot colour:", self._color_lbl)

        rv.addWidget(name_box)

        # Parameter editor
        self._editor = TypedSensorEditor()
        self._editor.changed.connect(self._on_editor_changed)
        rv.addWidget(self._editor)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(splitter)

        # Populate initial state
        self._refresh_table()
        self._select(0)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _color_for(self, idx: int) -> str:
        return _SENSOR_COLORS[idx % len(_SENSOR_COLORS)]

    def _refresh_table(self) -> None:
        """Rebuild all table rows from self._sensors (signals blocked)."""
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._sensors))
        for row, sensor in enumerate(self._sensors):
            self._update_table_row(row, sensor)
        self._table.blockSignals(False)

    def _select(self, idx: int) -> None:
        """Select a sensor row and refresh the editor — bypasses signal chain."""
        self._current = max(0, min(idx, len(self._sensors) - 1))
        # Visually select the row without triggering _on_row_changed
        self._quiet = True
        self._table.selectRow(self._current)
        self._quiet = False
        self._load_sensor_into_editor(self._current)
        self._del_btn.setEnabled(len(self._sensors) > 1)

    def _update_table_row(self, row: int, sensor: dict) -> None:
        for col, (key, _) in enumerate(_SENSOR_TABLE_COLS):
            raw = sensor.get(key)
            if raw is None:
                text = "—"
            elif key in ("name", "sensor_type"):
                text = str(raw)
            else:
                text = f"{float(raw):.3g}"
            item = QTableWidgetItem(text)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
                if key == "name"
                else Qt.AlignmentFlag.AlignCenter
            )
            if key == "name":
                color = self._color_for(row)
                item.setForeground(QColor(color))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._table.setItem(row, col, item)

    def _load_sensor_into_editor(self, idx: int) -> None:
        sensor = self._sensors[idx]
        self._quiet = True
        self._name_edit.setText(sensor["name"])
        self._editor.set_sensor(sensor)
        color = self._color_for(idx)
        self._color_lbl.setStyleSheet(
            f"background:{color}; border-radius:3px; border:1px solid #999;")
        self._quiet = False

    # ── slots ────────────────────────────────────────────────────────────────

    def _on_row_changed(self, row: int) -> None:
        if self._quiet or row < 0 or row >= len(self._sensors):
            return
        self._select(row)

    def _on_name_edited(self, text: str) -> None:
        if self._quiet or self._current >= len(self._sensors):
            return
        self._sensors[self._current]["name"] = text
        self._update_table_row(self._current, self._sensors[self._current])

    def _on_editor_changed(self) -> None:
        if self._quiet or self._current >= len(self._sensors):
            return
        sensor = self._sensors[self._current]
        sensor["sensor_type"] = self._editor.get_sensor_type()
        sensor.update(self._editor.get_params())
        self._update_table_row(self._current, sensor)

    def _add_sensor(self) -> None:
        name = f"Sensor {len(self._sensors) + 1}"
        self._sensors.append(_default_sensor(name))
        self._refresh_table()
        self._select(len(self._sensors) - 1)

    def _dup_sensor(self) -> None:
        src  = self._sensors[self._current]
        copy = dict(src)
        copy["name"] = src["name"] + " (copy)"
        self._sensors.append(copy)
        self._refresh_table()
        self._select(len(self._sensors) - 1)

    def _del_sensor(self) -> None:
        if len(self._sensors) <= 1:
            return
        self._sensors.pop(self._current)
        self._refresh_table()
        self._select(min(self._current, len(self._sensors) - 1))

    # ── CSV import / export ──────────────────────────────────────────────────

    # Union of all param keys across every sensor type (deduped, order preserved)
    _CSV_KEYS = ["name", "sensor_type"] + list(dict.fromkeys(
        key
        for td in _SENSOR_TYPE_DEFS.values()
        for key, *_ in td["params"]
    )) + ["for_shape", "for_boresight_az", "for_boresight_el",
           "for_width_deg", "for_height_deg", "for_custom_points"]

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export sensors", "sensors.csv",
            "CSV files (*.csv);;All files (*)")
        if not path:
            return
        import csv, json
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._CSV_KEYS,
                                        extrasaction="ignore")
                writer.writeheader()
                # Serialize for_custom_points list as JSON string for the cell
                rows = []
                for s in self._sensors:
                    row = dict(s)
                    row["for_custom_points"] = json.dumps(
                        s.get("for_custom_points", []))
                    rows.append(row)
                writer.writerows(rows)
            print(f"Exported {len(self._sensors)} sensor(s) → {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _import_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import sensors", "",
            "CSV files (*.csv);;All files (*)")
        if not path:
            return
        import csv, json
        loaded: list[dict] = []
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    stype  = row.get("sensor_type", _SENSOR_TYPE_NAMES[0])
                    if stype not in _SENSOR_TYPE_DEFS:
                        stype = _SENSOR_TYPE_NAMES[0]
                    sensor = _default_sensor(
                        row.get("name", f"Sensor {len(loaded)+1}"), stype)
                    for key, *_ in _SENSOR_TYPE_DEFS[stype]["params"]:
                        if key in row:
                            try:
                                sensor[key] = float(row[key])
                            except ValueError:
                                pass
                    # FoR scalar fields
                    for key in ("for_shape",):
                        if row.get(key):
                            sensor[key] = row[key]
                    for key in ("for_boresight_az", "for_boresight_el",
                                "for_width_deg", "for_height_deg"):
                        if row.get(key):
                            try:
                                sensor[key] = float(row[key])
                            except ValueError:
                                pass
                    # FoR custom points (JSON list)
                    raw_pts = row.get("for_custom_points", "")
                    if raw_pts:
                        try:
                            sensor["for_custom_points"] = json.loads(raw_pts)
                        except (ValueError, TypeError):
                            pass
                    loaded.append(sensor)
        except OSError as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return

        if not loaded:
            QMessageBox.warning(self, "Import", "No sensor rows found in file.")
            return

        reply = QMessageBox.question(
            self, "Import sensors",
            f"Replace current sensors with {len(loaded)} imported sensor(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._sensors = loaded
        self._refresh_table()
        self._select(0)
        print(f"Imported {len(loaded)} sensor(s) from {path}")

    # ── public API ───────────────────────────────────────────────────────────

    def get_sensors(self) -> list[dict]:
        """Return a snapshot of all sensor dicts (deep-ish copy)."""
        return [dict(s) for s in self._sensors]


# ─── results panel ────────────────────────────────────────────────────────────

_PASS_COLOR = "#2196F3"

# (key, tab label, enabled by default)
_PLOT_DEFS: list[tuple[str, str, bool]] = [
    ("elevation",   "Elevation",         True),
    ("range",       "Slant Range",       True),
    ("snr",         "SNR",               True),
    ("vmag",        "Visual Magnitude",  True),
    ("groundtrack", "Ground Track",      True),
]


class _PlotCanvas(QWidget):
    """One matplotlib figure + navigation toolbar in a widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fig    = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def clear_placeholder(self) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Run the analysis to see results",
                ha="center", va="center", fontsize=13, color="#aaa",
                transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw()

    def redraw(self) -> None:
        self.canvas.draw()


class ResultsPanel(QWidget):
    """
    Results tab with a checkbox row at the top and a QTabWidget below.
    Each checked plot gets its own tab with a dedicated matplotlib canvas.
    Checking / unchecking a plot immediately adds or removes its tab.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── checkbox strip ───────────────────────────────────────────────────
        cb_box = QGroupBox("Plots to show")
        cb_box.setMaximumHeight(54)
        cb_layout = QHBoxLayout(cb_box)
        cb_layout.setContentsMargins(8, 2, 8, 2)
        self._checkboxes: dict[str, QCheckBox] = {}
        for key, label, default in _PLOT_DEFS:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.toggled.connect(lambda checked, k=key: self._on_toggle(k, checked))
            cb_layout.addWidget(cb)
            self._checkboxes[key] = cb
        cb_layout.addStretch()
        root.addWidget(cb_box)

        # ── plot tab widget ──────────────────────────────────────────────────
        self._plot_tabs = QTabWidget()
        root.addWidget(self._plot_tabs)

        # Create one canvas per plot (they may not all be in the tab widget)
        self._canvases: dict[str, _PlotCanvas] = {
            key: _PlotCanvas() for key, *_ in _PLOT_DEFS
        }
        for key, _, default in _PLOT_DEFS:
            self._canvases[key].clear_placeholder()
            if default:
                label = next(l for k, l, _ in _PLOT_DEFS if k == key)
                self._plot_tabs.addTab(self._canvases[key], label)

        self._results: dict | None = None

    # ── checkbox toggle ──────────────────────────────────────────────────────

    def _on_toggle(self, key: str, checked: bool) -> None:
        canvas = self._canvases[key]
        label  = next(l for k, l, _ in _PLOT_DEFS if k == key)
        if checked:
            # Insert at the position that preserves _PLOT_DEFS order
            key_order = [k for k, *_ in _PLOT_DEFS]
            insert_at = sum(
                1 for k in key_order[:key_order.index(key)]
                if self._checkboxes[k].isChecked()
            )
            self._plot_tabs.insertTab(insert_at, canvas, label)
            if self._results:
                self._draw_one(key, self._results)
        else:
            idx = self._plot_tabs.indexOf(canvas)
            if idx >= 0:
                self._plot_tabs.removeTab(idx)

    # ── public update ────────────────────────────────────────────────────────

    def update(self, results: dict) -> None:  # type: ignore[override]
        self._results = results
        for key in self._checkboxes:
            if self._checkboxes[key].isChecked():
                self._draw_one(key, results)

    # ── per-plot draw methods ────────────────────────────────────────────────

    def _draw_one(self, key: str, r: dict) -> None:
        canvas = self._canvases[key]
        canvas.fig.clear()
        getattr(self, f"_draw_{key}")(canvas.fig, r)
        canvas.redraw()

    def _draw_elevation(self, fig: Figure, r: dict) -> None:
        sp  = r["scene_params"]
        t   = r["times_min"]
        el  = r["elevations"]
        ax  = fig.add_subplot(111)
        ax.plot(t, el, color="#555", lw=1, label="Elevation")
        ax.axhline(sp["min_el_deg"], color=_PASS_COLOR, ls="--", lw=1,
                   label=f"Min el. ({sp['min_el_deg']:.0f}°)")
        _shade_visible(ax, t, el >= sp["min_el_deg"], color=_PASS_COLOR, alpha=0.12)
        n_passes = _count_passes(el >= sp["min_el_deg"])
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Elevation (deg)")
        ax.set_title(f"Elevation angle  —  {n_passes} pass(es) above {sp['min_el_deg']:.0f}°")
        ax.legend(fontsize=9)
        ax.set_ylim(-90, 90); ax.grid(True, alpha=0.3)

    def _draw_range(self, fig: Figure, r: dict) -> None:
        ax = fig.add_subplot(111)
        ax.plot(r["times_min"], r["ranges"], color=_PASS_COLOR, lw=1.2)
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Range (km)")
        ax.set_title("Slant range"); ax.grid(True, alpha=0.3)

    def _draw_snr(self, fig: Figure, r: dict) -> None:
        t   = r["times_min"]
        ax  = fig.add_subplot(111)
        for s_idx, s in enumerate(r["sensors"]):
            color = _SENSOR_COLORS[s_idx % len(_SENSOR_COLORS)]
            snr_v = np.where(np.isnan(s["snrs"]), 0.0, s["snrs"])
            ax.semilogy(t, np.maximum(snr_v, 1e-3), color=color, lw=1.3,
                        label=s["name"])
            _shade_visible(ax, t, s["visible"], color=color, alpha=0.10)
        if r["sensors"]:
            thr = r["sensors"][0]["params"]["snr_threshold"]
            ax.axhline(thr, color="red", ls="--", lw=1, label=f"Threshold ({thr:.1f})")
        ax.set_xlabel("Time (min)"); ax.set_ylabel("SNR")
        ax.set_title("Signal-to-noise ratio")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")

    def _draw_vmag(self, fig: Figure, r: dict) -> None:
        t  = r["times_min"]
        ax = fig.add_subplot(111)
        for s_idx, s in enumerate(r["sensors"]):
            color  = _SENSOR_COLORS[s_idx % len(_SENSOR_COLORS)]
            vmag_v = np.where(np.isnan(s["vmags"]), 30.0, s["vmags"])
            ax.plot(t, vmag_v, color=color, lw=1.3, label=s["name"])
            _shade_visible(ax, t, s["visible"], color=color, alpha=0.10)
        if r["sensors"]:
            lim = r["sensors"][0]["params"]["vizmag_limit"]
            ax.axhline(lim, color="red", ls="--", lw=1, label=f"Limit ({lim:.1f})")
        ax.invert_yaxis()
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Visual magnitude")
        ax.set_title("Visual magnitude (brighter ↑)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    def _draw_groundtrack(self, fig: Figure, r: dict) -> None:
        sp  = r["scene_params"]
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#e8f0fe")
        ax.plot(r["lons_gt"], r["lats"], color="#999", lw=0.7,
                alpha=0.5, zorder=2, label="_nolegend_")
        for s_idx, s in enumerate(r["sensors"]):
            color   = _SENSOR_COLORS[s_idx % len(_SENSOR_COLORS)]
            visible = s["visible"]
            if visible.any():
                ax.scatter(r["lons_gt"][visible], r["lats"][visible],
                           c=color, s=8, zorder=4, alpha=0.8,
                           label=f"{s['name']} detected")
        ax.scatter([sp["obs_lon_deg"]], [sp["obs_lat_deg"]],
                   marker="^", s=90, c="red", zorder=5, label="Observer")
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")
        ax.set_title("Ground track"); ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        _draw_coastlines_approx(ax)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _shade_visible(ax, t: np.ndarray, mask: np.ndarray,
                   color: str, alpha: float) -> None:
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
    return int(np.sum(np.diff(above.astype(int)) == 1))


def _draw_coastlines_approx(ax) -> None:
    for lon0, lat0, lon1, lat1 in [
        (-125, 25, -65, 50), (-80, -55, -35, 10), (-10, 35, 40, 70),
        (25, -35, 52, 38), (60, 10, 140, 55), (113, -45, 155, -10),
    ]:
        ax.plot([lon0, lon1, lon1, lon0, lon0],
                [lat0, lat0, lat1, lat1, lat0],
                color="#bbb", lw=0.5, zorder=1)


# ─── target sweep ────────────────────────────────────────────────────────────

def _parse_float_list(
    text: str, lo: float, hi: float, name: str = "value"
) -> list[float]:
    """Parse a comma-separated string into validated floats."""
    out: list[float] = []
    for part in text.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            v = float(s)
        except ValueError:
            raise ValueError(f"Cannot parse {s!r} as a number in {name}")
        if not (lo <= v <= hi):
            raise ValueError(
                f"{name} value {v} is outside allowed range [{lo}, {hi}]"
            )
        out.append(v)
    if not out:
        raise ValueError(f"No {name} values provided")
    return out


def _sweep_combo_task(args: dict) -> dict:
    """
    Top-level picklable function executed by ProcessPoolExecutor workers.
    Computes SNR/vmag/visibility for one (diam, alb) combo across all sensors.
    args keys: diam, alb, sensors, above_el, illuminated, ranges, n_steps
    """
    diam        = args["diam"]
    alb         = args["alb"]
    sensors     = args["sensors"]
    above_el    = args["above_el"]
    illuminated = args["illuminated"]
    ranges      = args["ranges"]
    n_steps     = args["n_steps"]

    sensor_data: list[dict] = []
    for sensor in sensors:
        snrs    = np.full(n_steps, np.nan)
        vmags   = np.full(n_steps, np.nan)
        visible = np.zeros(n_steps, dtype=bool)
        is_radar = sensor.get("sensor_type") == "Radar"
        active   = above_el if is_radar else (above_el & illuminated)
        for idx in np.where(active)[0]:
            rng = float(ranges[idx])
            if is_radar:
                snr = _compute_snr_radar(
                    rng, diam,
                    sensor["freq_ghz"], sensor["power_kw"],
                    sensor["ant_gain_db"], sensor["pulse_us"],
                    sensor["noise_fig_db"], sensor["n_pulses"],
                    sensor["losses_db"],
                )
                snrs[idx]    = snr
                visible[idx] = snr >= sensor["snr_threshold"]
            else:
                snr = _compute_snr(rng, diam, alb,
                                   sensor["aperture_m"], sensor["exposure_s"],
                                   sensor["qe"], sensor["read_noise_e"],
                                   sensor["loop_gain_db"])
                vm  = _compute_vizmag(rng, diam, alb)
                snrs[idx]    = snr
                vmags[idx]   = vm
                visible[idx] = (snr >= sensor["snr_threshold"]
                                and vm <= sensor.get("vizmag_limit", 99.0))
        sensor_data.append(dict(name=sensor["name"],
                                snrs=snrs, vmags=vmags, visible=visible))

    return dict(
        label   = f"D={diam:.3g} m, alb={alb:.3g}",
        diam    = diam,
        albedo  = alb,
        sensors = sensor_data,
    )


class TargetSweepWorker(QObject):
    """Sweeps over (diameter × albedo) cross-product for all sensors."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(
        self,
        scene_params: dict,
        sensors: list[dict],
        diameters: list[float],
        albedos: list[float],
        n_workers: int = 1,
    ) -> None:
        super().__init__()
        self.scene_params = scene_params
        self.sensors      = sensors
        self.diameters    = diameters
        self.albedos      = albedos
        self.n_workers    = max(1, n_workers)
        self._cancelled   = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            self.finished.emit(self._analyse())
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def _analyse(self) -> dict:
        p = self.scene_params
        self.progress.emit(0, "Sweep: propagating orbit…")

        a_km = _R_E_KM + p["alt_km"]
        e    = p["ecc"]
        i    = math.radians(p["incl_deg"])
        raan = math.radians(p["raan_deg"])
        argp = math.radians(p["argp_deg"])
        M0   = math.radians(p["m0_deg"])
        n_mm = math.sqrt(_MU_KM3_S2 / a_km ** 3)

        n_steps = int(p["duration_min"] * 60.0 / p["step_sec"]) + 1
        times   = np.arange(n_steps) * p["step_sec"]

        elevations  = np.empty(n_steps)
        ranges      = np.empty(n_steps)
        illuminated = np.empty(n_steps, dtype=bool)

        for idx, t in enumerate(times):
            if self._cancelled:
                break
            M       = M0 + n_mm * t
            sat_eci = _keplerian_pos_eci(a_km, e, i, raan, argp, M)
            obs_eci = _lla_to_eci(p["obs_lat_deg"], p["obs_lon_deg"], p["obs_alt_m"], t)
            el, rng = _elevation_range(obs_eci, sat_eci,
                                        p["obs_lat_deg"], p["obs_lon_deg"], t)
            elevations[idx]  = el
            ranges[idx]      = rng
            illuminated[idx] = _is_illuminated(sat_eci, _sun_dir_eci(t))
            if idx % max(1, n_steps // 40) == 0:
                self.progress.emit(int(idx / n_steps * 35),
                                   f"Sweep: orbit {idx}/{n_steps}")

        above_el    = elevations >= p["min_el_deg"]

        # Cross-product of diameters × albedos
        import itertools
        combos   = list(itertools.product(self.diameters, self.albedos))
        n_combos = len(combos)

        # Build per-combo task argument dicts (shared arrays are read-only)
        common = dict(
            sensors     = self.sensors,
            above_el    = above_el,
            illuminated = illuminated,
            ranges      = ranges,
            n_steps     = n_steps,
        )
        task_args = [{**common, "diam": d, "alb": a} for d, a in combos]

        combo_results: list[dict | None] = [None] * n_combos

        if self.n_workers <= 1:
            for c_idx, args in enumerate(task_args):
                if self._cancelled:
                    break
                pct = 36 + int(c_idx / n_combos * 60)
                self.progress.emit(pct,
                    f"Sweep: D={args['diam']:.3g} m  alb={args['alb']:.3g}"
                    f"  ({c_idx + 1}/{n_combos})")
                combo_results[c_idx] = _sweep_combo_task(args)
        else:
            self.progress.emit(36, f"Sweep: launching {self.n_workers} workers…")
            try:
                with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                    fut_map = {pool.submit(_sweep_combo_task, a): i
                               for i, a in enumerate(task_args)}
                    done = 0
                    for fut in as_completed(fut_map):
                        if self._cancelled:
                            pool.shutdown(wait=False, cancel_futures=True)
                            break
                        c_idx = fut_map[fut]
                        combo_results[c_idx] = fut.result()
                        done += 1
                        pct = 36 + int(done / n_combos * 60)
                        self.progress.emit(pct,
                            f"Sweep: {done}/{n_combos} combos complete")
            except Exception as exc:
                # Workers failed (e.g. package not installed in sub-process):
                # fall back to sequential
                print(f"Parallel sweep failed ({exc}); retrying sequentially…")
                for c_idx, args in enumerate(task_args):
                    if self._cancelled:
                        break
                    combo_results[c_idx] = _sweep_combo_task(args)

        self.progress.emit(98, "Sweep: collating…")
        return dict(
            times_min    = times / 60.0,
            elevations   = elevations,
            ranges       = ranges,
            scene_params = p,
            sensor_names = [s["name"] for s in self.sensors],
            combos       = [r for r in combo_results if r is not None],
        )


# ── sweep panel ───────────────────────────────────────────────────────────────

_SWEEP_COLORS = [
    "#E53935", "#8E24AA", "#1E88E5", "#00ACC1",
    "#43A047", "#FB8C00", "#6D4C41", "#546E7A",
    "#F06292", "#AED581", "#FFD54F", "#80DEEA",
]


class TargetSweepPanel(QWidget):
    """
    Parametric analysis over multiple target diameters and/or albedos.
    Enter comma-separated values; the analysis runs the cross-product of
    diameters × albedos for every sensor defined in the Sensors tab.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: TargetSweepWorker | None = None
        self._thread: QThread | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── input group ──────────────────────────────────────────────────────
        inp = QGroupBox("Target properties  (comma-separated values)")
        form = QFormLayout(inp)
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._diam_edit = QLineEdit("0.5, 1.0, 2.0, 5.0")
        self._diam_edit.setToolTip(
            "Target cross-section diameters in metres, e.g.  0.1, 0.5, 1, 2, 5")
        self._diam_edit.setPlaceholderText("e.g.  0.1, 0.5, 1.0, 2.0")

        self._alb_edit = QLineEdit("0.3")
        self._alb_edit.setToolTip(
            "Albedo values (0–1).  A single value is applied to all diameters; "
            "multiple values are crossed with all diameters.")
        self._alb_edit.setPlaceholderText("e.g.  0.1, 0.3, 0.5")

        # Workers slider
        _n_cpu     = os.cpu_count() or 4
        _max_work  = max(1, _n_cpu - 2)
        self._workers_slider = QSlider(Qt.Orientation.Horizontal)
        self._workers_slider.setRange(1, _max_work)
        self._workers_slider.setValue(1)
        self._workers_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._workers_slider.setTickInterval(1)
        self._workers_slider.setSingleStep(1)
        self._workers_slider.setFixedWidth(120)
        self._workers_val_lbl = QLabel("1")
        self._workers_val_lbl.setFixedWidth(22)
        self._workers_val_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._workers_slider.valueChanged.connect(
            lambda v: self._workers_val_lbl.setText(str(v)))
        workers_row = QWidget()
        wrl = QHBoxLayout(workers_row)
        wrl.setContentsMargins(0, 0, 0, 0)
        wrl.addWidget(self._workers_slider)
        wrl.addWidget(self._workers_val_lbl)
        wrl.addStretch()
        form.addRow("Diameters (m):", self._diam_edit)
        form.addRow("Albedos:", self._alb_edit)
        form.addRow(
            f"Parallel workers (1–{_max_work}):", workers_row)
        root.addWidget(inp)

        # ── run bar ──────────────────────────────────────────────────────────
        run_bar = QHBoxLayout()
        self._run_btn = QPushButton("▶  Run Sweep")
        self._run_btn.setFixedHeight(30)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#388E3C; color:white; font-weight:bold;"
            "  border-radius:4px; padding:0 16px; }"
            "QPushButton:hover   { background:#2E7D32; }"
            "QPushButton:pressed { background:#1B5E20; }"
            "QPushButton:disabled{ background:#aaa; }"
        )
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(30)
        self._cancel_btn.setEnabled(False)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setFixedHeight(22)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._run_btn.clicked.connect(self._on_run_clicked)
        self._cancel_btn.clicked.connect(self._on_cancel)

        run_bar.addWidget(self._run_btn)
        run_bar.addWidget(self._cancel_btn)
        run_bar.addWidget(self._progress, stretch=1)
        run_bar.addWidget(self._status_lbl)
        root.addLayout(run_bar)

        # ── result plots ─────────────────────────────────────────────────────
        self._plot_tabs = QTabWidget()
        self._cv_snr_time  = _PlotCanvas()
        self._cv_peak_snr  = _PlotCanvas()
        self._cv_det_pct   = _PlotCanvas()
        self._cv_vmag_time = _PlotCanvas()
        self._plot_tabs.addTab(self._cv_snr_time,  "SNR vs Time")
        self._plot_tabs.addTab(self._cv_peak_snr,  "Peak SNR vs Diameter")
        self._plot_tabs.addTab(self._cv_det_pct,   "Detection %")
        self._plot_tabs.addTab(self._cv_vmag_time, "Vmag vs Time")
        for cv in (self._cv_snr_time, self._cv_peak_snr,
                   self._cv_det_pct, self._cv_vmag_time):
            cv.clear_placeholder()
        root.addWidget(self._plot_tabs)

        # Stored so MainWindow can inject scene/sensor data before running
        self._scene_params: dict | None = None
        self._sensors: list[dict] | None = None

    # ── called by MainWindow to inject current scene + sensor data ────────────

    def set_context(self, scene_params: dict, sensors: list[dict]) -> None:
        self._scene_params = scene_params
        self._sensors      = sensors

    # ── internal ─────────────────────────────────────────────────────────────

    def _on_run_clicked(self) -> None:
        if self._thread and self._thread.isRunning():
            return
        if not self._scene_params or not self._sensors:
            QMessageBox.warning(self, "No context",
                                "Click '▶ Run Analysis' first to compute the "
                                "base orbit, then run the sweep.")
            return
        try:
            diameters = _parse_float_list(
                self._diam_edit.text(), 0.001, 1000.0, "diameter")
            albedos   = _parse_float_list(
                self._alb_edit.text(), 0.0, 1.0, "albedo")
        except ValueError as exc:
            QMessageBox.warning(self, "Bad input", str(exc))
            return

        self._worker = TargetSweepWorker(
            self._scene_params, self._sensors, diameters, albedos,
            n_workers=self._workers_slider.value())
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
        self._status_lbl.setText("Running sweep…")
        self._thread.start()

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._status_lbl.setText("Cancelled.")

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self._status_lbl.setText(msg)

    def _on_finished(self, results: dict) -> None:
        self._progress.setValue(100)
        self._status_lbl.setText(
            f"Sweep done — {len(results['combos'])} combinations × "
            f"{len(results['sensor_names'])} sensor(s).")
        self._draw_results(results)

    def _on_error(self, msg: str) -> None:
        self._progress.setValue(0)
        self._status_lbl.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Sweep error", msg)

    def _on_thread_done(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None
        self._thread = None

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_results(self, r: dict) -> None:
        combos       = r["combos"]
        sensor_names = r["sensor_names"]
        times        = r["times_min"]
        n_sensors    = len(sensor_names)

        albedos_unique   = sorted({c["albedo"]  for c in combos})

        # ── SNR vs Time ──────────────────────────────────────────────────────
        fig = self._cv_snr_time.fig
        fig.clear()
        axes = fig.subplots(1, n_sensors, squeeze=False)[0] if n_sensors else []
        for s_idx, s_name in enumerate(sensor_names):
            ax = axes[s_idx]
            for c_idx, combo in enumerate(combos):
                color = _SWEEP_COLORS[c_idx % len(_SWEEP_COLORS)]
                snr_v = np.where(np.isnan(combo["sensors"][s_idx]["snrs"]),
                                 0.0, combo["sensors"][s_idx]["snrs"])
                ax.semilogy(times, np.maximum(snr_v, 1e-3),
                            color=color, lw=1.1, label=combo["label"])
            ax.set_title(s_name, fontsize=9)
            ax.set_xlabel("Time (min)"); ax.set_ylabel("SNR")
            ax.grid(True, alpha=0.3, which="both")
            if s_idx == 0:
                ax.legend(fontsize=7, loc="upper right")
        fig.suptitle("SNR vs Time — target sweep", fontsize=9)
        self._cv_snr_time.redraw()

        # ── Vmag vs Time ─────────────────────────────────────────────────────
        fig = self._cv_vmag_time.fig
        fig.clear()
        axes = fig.subplots(1, n_sensors, squeeze=False)[0] if n_sensors else []
        for s_idx, s_name in enumerate(sensor_names):
            ax = axes[s_idx]
            for c_idx, combo in enumerate(combos):
                color  = _SWEEP_COLORS[c_idx % len(_SWEEP_COLORS)]
                vmag_v = np.where(np.isnan(combo["sensors"][s_idx]["vmags"]),
                                  30.0, combo["sensors"][s_idx]["vmags"])
                ax.plot(times, vmag_v, color=color, lw=1.1, label=combo["label"])
            ax.invert_yaxis()
            ax.set_title(s_name, fontsize=9)
            ax.set_xlabel("Time (min)"); ax.set_ylabel("Visual magnitude")
            ax.grid(True, alpha=0.3)
            if s_idx == 0:
                ax.legend(fontsize=7, loc="upper right")
        fig.suptitle("Visual magnitude vs Time — target sweep", fontsize=9)
        self._cv_vmag_time.redraw()

        # ── Peak SNR vs Diameter ─────────────────────────────────────────────
        fig = self._cv_peak_snr.fig
        fig.clear()
        ax  = fig.add_subplot(111)
        for s_idx, s_name in enumerate(sensor_names):
            color = _SENSOR_COLORS[s_idx % len(_SENSOR_COLORS)]
            for alb in albedos_unique:
                xs, ys = [], []
                for combo in combos:
                    if combo["albedo"] != alb:
                        continue
                    snrs = combo["sensors"][s_idx]["snrs"]
                    peak = float(np.nanmax(snrs)) if not np.all(np.isnan(snrs)) else 0.0
                    xs.append(combo["diam"])
                    ys.append(peak)
                lbl = f"{s_name} (alb={alb:.2g})"
                ax.loglog(xs, ys, "o-", color=color, lw=1.3, label=lbl)
        ax.set_xlabel("Target diameter (m)")
        ax.set_ylabel("Peak SNR")
        ax.set_title("Peak SNR vs diameter")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
        self._cv_peak_snr.redraw()

        # ── Detection % vs Diameter ──────────────────────────────────────────
        fig = self._cv_det_pct.fig
        fig.clear()
        ax  = fig.add_subplot(111)
        n_t = len(times)
        for s_idx, s_name in enumerate(sensor_names):
            color = _SENSOR_COLORS[s_idx % len(_SENSOR_COLORS)]
            for alb in albedos_unique:
                xs, ys = [], []
                for combo in combos:
                    if combo["albedo"] != alb:
                        continue
                    pct = float(combo["sensors"][s_idx]["visible"].sum()) / n_t * 100
                    xs.append(combo["diam"])
                    ys.append(pct)
                lbl = f"{s_name} (alb={alb:.2g})"
                ax.semilogx(xs, ys, "o-", color=color, lw=1.3, label=lbl)
        ax.set_xlabel("Target diameter (m)")
        ax.set_ylabel("Detection fraction (%)")
        ax.set_title("Detection % vs diameter")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        self._cv_det_pct.redraw()


# ─── stdout redirector + console widget ──────────────────────────────────────

class StdoutRedirector(QObject):
    """Intercepts sys.stdout writes and re-emits them as a Qt signal."""

    text_written = pyqtSignal(str)

    def write(self, text: str) -> None:
        if text:
            self.text_written.emit(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class ConsoleWidget(QWidget):
    """Read-only scrollable console that shows redirected stdout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header row
        header = QHBoxLayout()
        lbl = QLabel("Console output")
        lbl.setStyleSheet("font-size: 11px; color: #555; font-weight: bold;")
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(20)
        clear_btn.setStyleSheet(
            "QPushButton { font-size:11px; padding:0 8px; border:1px solid #ccc;"
            "  border-radius:3px; }"
            "QPushButton:hover { background:#f0f0f0; }"
        )
        header.addWidget(lbl)
        header.addStretch()
        header.addWidget(clear_btn)

        # Text area
        self._edit = QTextEdit()
        self._edit.setReadOnly(True)
        self._edit.setFont(QFont("Consolas", 9))
        self._edit.setStyleSheet(
            "QTextEdit { background:#1e1e1e; color:#d4d4d4;"
            "  border:none; border-top:1px solid #ccc; }"
        )
        self._edit.setMinimumHeight(80)

        clear_btn.clicked.connect(self._edit.clear)

        layout.addLayout(header)
        layout.addWidget(self._edit)

    def append(self, text: str) -> None:
        """Append *text* and auto-scroll to bottom."""
        # Strip trailing newline so we don't double-space via appendPlainText
        self._edit.moveCursor(self._edit.textCursor().MoveOperation.End)
        self._edit.insertPlainText(text)
        self._edit.ensureCursorVisible()


# ─── banner ───────────────────────────────────────────────────────────────────

class BannerWidget(QWidget):
    _BG_LEFT  = QColor("#0D47A1")
    _BG_RIGHT = QColor("#1565C0")
    _ACCENT   = QColor("#42A5F5")

    def __init__(
        self,
        title:    str = "EO Ground Station Analysis",
        subtitle: str = "Keplerian orbit propagation · SNR · Visual magnitude · Multi-sensor",
        version:  str = "v0.2",
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
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0.0, self._BG_LEFT)
        grad.setColorAt(1.0, self._BG_RIGHT)
        p.fillRect(self.rect(), QBrush(grad))
        p.fillRect(0, 0, 5, self.height(), self._ACCENT)
        p.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        p.setPen(QColor("#FFFFFF"))
        p.drawText(18, 0, self.width() - 120, self.height(),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self._title)
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor("#90CAF9"))
        p.drawText(20, 26, self.width() - 130, self.height() - 26,
                   Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                   self._subtitle)
        p.setFont(QFont("Consolas", 8))
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
        self.resize(1260, 820)
        self._worker: AnalysisWorker | None = None
        self._thread: QThread | None = None
        self._stdout_orig = sys.stdout
        self._build_ui()
        # Install stdout redirector after UI exists
        self._redirector = StdoutRedirector()
        self._redirector.text_written.connect(self._console.append)
        sys.stdout = self._redirector  # type: ignore[assignment]

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        root.addWidget(BannerWidget())

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        # Tab 0: Scene/Orbit params + console (vertical splitter)
        self._params_panel = ParameterPanel()
        self._console = ConsoleWidget()
        scene_splitter = QSplitter(Qt.Orientation.Vertical)
        scene_splitter.addWidget(self._params_panel)
        scene_splitter.addWidget(self._console)
        scene_splitter.setStretchFactor(0, 3)
        scene_splitter.setStretchFactor(1, 1)
        scene_splitter.setSizes([520, 160])
        self._tabs.addTab(scene_splitter, "Scene / Orbit")

        self._sensors_tab = SensorsTab()
        self._tabs.addTab(self._sensors_tab, "Sensors")

        self._results_panel = ResultsPanel()
        self._tabs.addTab(self._results_panel, "Results")

        self._sweep_panel = TargetSweepPanel()
        self._tabs.addTab(self._sweep_panel, "Target Sweep")

        # Bottom bar
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

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
            "QPushButton { background:#1976D2; color:white; font-weight:bold;"
            "  border-radius:4px; padding:0 16px; }"
            "QPushButton:hover   { background:#1565C0; }"
            "QPushButton:pressed { background:#0D47A1; }"
            "QPushButton:disabled{ background:#aaa; }"
        )
        self._run_btn.clicked.connect(self._start_analysis)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(30)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_analysis)

        bar.addWidget(self._status_lbl)
        bar.addWidget(self._progress, stretch=1)
        bar.addWidget(self._run_btn)
        bar.addWidget(self._cancel_btn)
        root.addLayout(bar)

    # ── analysis lifecycle ───────────────────────────────────────────────────

    def _start_analysis(self) -> None:
        if self._thread and self._thread.isRunning():
            return

        sensors = self._sensors_tab.get_sensors()
        if not sensors:
            QMessageBox.warning(self, "No sensors", "Add at least one sensor before running.")
            return

        scene_params = self._params_panel.get_params()

        self._worker = AnalysisWorker(scene_params, sensors)
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

    def _on_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self._status_lbl.setText(msg)

    def _on_finished(self, results: dict) -> None:
        self._progress.setValue(100)
        self._status_lbl.setText("Done — updating plots…")
        self._results_panel.update(results)
        self._sweep_panel.set_context(results["scene_params"],
                                      self._sensors_tab.get_sensors())
        self._tabs.setCurrentWidget(self._results_panel)
        n = len(results["sensors"])
        self._status_lbl.setText(
            f"Analysis complete — {n} sensor{'s' if n != 1 else ''}.")

    def _on_error(self, msg: str) -> None:
        self._progress.setValue(0)
        self._status_lbl.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Analysis Error", msg)

    def _on_thread_done(self) -> None:
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None
        self._thread = None

    def closeEvent(self, event) -> None:
        sys.stdout = self._stdout_orig
        self._cancel_analysis()
        super().closeEvent(event)


# ─── entry point ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    # Required for ProcessPoolExecutor on Windows (frozen or script entry points)
    import multiprocessing
    multiprocessing.freeze_support()

    app = QApplication(argv or sys.argv)
    app.setApplicationName("EO Analysis")
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window,        QColor("#f5f5f5"))
    pal.setColor(QPalette.ColorRole.WindowText,    QColor("#212121"))
    pal.setColor(QPalette.ColorRole.Base,          QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#fafafa"))
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parents[3]))
    raise SystemExit(main())

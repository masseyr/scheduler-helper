"""
tasking_helper.viz.globe — Interactive 3-D Earth globe.
========================================================

Renders a WGS-84 Earth sphere with a lat/lon graticule, target grids,
individual ground targets, and optional satellite ground tracks.

Backend: plotly (interactive HTML) with optional matplotlib PNG export.

Usage
-----
    from tasking_helper.viz import Globe3D, Target

    globe = Globe3D(title="Mission targets")

    # Regular grid of surface targets
    globe.add_target_grid(lat_range=(-60, 60), lon_range=(-180, 180), step=15)

    # Custom point targets
    globe.add_targets([
        Target(lat=51.5,  lon=-0.1,   name="London"),
        Target(lat=40.7,  lon=-74.0,  name="New York"),
        Target(lat=35.7,  lon=139.7,  name="Tokyo"),
    ])

    # Satellite ground track
    from tasking_helper.utils.tle import parse_tle
    from tasking_helper.utils.satellite import Satellite
    from tasking_helper.utils.jdate import epoch_to_jd

    tle = parse_tle(name, line1, line2)
    sat = Satellite.from_tle(tle)
    globe.add_ground_track(sat, jd_start=tle.epoch_jd, duration_min=90)

    globe.show()            # opens browser tab
    globe.save("globe.html")
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tasking_helper.utils.satellite import Satellite


# ── Earth geometry ────────────────────────────────────────────────────────────
_R_EARTH_KM = 6_378.137          # WGS-84 semi-major axis [km]
_F_EARTH    = 1.0 / 298.257223563
_B_EARTH_KM = _R_EARTH_KM * (1.0 - _F_EARTH)  # semi-minor axis [km]


def _lla_to_xyz(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> tuple[float, float, float]:
    """Geodetic (deg, deg, km) → Cartesian (km) on WGS-84 ellipsoid."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e2  = 1.0 - (_B_EARTH_KM / _R_EARTH_KM) ** 2
    N   = _R_EARTH_KM / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    r   = N + alt_km
    x   = r * math.cos(lat) * math.cos(lon)
    y   = r * math.cos(lat) * math.sin(lon)
    z   = (N * (1.0 - e2) + alt_km) * math.sin(lat)
    return x, y, z


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Target:
    """A ground target with geographic coordinates and optional metadata.

    Parameters
    ----------
    lat : float — geodetic latitude [deg], positive North
    lon : float — longitude [deg], positive East
    alt : float — altitude above WGS-84 ellipsoid [km] (default 0)
    name : str  — label shown in hover tooltip
    priority : int — 1 (highest) … 10 (lowest); used for colour coding
    """
    lat: float
    lon: float
    alt: float = 0.0
    name: str = ""
    priority: int = 5


@dataclass
class _TrackLayer:
    """Internal: satellite ground track."""
    lats: list[float]
    lons: list[float]
    alts: list[float]
    name: str
    color: str
    dash: str = "solid"


# ── Globe3D ───────────────────────────────────────────────────────────────────

class Globe3D:
    """Interactive 3-D globe for visualising target grids and satellite tracks.

    Parameters
    ----------
    title : str — figure title
    graticule_step : float — lat/lon grid line spacing [deg] (default 30)
    sphere_opacity : float — Earth surface opacity 0–1 (default 0.35)
    dark_theme : bool — use dark background (default True)
    """

    # Priority → hex colour (1=red … 10=blue)
    _PRIORITY_COLORS = {
        1: "#ff1a1a", 2: "#ff6600", 3: "#ff9900", 4: "#ffcc00",
        5: "#ffe066", 6: "#99dd55", 7: "#33cc77", 8: "#00aaff",
        9: "#3366ff", 10: "#9933ff",
    }
    _TRACK_PALETTE = [
        "#00e5ff", "#ff6d00", "#76ff03", "#d500f9",
        "#ffea00", "#ff1744", "#00e676", "#2979ff",
    ]

    def __init__(
        self,
        title: str = "3-D Target Globe",
        graticule_step: float = 30.0,
        sphere_opacity: float = 0.35,
        dark_theme: bool = True,
    ) -> None:
        self.title = title
        self.graticule_step = graticule_step
        self.sphere_opacity = sphere_opacity
        self.dark_theme = dark_theme

        self._target_groups: list[tuple[list[Target], str | None]] = []
        self._track_layers: list[_TrackLayer] = []
        self._satellite_points: list[tuple[float, float, float, str]] = []  # x,y,z,name
        self._track_color_idx = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_targets(
        self,
        targets: list[Target],
        group_name: str | None = None,
    ) -> "Globe3D":
        """Add a list of individual :class:`Target` objects.

        Parameters
        ----------
        targets : list[Target]
        group_name : str | None — legend group label (defaults to "Targets")
        """
        self._target_groups.append((list(targets), group_name))
        return self

    def add_target_grid(
        self,
        lat_range: tuple[float, float] = (-75.0, 75.0),
        lon_range: tuple[float, float] = (-180.0, 180.0),
        step: float = 15.0,
        alt_km: float = 0.0,
        priority: int = 5,
        group_name: str | None = None,
    ) -> "Globe3D":
        """Generate and add a regular lat/lon grid of targets.

        Parameters
        ----------
        lat_range : (lat_min, lat_max) [deg]
        lon_range : (lon_min, lon_max) [deg]
        step : float — grid spacing [deg]
        alt_km : float — surface altitude [km]
        priority : int — priority assigned to all grid points
        group_name : str | None — legend label (default: auto-generated)
        """
        lats = np.arange(lat_range[0], lat_range[1] + 1e-9, step)
        lons = np.arange(lon_range[0], lon_range[1] + 1e-9, step)
        targets = [
            Target(lat=float(la), lon=float(lo), alt=alt_km, priority=priority,
                   name=f"{la:.0f}N {lo:.0f}E")
            for la in lats for lo in lons
        ]
        label = group_name or f"Grid {step:.0f}deg step ({len(targets)} pts)"
        return self.add_targets(targets, group_name=label)

    def add_ground_track(
        self,
        satellite: "Satellite",
        jd_start: float,
        duration_min: float = 90.0,
        step_sec: float = 30.0,
        color: str | None = None,
        alt_km: float = 0.0,
    ) -> "Globe3D":
        """Add a satellite ground track.

        Parameters
        ----------
        satellite : Satellite
        jd_start : float — start Julian Date
        duration_min : float — track duration [min]
        step_sec : float — sample interval [s]
        color : str | None — hex color (cycles through palette if None)
        alt_km : float — project track onto sphere at this altitude [km]
        """
        color = color or self._next_track_color()
        n_steps = max(2, int(duration_min * 60 / step_sec))
        lats, lons, alts = [], [], []
        for i in range(n_steps):
            jd = jd_start + i * step_sec / 86400.0
            try:
                lat, lon, alt_m = satellite.lat_lon_alt(jd)
            except Exception:
                continue
            lats.append(lat)
            lons.append(lon)
            alts.append(alt_km if alt_km else alt_m / 1000.0)

        self._track_layers.append(_TrackLayer(
            lats=lats, lons=lons, alts=alts,
            name=satellite.name or f"SAT-{satellite.norad_id}",
            color=color,
        ))
        return self

    def add_satellite_position(
        self,
        satellite: "Satellite",
        jd: float,
        color: str | None = None,
    ) -> "Globe3D":
        """Mark the current position of a satellite as a 3-D point.

        Parameters
        ----------
        satellite : Satellite
        jd : float — Julian Date
        color : str | None — marker color
        """
        lat, lon, alt_m = satellite.lat_lon_alt(jd)
        x, y, z = _lla_to_xyz(lat, lon, alt_m / 1000.0)
        name = satellite.name or f"SAT-{satellite.norad_id}"
        self._satellite_points.append((x, y, z, name, color or self._next_track_color()))
        return self

    def show(self) -> None:
        """Render and open the globe in the default browser."""
        fig = self._build_figure()
        fig.show()

    def save(self, path: str | pathlib.Path, width: int = 1200, height: int = 900) -> None:
        """Save the globe.

        Parameters
        ----------
        path : str | Path — output file.  Extension determines format:
            .html  — interactive plotly HTML (default, no extra deps)
            .png / .jpg / .svg / .pdf — static image (requires kaleido)
        width, height : int — pixel dimensions for static export
        """
        import plotly.io as pio
        path = pathlib.Path(path)
        fig = self._build_figure()
        if path.suffix.lower() == ".html":
            fig.write_html(str(path), include_plotlyjs="cdn")
        else:
            pio.write_image(fig, str(path), width=width, height=height)
        print(f"Saved globe to {path}")

    # ── Figure construction ────────────────────────────────────────────────────

    def _build_figure(self):
        import plotly.graph_objects as go

        traces: list = []

        # ── Earth sphere ──────────────────────────────────────────────────────
        traces += self._make_earth_surface()

        # ── Graticule ─────────────────────────────────────────────────────────
        traces += self._make_graticule()

        # ── Target groups ─────────────────────────────────────────────────────
        for targets, group in self._target_groups:
            traces += self._make_target_traces(targets, group)

        # ── Ground tracks ─────────────────────────────────────────────────────
        for layer in self._track_layers:
            traces.append(self._make_track_trace(layer))

        # ── Satellite positions ───────────────────────────────────────────────
        for x, y, z, name, color in self._satellite_points:
            traces.append(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode="markers+text",
                text=[name],
                textposition="top center",
                marker=dict(size=10, color=color, symbol="diamond",
                            line=dict(width=1, color="white")),
                name=name,
                hovertemplate=f"<b>{name}</b><br>({x:.0f}, {y:.0f}, {z:.0f}) km",
            ))

        bg = "#0d0d1a" if self.dark_theme else "#f0f4f8"
        grid_color = "#334" if self.dark_theme else "#ccd"

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=self.title, font=dict(size=18, color="white" if self.dark_theme else "#222")),
            paper_bgcolor=bg,
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                           showbackground=False, title=""),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                           showbackground=False, title=""),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                           showbackground=False, title=""),
                bgcolor=bg,
                aspectmode="data",
                camera=dict(eye=dict(x=1.4, y=1.0, z=0.7)),
            ),
            legend=dict(
                bgcolor="rgba(30,30,50,0.8)" if self.dark_theme else "rgba(255,255,255,0.8)",
                bordercolor="#556",
                borderwidth=1,
                font=dict(color="white" if self.dark_theme else "#222", size=11),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    # ── Trace builders ─────────────────────────────────────────────────────────

    def _make_earth_surface(self):
        import plotly.graph_objects as go

        n = 72
        lat_vals = np.linspace(-90, 90, n // 2)
        lon_vals = np.linspace(-180, 180, n)
        lo_grid, la_grid = np.meshgrid(np.radians(lon_vals), np.radians(lat_vals))

        e2 = 1.0 - (_B_EARTH_KM / _R_EARTH_KM) ** 2
        N  = _R_EARTH_KM / np.sqrt(1.0 - e2 * np.sin(la_grid) ** 2)
        xs = N * np.cos(la_grid) * np.cos(lo_grid)
        ys = N * np.cos(la_grid) * np.sin(lo_grid)
        zs = N * (1.0 - e2) * np.sin(la_grid)

        return [go.Surface(
            x=xs, y=ys, z=zs,
            colorscale=[[0, "#0a2a4a"], [0.4, "#0d5a6e"],
                        [0.6, "#1a7a5e"], [1, "#2a4a2a"]],
            showscale=False,
            opacity=self.sphere_opacity,
            hoverinfo="skip",
            name="Earth",
            showlegend=False,
        )]

    def _make_graticule(self):
        import plotly.graph_objects as go

        color = "rgba(80,100,140,0.6)" if self.dark_theme else "rgba(100,120,180,0.5)"
        traces = []
        step = self.graticule_step
        pts = 181  # interpolation points per line

        # Parallels (constant latitude)
        for lat in np.arange(-90, 90 + 1e-9, step):
            lons = np.linspace(-180, 180, pts)
            xyz = [_lla_to_xyz(lat, lo) for lo in lons]
            xs, ys, zs = zip(*xyz)
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=1),
                hoverinfo="skip",
                showlegend=False,
            ))

        # Meridians (constant longitude)
        for lon in np.arange(-180, 180 + 1e-9, step):
            lats = np.linspace(-90, 90, pts)
            xyz = [_lla_to_xyz(la, lon) for la in lats]
            xs, ys, zs = zip(*xyz)
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=1),
                hoverinfo="skip",
                showlegend=False,
            ))

        return traces

    def _make_target_traces(self, targets: list[Target], group: str | None):
        """Return one Scatter3d trace per unique priority value."""
        import plotly.graph_objects as go
        from collections import defaultdict

        by_priority: dict[int, list[Target]] = defaultdict(list)
        for t in targets:
            by_priority[t.priority].append(t)

        traces = []
        group_label = group or "Targets"
        first = True
        for priority in sorted(by_priority.keys()):
            pts = by_priority[priority]
            xs, ys, zs, names = [], [], [], []
            for t in pts:
                x, y, z = _lla_to_xyz(t.lat, t.lon, t.alt)
                xs.append(x); ys.append(y); zs.append(z)
                names.append(t.name or f"{t.lat:.1f}N {t.lon:.1f}E")

            color = self._PRIORITY_COLORS.get(priority, "#aaaaaa")
            hover = [
                f"<b>{n}</b><br>Lat {pts[i].lat:.2f} Lon {pts[i].lon:.2f}"
                f"<br>Alt {pts[i].alt:.1f} km  Priority {priority}"
                for i, n in enumerate(names)
            ]
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(
                    size=4,
                    color=color,
                    opacity=0.85,
                    line=dict(width=0),
                ),
                text=names,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
                name=f"{group_label} P{priority}",
                legendgroup=group_label,
                showlegend=first,
            ))
            first = False

        return traces

    def _make_track_trace(self, layer: _TrackLayer):
        import plotly.graph_objects as go

        # Handle antimeridian discontinuities: insert NaN gaps at ±180° crossings
        xs, ys, zs, hovers = [], [], [], []
        prev_lon = None
        for lat, lon, alt in zip(layer.lats, layer.lons, layer.alts):
            if prev_lon is not None and abs(lon - prev_lon) > 180:
                xs.append(None); ys.append(None); zs.append(None)
                hovers.append(None)
            x, y, z = _lla_to_xyz(lat, lon, alt)
            xs.append(x); ys.append(y); zs.append(z)
            hovers.append(f"<b>{layer.name}</b><br>Lat {lat:.2f} Lon {lon:.2f}<br>Alt {alt:.0f} km")
            prev_lon = lon

        return go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=layer.color, width=3, dash=layer.dash),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers,
            name=layer.name,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _next_track_color(self) -> str:
        color = self._TRACK_PALETTE[self._track_color_idx % len(self._TRACK_PALETTE)]
        self._track_color_idx += 1
        return color

"""
plot_rri_comparison.py -- Revisit Rate Interval comparison across teams and epochs.

Plots RRI data from three teams, each with 2-4 scenarios, for epochs
2027, 2032, and 2035.  For each team the plot shows:
  - Individual scenario data points (filled markers, team colour)
  - Semi-transparent min/max band (data limits)
  - Inner ±1 sigma band (tighter spread)
  - Mean line connecting epoch means
  - Linear trendline fitted to all scenario data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Example dataset
# RRI values [hours] keyed as  {team: {epoch: [sc1, sc2, ...]}}
# ---------------------------------------------------------------------------

EPOCHS = [2027, 2032, 2035]

DATA = {
    "Team Alpha": {
        2027: [8.5, 9.1, 7.8, 10.2],
        2032: [6.2, 7.0, 5.5,  8.1],
        2035: [4.8, 5.5, 4.2,  6.3],
    },
    "Team Bravo": {
        2027: [6.5, 7.2, 5.9, 6.9],
        2032: [4.8, 5.3, 4.2, 5.0],
        2035: [3.2, 3.8, 2.7, 3.5],
    },
    "Team Charlie": {
        2027: [4.5, 5.2],
        2032: [3.1, 3.7],
        2035: [2.2, 2.8],
    },
}

# Okabe-Ito colorblind-safe palette
TEAM_STYLE = {
    "Team Alpha":   dict(color="#E69F00", marker="o"),   # orange
    "Team Bravo":   dict(color="#0072B2", marker="s"),   # blue
    "Team Charlie": dict(color="#009E73", marker="^"),   # bluish green
}

# ---------------------------------------------------------------------------
# Helper: per-epoch statistics
# ---------------------------------------------------------------------------

def epoch_stats(team_data):
    x      = np.array(EPOCHS, dtype=float)
    means  = np.array([np.mean(team_data[e]) for e in EPOCHS])
    stds   = np.array([np.std(team_data[e],  ddof=0) for e in EPOCHS])
    mins   = np.array([np.min(team_data[e])  for e in EPOCHS])
    maxs   = np.array([np.max(team_data[e])  for e in EPOCHS])
    return x, means, stds, mins, maxs


def all_pairs(team_data):
    """Flatten all (epoch, rri) pairs for trendline fitting."""
    xs, ys = [], []
    for epoch in EPOCHS:
        for v in team_data[epoch]:
            xs.append(float(epoch))
            ys.append(v)
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Build figure: main panel + spread panel
# ---------------------------------------------------------------------------

fig, (ax_main, ax_spread) = plt.subplots(
    1, 2,
    figsize=(14, 6),
    gridspec_kw={"width_ratios": [3, 1.2]},
)
fig.suptitle("Revisit Rate Interval - Multi-Team Comparison", fontsize=14, y=1.01)

x_arr = np.array(EPOCHS, dtype=float)
x_pad = np.linspace(EPOCHS[0] - 1.5, EPOCHS[-1] + 1.5, 200)

# Shared y-axis limits: 10 % padding below min and above max across all data
_all_vals = [v for td in DATA.values() for e in EPOCHS for v in td[e]]
_y_min, _y_max = min(_all_vals), max(_all_vals)
_y_pad = 0.10 * (_y_max - _y_min)
Y_LIM = (_y_min - _y_pad, _y_max + _y_pad)

# ── Left panel: RRI values ──────────────────────────────────────────────────

# Overall band spanning all teams and scenarios at each epoch
overall_mins = np.array([
    min(v for td in DATA.values() for v in td[e]) for e in EPOCHS
], dtype=float)
overall_maxs = np.array([
    max(v for td in DATA.values() for v in td[e]) for e in EPOCHS
], dtype=float)

ax_main.fill_between(x_arr, overall_mins, overall_maxs,
                     color="silver", alpha=0.35, linewidth=0,
                     zorder=1, label="Overall data limits")
ax_main.plot(x_arr, overall_mins, color="gray", linewidth=0.8,
             linestyle=":", zorder=2)
ax_main.plot(x_arr, overall_maxs, color="gray", linewidth=0.8,
             linestyle=":", zorder=2)

pending_labels = []   # collect (y_raw, team, color, marker, n_sc) for nudging

for team, style in TEAM_STYLE.items():
    color  = style["color"]
    marker = style["marker"]
    td     = DATA[team]
    x, means, stds, mins, maxs = epoch_stats(td)

    # mean line
    ax_main.plot(x, means, color=color, linewidth=2.2,
                 marker=marker, markersize=8, zorder=4,
                 label=f"_{team} mean")

    # individual scenario points at exact epoch (no jitter)
    for epoch in EPOCHS:
        for v in td[epoch]:
            ax_main.scatter(epoch, v,
                            color=color, marker=marker,
                            s=50, zorder=5, alpha=0.65,
                            edgecolors="white", linewidths=0.5)

    # linear trendline
    xs_all, ys_all = all_pairs(td)
    coeffs  = np.polyfit(xs_all, ys_all, 1)
    y_trend = np.polyval(coeffs, x_pad)
    ax_main.plot(x_pad, y_trend,
                 color=color, linestyle="--", linewidth=1.4,
                 alpha=0.75, zorder=2)

    n_sc  = len(DATA[team][EPOCHS[0]])
    y_raw = float(np.polyval(coeffs, x_pad[-1]))
    pending_labels.append((y_raw, team, color, marker, n_sc))

# -- resolve vertical label overlaps ----------------------------------------
MIN_GAP = 0.75   # minimum vertical gap between label centres [hours]

pending_labels.sort(key=lambda t: t[0])          # sort by y ascending
ys_adj = [t[0] for t in pending_labels]

for _ in range(300):                              # iterative nudge
    changed = False
    for i in range(len(ys_adj) - 1):
        gap = ys_adj[i + 1] - ys_adj[i]
        if gap < MIN_GAP:
            mid          = (ys_adj[i] + ys_adj[i + 1]) / 2.0
            ys_adj[i]     = mid - MIN_GAP / 2.0
            ys_adj[i + 1] = mid + MIN_GAP / 2.0
            changed = True
    if not changed:
        break

x_lbl = x_pad[-1] + 0.2
for (y_raw, team, color, marker, n_sc), y_adj in zip(pending_labels, ys_adj):
    # thin connector from trendline tip to (possibly nudged) label
    if abs(y_raw - y_adj) > 0.05:
        ax_main.plot([x_pad[-1], x_lbl - 0.1], [y_raw, y_adj],
                     color=color, linewidth=0.7, alpha=0.5,
                     linestyle="-", zorder=5)
    ax_main.text(x_lbl, y_adj, f"{team}\n({n_sc} sc)",
                 color=color, fontsize=8.5, fontweight="bold",
                 va="center", ha="left", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                           ec=color, alpha=0.9, linewidth=0.8))

ax_main.set_xticks(EPOCHS)
ax_main.set_xticklabels([str(e) for e in EPOCHS], fontsize=11)
ax_main.set_xlim(EPOCHS[0] - 2, EPOCHS[-1] + 4)   # extra right margin for labels
ax_main.set_ylim(*Y_LIM)
ax_main.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax_main.set_xlabel("Epoch", fontsize=12)
ax_main.set_ylabel("Revisit Rate Interval [hours]", fontsize=12)
ax_main.set_title("RRI by Team and Epoch", fontsize=12)
ax_main.grid(True, alpha=0.25, linestyle="--")

# compact legend: band and line style only (teams now labelled directly)
legend_elems = [
    Line2D([0], [0], color="gray", linewidth=1.4, linestyle="--",
           label="Linear trendline"),
    mpatches.Patch(facecolor="silver", alpha=0.55,
                   edgecolor="gray", linewidth=0.8,
                   label="Overall data limits (all teams)"),
]
ax_main.legend(handles=legend_elems, loc="upper right",
               fontsize=9.5, framealpha=0.9)

# ── Right panel: box plots per team per epoch ──────────────────────────────
# Layout: groups of teams side-by-side at each epoch tick
n_teams   = len(TEAM_STYLE)
bw        = 0.22          # box width in x-units (epochs are integers 0,1,2)
offsets   = np.linspace(-(n_teams - 1) / 2, (n_teams - 1) / 2, n_teams) * bw

x_epoch_idx = np.arange(len(EPOCHS))   # 0, 1, 2

for ti, (team, style) in enumerate(TEAM_STYLE.items()):
    color = style["color"]
    td    = DATA[team]

    box_data  = [td[e] for e in EPOCHS]
    positions = x_epoch_idx + offsets[ti]

    bp = ax_spread.boxplot(
        box_data,
        positions=positions,
        widths=bw * 0.85,
        patch_artist=True,
        manage_ticks=False,
        # percentiles: whiskers = min/max (0th/100th), box = 25th-75th, line = median
        whis=[0, 100],
        medianprops=dict(color="white",   linewidth=2.0),
        boxprops=dict(   facecolor=color, alpha=0.6, linewidth=0.8,
                         edgecolor=color),
        whiskerprops=dict(color=color, linewidth=1.2, linestyle="-"),
        capprops=dict(   color=color, linewidth=1.5),
        flierprops=dict( marker="o", markerfacecolor=color, markersize=4,
                         alpha=0.7, markeredgewidth=0),
    )

ax_spread.set_xticks(x_epoch_idx)
ax_spread.set_xticklabels([str(e) for e in EPOCHS], fontsize=11)
ax_spread.set_xlim(-0.5, len(EPOCHS) - 0.5)
ax_spread.tick_params(axis="y", labelleft=False, length=0)
ax_spread.set_title("RRI Distribution per Team/Epoch", fontsize=12)
ax_spread.grid(True, axis="y", alpha=0.25, linestyle="--")
ax_spread.set_ylim(*Y_LIM)
ax_spread.yaxis.set_major_locator(ticker.MultipleLocator(1))

# team colour swatches as legend
spread_legend = [
    mpatches.Patch(facecolor=style["color"], alpha=0.6,
                   edgecolor=style["color"], label=team)
    for team, style in TEAM_STYLE.items()
]
ax_spread.legend(handles=spread_legend, fontsize=9,
                 loc="upper right", framealpha=0.9)

# ── Save ────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.subplots_adjust(wspace=0.05)
plt.savefig("rri_comparison.png", dpi=150, bbox_inches="tight")
print("Saved rri_comparison.png")
plt.show()

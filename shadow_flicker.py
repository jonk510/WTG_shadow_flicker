"""
Shadow Flicker Calculator — core engine.

For each sun-up hour the algorithm checks whether the turbine rotor disc
occults the sun as seen from each receptor/grid point. Flicker occurs when
the angular distance between the sun and the turbine hub (as seen from the
receptor) is less than the apparent angular half-size of the rotor disc plus
the angular radius of the sun (~0.267°).

Azimuth convention: clockwise from North, matching pvlib output and
np.arctan2(east_component, north_component).

Flat-terrain assumption: hub height is taken above local ground at each
receptor; terrain-following shadow paths are not modelled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
SUN_HALF_ANGLE_RAD      = np.radians(0.267)   # angular radius of solar disc
DEFAULT_MIN_SUN_EL_DEG  = 3.0                 # ignore sun below this elevation
DEFAULT_RECEIVER_HT_M   = 1.5                 # receptor height (m) — ISO 9613-2
DEFAULT_HUB_HEIGHT_M    = 150.0
DEFAULT_ROTOR_DIAM_M    = 180.0
DEFAULT_BLADE_CHORD_M   = 4.5                 # typical blade chord (m)
# NEPC 2010: assess within 265 × max blade chord
NEPC_CHORD_MULTIPLIER   = 265
DEFAULT_ANNUAL_THRESHOLD  = 30.0   # NEPC 2010 annual limit (hr/yr)
DEFAULT_CLOUD_THRESHOLD   = 8.0    # indicative actual limit with cloud correction (hr/yr)
DEFAULT_EPSG            = 7850                     # GDA2020 / MGA zone 50


# ── Coordinate helper ─────────────────────────────────────────────────────────
def get_site_latlon(wtg_xy: np.ndarray, epsg_code: int) -> tuple[float, float]:
    """Return WGS84 (lat, lon) of the WTG layout centroid."""
    from pyproj import Transformer
    t = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    cx, cy = float(wtg_xy[:, 0].mean()), float(wtg_xy[:, 1].mean())
    lon, lat = t.transform(cx, cy)
    return lat, lon


# ── Cloud correction via NASA POWER ──────────────────────────────────────────
def fetch_cloud_correction(lat: float, lon: float, year: int) -> dict[int, float]:
    """
    Fetch monthly clearness index (ALLSKY_KT) from NASA POWER API.
    Returns {month_number: clearness_index} where 0 = fully overcast, 1 = clear.
    Used to convert worst-case (no cloud) flicker to a realistic estimate.

    Reference: NEPC 2010 guidelines use worst-case; cloud correction gives
    a realistic (non-compliance) estimate.
    """
    import requests
    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=ALLSKY_KT&community=RE"
        f"&longitude={lon:.4f}&latitude={lat:.4f}"
        f"&start={year}&end={year}&format=JSON"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    raw = resp.json()
    monthly_raw = raw["properties"]["parameter"]["ALLSKY_KT"]
    # API returns {"YYYYMM": value, ...}
    result = {}
    for key, val in monthly_raw.items():
        try:
            month = int(str(key)[4:6])  # YYYYMM → MM
            if 1 <= month <= 12:
                result[month] = float(val)
        except (ValueError, IndexError):
            pass
    return result


def apply_cloud_correction(
    flicker_by_month: dict[int, np.ndarray],
    cloud_correction: dict[int, float],
) -> np.ndarray:
    """
    Apply monthly clearness index to monthly flicker grids.
    Returns corrected annual flicker array.
    """
    corrected = None
    for month, grid in flicker_by_month.items():
        kt = cloud_correction.get(month, 1.0)
        c = grid * kt
        corrected = c if corrected is None else corrected + c
    return corrected if corrected is not None else np.zeros(1)


# ── Core flicker computation ──────────────────────────────────────────────────
def compute_shadow_flicker(
    wtg_xy: np.ndarray,
    rotor_diameter: float,
    hub_height: float,
    xx: np.ndarray,
    yy: np.ndarray,
    site_lat: float,
    site_lon: float,
    year: int = 2024,
    receiver_height: float = DEFAULT_RECEIVER_HT_M,
    min_sun_el_deg: float = DEFAULT_MIN_SUN_EL_DEG,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute shadow flicker across a 2-D grid.

    Parameters
    ----------
    wtg_xy         : (M, 2) turbine Easting/Northing in project CRS
    rotor_diameter : rotor diameter (m)
    hub_height     : hub height above ground (m)
    xx, yy         : (ny, nx) meshgrid in project CRS
    site_lat/lon   : WGS84 lat/lon of site centroid
    year           : calendar year for sun positions
    receiver_height: receptor height above local ground (m)
    min_sun_el_deg : minimum sun elevation to process (deg)
    progress_cb    : optional callable(fraction) for progress reporting

    Returns
    -------
    flicker_annual    : (ny, nx) float32 — total flicker hours/year
    flicker_max_day   : (ny, nx) float32 — max flicker hours in any single day
    flicker_by_month  : dict {month: (ny, nx)} — monthly totals for cloud correction
    """
    try:
        import pvlib
    except ImportError:
        raise ImportError("pvlib is required: pip install pvlib")

    # ── Solar positions, full year, hourly UTC ────────────────────────────────
    times = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01",
                          freq="1h", tz="UTC")[:-1]
    n_days = 366 if year % 4 == 0 else 365

    solpos   = pvlib.solarposition.get_solarposition(times, site_lat, site_lon)
    sun_az   = np.radians(solpos["azimuth"].values.astype(np.float32))
    sun_el   = np.radians(solpos["apparent_elevation"].values.astype(np.float32))

    up_mask    = sun_el > np.radians(min_sun_el_deg)
    sun_az_up  = sun_az[up_mask]
    sun_el_up  = sun_el[up_mask]
    day_of_up  = (np.where(up_mask)[0] // 24).astype(np.int16)

    # ── Precompute per-turbine geometry (constant across all time steps) ──────
    xx_f = xx.astype(np.float32)
    yy_f = yy.astype(np.float32)
    rotor_radius = np.float32(rotor_diameter * 0.5)
    dz = np.float32(hub_height - receiver_height)

    turbine_geom = []
    for T in wtg_xy:
        dx  = np.float32(T[0]) - xx_f
        dy  = np.float32(T[1]) - yy_f
        dh  = np.maximum(np.sqrt(dx * dx + dy * dy), np.float32(1.0))
        az  = np.arctan2(dx, dy).astype(np.float32)                    # (ny,nx)
        el  = np.arctan2(dz, dh).astype(np.float32)                    # (ny,nx)
        ha  = (np.arctan(rotor_radius / dh) + SUN_HALF_ANGLE_RAD      # (ny,nx)
               ).astype(np.float32)
        turbine_geom.append((az, el, ha))

    # ── Day-by-day accumulation (avoids large day×grid arrays) ───────────────
    ny, nx      = xx.shape
    flicker_annual   = np.zeros((ny, nx), dtype=np.float32)
    flicker_max_day  = np.zeros((ny, nx), dtype=np.float32)
    flicker_by_month: dict[int, np.ndarray] = {m: np.zeros((ny, nx), dtype=np.float32)
                                                for m in range(1, 13)}

    # Build day → index-into-sun_az_up mapping once
    from collections import defaultdict
    day_hours: dict[int, list[int]] = defaultdict(list)
    for i, d in enumerate(day_of_up):
        day_hours[int(d)].append(i)

    # Map day-of-year → calendar month using a reference year
    import datetime
    ref_jan1 = datetime.date(int(year), 1, 1)

    days_with_sun = sorted(day_hours.keys())
    n_days_sun    = len(days_with_sun)

    for step, day in enumerate(days_with_sun):
        idxs  = day_hours[day]
        c_az  = sun_az_up[idxs]   # (C,)  C ≈ 6–18
        c_el  = sun_el_up[idxs]   # (C,)

        day_flicker = np.zeros((ny, nx), dtype=np.float32)

        for az_T, el_T, ha_T in turbine_geom:
            d_az = c_az[:, None, None] - az_T[None, :, :]
            d_az = (d_az + np.pi) % (2.0 * np.pi) - np.pi
            d_el = c_el[:, None, None] - el_T[None, :, :]
            ang  = np.sqrt(d_az * d_az + d_el * d_el)
            day_flicker += (ang < ha_T[None, :, :]).sum(axis=0).astype(np.float32)

        flicker_annual += day_flicker
        np.maximum(flicker_max_day, day_flicker, out=flicker_max_day)

        month = (ref_jan1 + datetime.timedelta(days=int(day))).month
        flicker_by_month[month] += day_flicker

        if progress_cb is not None:
            progress_cb((step + 1) / n_days_sun)

    return flicker_annual, flicker_max_day, flicker_by_month


def compute_receptor_flicker(
    wtg_xy: np.ndarray,
    rotor_diameter: float,
    hub_height: float,
    receptor_xy: np.ndarray,
    site_lat: float,
    site_lon: float,
    year: int = 2024,
    receiver_height: float = DEFAULT_RECEIVER_HT_M,
    min_sun_el_deg: float = DEFAULT_MIN_SUN_EL_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flicker at discrete receptor points.
    Returns (annual_hrs, max_day_hrs), each shape (N_receptors,).
    """
    rx = receptor_xy[:, 0].reshape(1, -1).astype(np.float32)
    ry = receptor_xy[:, 1].reshape(1, -1).astype(np.float32)
    annual, max_day, _ = compute_shadow_flicker(
        wtg_xy, rotor_diameter, hub_height, rx, ry,
        site_lat, site_lon, year=year,
        receiver_height=receiver_height, min_sun_el_deg=min_sun_el_deg,
    )
    return annual.ravel(), max_day.ravel()


# ── Plotting helpers ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath


def _threshold_fill(ax, xx, yy, data, threshold, cmap="YlOrRd", alpha=0.65):
    """Smooth gradient fill + single dashed contour line at threshold."""
    vmax = max(float(data.max()), threshold * 1.2)
    levels = np.linspace(0, vmax, 64)
    cf = ax.contourf(xx, yy, data, levels=levels,
                     cmap=cmap, alpha=alpha, extend="max")
    cl = ax.contour(xx, yy, data, levels=[threshold],
                    colors=["#cc0000"], linewidths=2.5, linestyles="--")
    return cf, cl, vmax


def _make_wtg_marker():
    def _blade(angle_deg):
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        pts = np.array([
            [0, 0], [-0.10, 0.18], [-0.05, 1.00],
            [0.05, 1.00], [0.10, 0.18], [0, 0],
        ])
        rot = np.array([[c, -s], [s, c]])
        return (pts @ rot.T).tolist()
    verts, codes = [], []
    for angle in [90, 210, 330]:
        blade = _blade(angle)
        verts += blade
        codes += ([MplPath.MOVETO]
                  + [MplPath.LINETO] * (len(blade) - 2)
                  + [MplPath.CLOSEPOLY])
    t   = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    hub = np.column_stack([0.16 * np.cos(t), 0.16 * np.sin(t)])
    verts += hub.tolist() + [hub[0].tolist()]
    codes += ([MplPath.MOVETO]
              + [MplPath.LINETO] * (len(hub) - 1)
              + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)


_WTG_MARKER = _make_wtg_marker()


def _add_satellite(ax, epsg_code, bing_key=None):
    try:
        import contextily as ctx
        if bing_key:
            src = ctx.providers.Bing.Aerial(key=bing_key)
        else:
            src = ctx.providers.OpenStreetMap.Mapnik
        ctx.add_basemap(ax, crs=f"EPSG:{epsg_code}", source=src,
                        zoom="auto", attribution=False)
        return True
    except Exception:
        return False


def _format_map_axis(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    ax.set_xlabel("Easting (m)", fontsize=22)
    ax.set_ylabel("Northing (m)", fontsize=22)
    ax.tick_params(labelsize=20)


def _scatter_turbines(ax, wtg_xy):
    ax.scatter(wtg_xy[:, 0], wtg_xy[:, 1],
               marker=_WTG_MARKER, s=400, c="white",
               edgecolors="black", linewidths=1.2, zorder=10)
    for i, pos in enumerate(wtg_xy):
        ax.annotate(
            f"T{i + 1}", pos, xytext=(5, 4),
            textcoords="offset points", fontsize=18,
            color="white", fontweight="bold",
            path_effects=[mpe.withStroke(linewidth=2, foreground="black")])


def _scatter_flicker_receptors(ax, receptor_xy, annual_hrs, max_day_hrs,
                                receptor_names=None):
    for i, (pos, ann, mday) in enumerate(zip(receptor_xy, annual_hrs, max_day_hrs)):
        colour = "#e74c3c" if ann > 30 else ("#f39c12" if ann > 10 else "#2ecc71")
        ax.scatter(pos[0], pos[1], marker="D", s=100, c=colour,
                   edgecolors="black", linewidths=1.0, zorder=11)
        name = receptor_names[i] if receptor_names else f"R{i + 1}"
        label = f"{name}\n{ann:.1f} hr/yr\n{mday * 60:.0f} min/day"
        ax.annotate(label, pos, xytext=(6, 4), textcoords="offset points",
                    fontsize=16, fontweight="bold", color="white",
                    path_effects=[mpe.withStroke(linewidth=2, foreground="black")])


def plot_flicker_results(
    wtg_xy, flicker_annual, flicker_max_day, xx, yy,
    epsg_code,
    annual_threshold=DEFAULT_ANNUAL_THRESHOLD,
    use_satellite=True, bing_key=None, alpha_fill=0.60,
    hub_height=None, rotor_diameter=None, year=2024,
    receptor_xy=None, receptor_annual=None,
    receptor_max_day=None, receptor_names=None,
    is_cloud_corrected=False,
    save_path=None,
):
    xmin, xmax = float(xx.min()), float(xx.max())
    ymin, ymax = float(yy.min()), float(yy.max())

    fig = plt.figure(figsize=(44, 36))
    hub_str   = f"  ·  Hub {hub_height:.0f} m"    if hub_height    else ""
    rotor_str = f"  ·  Rotor ⌀{rotor_diameter:.0f} m" if rotor_diameter else ""
    cc_str    = "  ·  Cloud-corrected (indicative)" if is_cloud_corrected else "  ·  Worst-case"
    fig.suptitle(
        f"Shadow Flicker Assessment  ·  {year}{hub_str}{rotor_str}{cc_str}",
        fontsize=30, fontweight="bold", y=0.99)

    gs = GridSpec(2, 3, figure=fig,
                  height_ratios=[3, 1], width_ratios=[1, 1, 1],
                  hspace=0.30, wspace=0.28,
                  left=0.04, right=0.97, top=0.96, bottom=0.05)

    ax_ann   = fig.add_subplot(gs[0, :])   # full-width — annual hours
    ax_day   = fig.add_subplot(gs[1, 0])   # max hours/day map
    ax_dist  = fig.add_subplot(gs[1, 1])   # flicker vs distance
    ax_info  = fig.add_subplot(gs[1, 2])   # summary text

    # ── Panel 1 : annual flicker ──────────────────────────────────────────────
    ax_ann.set_xlim(xmin, xmax)
    ax_ann.set_ylim(ymin, ymax)
    sat_ok = _add_satellite(ax_ann, epsg_code, bing_key) if use_satellite else False

    _, cl1, vmax_ann = _threshold_fill(
        ax_ann, xx, yy, flicker_annual, annual_threshold, alpha=alpha_fill)
    ax_ann.clabel(cl1, fmt=f"{annual_threshold:g} hr/yr", fontsize=18, inline=True)

    _scatter_turbines(ax_ann, wtg_xy)

    legend_handles = [
        Line2D([0], [0], marker=_WTG_MARKER, color="w",
               markerfacecolor="white", markeredgecolor="black",
               markersize=14, label="Wind turbine"),
        Line2D([0], [0], color="#cc0000", lw=2.5, linestyle="--",
               label=f"{annual_threshold:g} hr/yr limit"),
    ]
    if receptor_xy is not None and receptor_annual is not None:
        _scatter_flicker_receptors(ax_ann, receptor_xy, receptor_annual,
                                   receptor_max_day if receptor_max_day is not None
                                   else np.zeros(len(receptor_xy)),
                                   receptor_names)
        legend_handles.append(
            Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="yellow", markeredgecolor="black",
                   markersize=10, label="Sensitive receptor"))

    title_suffix = " — Satellite" if sat_ok else ""
    ax_ann.set_title(f"Annual Shadow Flicker (hr/yr){title_suffix}",
                     fontsize=26, fontweight="bold")
    _format_map_axis(ax_ann)
    ax_ann.legend(handles=legend_handles, loc="upper left",
                  fontsize=20, framealpha=0.85)
    sm1 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(0, vmax_ann))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax_ann, shrink=0.80, pad=0.02, aspect=28)
    cb1.set_label("Shadow flicker (hours/year)", fontsize=20)
    cb1.ax.tick_params(labelsize=18)

    # ── Panel 2 : max hours/day ───────────────────────────────────────────────
    DAY_THRESHOLD = 0.5   # 30 min/day in hours
    _, cl2, vmax_day = _threshold_fill(
        ax_day, xx, yy, flicker_max_day, DAY_THRESHOLD, alpha=0.80)
    ax_day.clabel(cl2, fmt="30 min/day", fontsize=12, inline=True)
    _scatter_turbines(ax_day, wtg_xy)
    ax_day.set_title("Max Flicker — Single Day (hr/day)",
                     fontsize=14, fontweight="bold")
    _format_map_axis(ax_day)
    sm2 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(0, vmax_day))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax_day, shrink=0.80, pad=0.02, aspect=28)
    cb2.set_label("Max flicker (hr/day)", fontsize=11)
    cb2.ax.tick_params(labelsize=10)

    # ── Panel 3 : flicker vs distance ────────────────────────────────────────
    centroid   = wtg_xy.mean(axis=0)
    flat       = flicker_annual.ravel()
    grid_pts   = np.column_stack([xx.ravel(), yy.ravel()])
    r_from_cen = np.sqrt(((grid_pts - centroid) ** 2).sum(axis=1))
    r_max      = float(r_from_cen.max())

    bin_edges = np.linspace(0, r_max, 60)
    bin_r, bin_max = [], []
    for j in range(len(bin_edges) - 1):
        m = (r_from_cen >= bin_edges[j]) & (r_from_cen < bin_edges[j + 1])
        if m.sum() > 0:
            bin_r.append(0.5 * (bin_edges[j] + bin_edges[j + 1]))
            bin_max.append(float(flat[m].max()))

    ax_dist.plot(bin_r, bin_max, "b-", lw=2.0, label="Max hr/yr")
    ax_dist.axhline(annual_threshold, color="#cc0000", lw=2.0, linestyle="--",
                    label=f"{annual_threshold:g} hr/yr limit")
    ax_dist.set_xlabel("Distance from layout centroid (m)", fontsize=13)
    ax_dist.set_ylabel("Max shadow flicker (hr/yr)", fontsize=13)
    ax_dist.set_title("Flicker vs. Distance", fontsize=14, fontweight="bold")
    ax_dist.set_xlim(0, r_max)
    ymax_dist = max(max(bin_max) * 1.1 if bin_max else annual_threshold * 1.2,
                    annual_threshold * 1.2)
    ax_dist.set_ylim(0, ymax_dist)
    ax_dist.grid(True, alpha=0.3)
    ax_dist.tick_params(labelsize=12)
    ax_dist.legend(fontsize=11)

    mask_thr = flat >= annual_threshold
    if mask_thr.any():
        r_thr = float(r_from_cen[mask_thr].max())
        ax_dist.axvline(r_thr, color="#cc0000", lw=0.8, linestyle=":")
        ax_dist.text(r_thr, ymax_dist * 0.92,
                     f"{r_thr:.0f} m", fontsize=10, color="#cc0000", ha="center")

    # ── Panel 4 : summary stats ───────────────────────────────────────────────
    ax_info.axis("off")
    max_ann  = float(flicker_annual.max())
    max_day  = float(flicker_max_day.max())

    mask_thr2 = flat >= annual_threshold
    r_thr2    = float(r_from_cen[mask_thr2].max()) if mask_thr2.any() else 0.0
    mode_str  = "Cloud-corrected" if is_cloud_corrected else "Worst-case"

    lines = [
        "── Assessment Summary ──",
        "",
        f"Mode:                 {mode_str}",
        f"Peak annual flicker:  {max_ann:.1f} hr/yr",
        f"Peak single-day:       {max_day * 60:.0f} min/day",
        f"{annual_threshold:g} hr/yr radius:  {r_thr2:.0f} m",
        "",
        "── Guideline thresholds ──",
        "",
        f"Annual:    {annual_threshold:g} hr/yr (NEPC 2010)" if not is_cloud_corrected
        else f"Annual:    {annual_threshold:g} hr/yr (cloud-corrected)",
        "Daily:     30 min/day (German practice)",
        "",
        "── Model notes ──",
        "",
        "• Flat-terrain assumption",
        "• 100 % turbine availability",
        "• Hourly sun positions via pvlib",
        f"• Assessment year: {year}",
    ]
    ax_info.text(0.05, 0.95, "\n".join(lines),
                 transform=ax_info.transAxes,
                 fontsize=12, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                           edgecolor="#aaaaaa", linewidth=1.0, alpha=0.95))

    note = (f"Shadow flicker  ·  {len(wtg_xy)} turbines  ·  "
            f"Hub {hub_height:.0f} m  ·  Rotor ⌀{rotor_diameter:.0f} m  ·  "
            f"{mode_str}")
    fig.text(0.5, 0.008, note, ha="center", va="bottom", fontsize=14,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbe6",
                       edgecolor="#c8a800", linewidth=1.2, alpha=0.92))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

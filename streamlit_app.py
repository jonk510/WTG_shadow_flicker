"""
JK's Shadow Flicker Estimator — Streamlit Web App
Geometrical shadow flicker model using pvlib solar positions.
"""

import io
import os
import sys
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shadow_flicker import (
    DEFAULT_EPSG, DEFAULT_HUB_HEIGHT_M, DEFAULT_ROTOR_DIAM_M,
    DEFAULT_BLADE_CHORD_M, NEPC_CHORD_MULTIPLIER,
    DEFAULT_RECEIVER_HT_M, DEFAULT_FLICKER_LEVELS, DEFAULT_MIN_SUN_EL_DEG,
    get_site_latlon, compute_shadow_flicker, compute_receptor_flicker,
    fetch_cloud_correction, apply_cloud_correction,
    plot_flicker_results,
)


# ── File upload helpers (shapefile + KMZ/KML) ────────────────────────────────
def _load_shapefile_points(uploaded_files, target_epsg: int):
    try:
        import geopandas as gpd
    except ImportError:
        st.error("Install `geopandas` to use shapefile upload.")
        return None, None
    with tempfile.TemporaryDirectory() as tmp:
        for f in uploaded_files:
            with open(os.path.join(tmp, f.name), "wb") as fh:
                fh.write(f.read())
        shp_files = [p for p in os.listdir(tmp) if p.endswith(".shp")]
        if not shp_files:
            st.error("No .shp file found.")
            return None, None
        gdf = gpd.read_file(os.path.join(tmp, shp_files[0]))
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf.geometry.geom_type.isin(["Point", "MultiPoint"])]
    if gdf.empty:
        st.error("Shapefile contains no point features.")
        return None, None
    gdf = gdf.to_crs(epsg=target_epsg)
    xy  = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    nc  = next((c for c in gdf.columns
                if c.lower() in ("name", "label", "id", "receptor")), None)
    names = (gdf[nc].astype(str).tolist() if nc
             else [f"R{i+1}" for i in range(len(xy))])
    return xy, names


def _load_kmz_points(uploaded_file, target_epsg: int):
    import zipfile
    import xml.etree.ElementTree as ET
    from pyproj import Transformer
    raw = uploaded_file.read()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                st.error("No KML found inside KMZ.")
                return None, None
            kml_bytes = z.read(kml_names[0])
    else:
        kml_bytes = raw
    root = ET.fromstring(kml_bytes)
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
    lons, lats, names = [], [], []
    for pm in root.iter("Placemark"):
        pt = pm.find(".//Point")
        if pt is None:
            continue
        coords_el = pt.find("coordinates")
        if coords_el is None or not coords_el.text:
            continue
        parts = coords_el.text.strip().split(",")
        lons.append(float(parts[0]))
        lats.append(float(parts[1]))
        name_el = pm.find("name")
        names.append(name_el.text.strip()
                     if name_el is not None and name_el.text
                     else f"P{len(lons)}")
    if not lons:
        st.error("No point features found in KMZ/KML.")
        return None, None
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
    xs, ys = t.transform(lons, lats)
    return np.column_stack([xs, ys]), names


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Shadow Flicker Estimator", layout="wide")
st.title("JK's Shadow Flicker Estimator")
st.caption("Geometrical shadow flicker model · pvlib solar positions · worst-case (100 % availability)")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Coordinate System")
    epsg_code = st.number_input(
        "Coordinate System's EPSG code",
        value=int(DEFAULT_EPSG), min_value=1000, max_value=99999, step=1)
    try:
        from pyproj import CRS
        st.caption(f"📐 {CRS.from_epsg(int(epsg_code)).name}")
    except Exception:
        st.caption("⚠️ Unrecognised EPSG code")

    st.divider()
    st.subheader("Turbine Geometry")
    hub_height     = st.number_input("Hub height (m)",
                                     value=float(DEFAULT_HUB_HEIGHT_M),
                                     min_value=50.0, max_value=300.0, step=5.0)
    rotor_diameter = st.number_input("Rotor diameter (m)",
                                     value=float(DEFAULT_ROTOR_DIAM_M),
                                     min_value=50.0, max_value=400.0, step=5.0)
    blade_chord    = st.number_input("Max blade chord (m)",
                                     value=float(DEFAULT_BLADE_CHORD_M),
                                     min_value=1.0, max_value=20.0, step=0.5,
                                     help="Used to calculate NEPC assessment distance "
                                          f"({NEPC_CHORD_MULTIPLIER} × chord).")
    nepc_dist = blade_chord * NEPC_CHORD_MULTIPLIER
    st.caption(f"NEPC assessment distance: **{nepc_dist:.0f} m** "
               f"({NEPC_CHORD_MULTIPLIER} × {blade_chord:.1f} m chord)")

    st.divider()
    st.subheader("Assessment Parameters")
    year = st.number_input("Assessment year", value=2024,
                            min_value=2000, max_value=2100, step=1)
    receiver_height = st.number_input(
        "Receptor height (m)", value=float(DEFAULT_RECEIVER_HT_M),
        min_value=0.5, max_value=10.0, step=0.5,
        help="Height of the receptor above local ground. "
             "1.5 m = person standing outdoors (ISO 9613-2).")
    min_sun_el = st.slider(
        "Min sun elevation (°)", 0.0, 10.0, float(DEFAULT_MIN_SUN_EL_DEG), 0.5,
        help="Ignore sun angles below this value to avoid grazing-angle artefacts.")
    use_satellite = st.toggle("Satellite background", value=True)

    st.divider()
    st.subheader("Grid")
    grid_spacing_m = st.number_input(
        "Grid spacing (m)", value=50.0, min_value=10.0, max_value=500.0, step=10.0,
        help="Smaller = finer contours but slower computation.")
    buffer_m = st.number_input(
        "Buffer beyond layout (m)", value=3000.0, min_value=500.0, step=500.0)

    st.divider()
    st.subheader("Contour levels (hr/yr)")
    levels_str = st.text_input(
        "Comma-separated", value=", ".join(str(x) for x in DEFAULT_FLICKER_LEVELS))
    try:
        contour_levels = sorted(float(x.strip())
                                for x in levels_str.split(",") if x.strip())
    except ValueError:
        contour_levels = list(DEFAULT_FLICKER_LEVELS)
        st.warning("Invalid levels — using defaults.")

    alpha_fill = st.slider("Contour opacity", 0.10, 1.0, 0.60, 0.05)

    st.divider()
    st.subheader("Cloud correction")
    apply_cloud = st.toggle(
        "Apply cloud correction (NASA POWER)",
        value=False,
        help="Fetches monthly clearness index from NASA POWER and applies it "
             "to give a realistic (non-worst-case) estimate alongside the "
             "NEPC worst-case result.")
    st.caption("NEPC 2010 compliance uses the worst-case result. "
               "Cloud-corrected result is indicative only.")

    st.divider()
    st.subheader("Model notes")
    st.info(
        "**NEPC 2010 compliant** — worst-case methodology (100 % availability, "
        "no cloud cover). Threshold: 30 hr/yr at receptor.\n\n"
        "**Flat terrain** — terrain-following shadows not modelled.\n\n"
        "**Assessment distance**: 265 × max blade chord (NEPC 2010)."
    )

# ── Input columns ─────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

# Column 1 — WTG layout
with c1:
    st.subheader("1 · Turbine Layout")
    wtg_fmt = st.radio("Format", ["CSV", "Shapefile", "KMZ / KML"],
                       horizontal=True, key="wtg_fmt")
    wtg_xy = None
    if wtg_fmt == "CSV":
        wtg_file = st.file_uploader("CSV with X, Y columns",
                                    type=["csv", "txt"], key="wtg_csv")
        if wtg_file:
            wtg_df = pd.read_csv(wtg_file)
            wtg_df.columns = [c.strip().lstrip("﻿").upper()
                               for c in wtg_df.columns]
            wtg_df.dropna(subset=["X", "Y"], inplace=True)
            wtg_xy = wtg_df[["X", "Y"]].values.astype(float)
    elif wtg_fmt == "Shapefile":
        wtg_shp = st.file_uploader(
            "Shapefile parts (.shp .shx .dbf .prj)",
            type=["shp", "shx", "dbf", "prj", "cpg"],
            accept_multiple_files=True, key="wtg_shp")
        if wtg_shp:
            wtg_xy, _ = _load_shapefile_points(wtg_shp, int(epsg_code))
    else:
        wtg_kmz = st.file_uploader("KMZ or KML file",
                                   type=["kmz", "kml"], key="wtg_kmz")
        if wtg_kmz:
            wtg_xy, _ = _load_kmz_points(wtg_kmz, int(epsg_code))

    if wtg_xy is not None:
        st.success(f"{len(wtg_xy)} turbines loaded")
        df_disp = pd.DataFrame(wtg_xy, columns=["Easting", "Northing"])
        df_disp.index = range(1, len(df_disp) + 1)
        st.dataframe(df_disp, use_container_width=True)

        # Show derived lat/lon for info
        try:
            lat, lon = get_site_latlon(wtg_xy, int(epsg_code))
            st.caption(f"Site centroid: {lat:.4f}°N, {lon:.4f}°E")
        except Exception:
            pass

# Column 2 — Site summary / parameter confirmation
with c2:
    st.subheader("2 · Parameter Summary")
    st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Hub height | {hub_height:.0f} m |
| Rotor diameter | {rotor_diameter:.0f} m |
| Receiver height | {receiver_height:.1f} m |
| Assessment year | {year} |
| Min sun elevation | {min_sun_el:.1f}° |
| Grid spacing | {grid_spacing_m:.0f} m |
| Buffer | {buffer_m:.0f} m |
    """)

    st.subheader("3 · SA / International Guidelines")
    st.markdown("""
| Criterion | Threshold |
|-----------|-----------|
| Annual flicker | **30 hr/yr** |
| Daily flicker | **30 min/day** |

*South Australia EPA Wind Farm Guidelines 2021 · German TA Lärm approach*
    """)

# ── Sensitive receptors ───────────────────────────────────────────────────────
st.divider()
st.subheader("4 · Sensitive Receptors (optional)")
rec_fmt = st.radio("Format", ["CSV", "Shapefile", "KMZ / KML"],
                   horizontal=True, key="rec_fmt")
receptor_xy, receptor_names = None, None
if rec_fmt == "CSV":
    rec_file = st.file_uploader(
        "CSV — columns: X, Y (and optionally Name)",
        type=["csv", "txt"], key="rec_csv")
    if rec_file:
        rec_df = pd.read_csv(rec_file)
        rec_df.columns = [c.strip().lstrip("﻿").upper()
                          for c in rec_df.columns]
        rec_df.dropna(subset=["X", "Y"], inplace=True)
        receptor_xy    = rec_df[["X", "Y"]].values.astype(float)
        receptor_names = (rec_df["NAME"].tolist() if "NAME" in rec_df.columns
                          else [f"R{i+1}" for i in range(len(receptor_xy))])
        st.success(f"{len(receptor_xy)} receptors loaded: "
                   f"{', '.join(receptor_names)}")
elif rec_fmt == "Shapefile":
    rec_shp = st.file_uploader(
        "Shapefile parts (.shp .shx .dbf .prj)",
        type=["shp", "shx", "dbf", "prj", "cpg"],
        accept_multiple_files=True, key="rec_shp")
    if rec_shp:
        receptor_xy, receptor_names = _load_shapefile_points(
            rec_shp, int(epsg_code))
        if receptor_xy is not None:
            st.success(f"{len(receptor_xy)} receptors loaded: "
                       f"{', '.join(receptor_names)}")
else:
    rec_kmz = st.file_uploader("KMZ or KML file",
                               type=["kmz", "kml"], key="rec_kmz")
    if rec_kmz:
        receptor_xy, receptor_names = _load_kmz_points(rec_kmz, int(epsg_code))
        if receptor_xy is not None:
            st.success(f"{len(receptor_xy)} receptors loaded: "
                       f"{', '.join(receptor_names)}")

# ── Run button ────────────────────────────────────────────────────────────────
st.divider()
ready = wtg_xy is not None
if not ready:
    st.info("Upload a turbine layout to begin.")

if st.button("Run Shadow Flicker Analysis", type="primary",
             disabled=not ready, use_container_width=True):

    with st.spinner("Running shadow flicker analysis…"):

        # Grid
        xmin = wtg_xy[:, 0].min() - buffer_m
        xmax = wtg_xy[:, 0].max() + buffer_m
        ymin = wtg_xy[:, 1].min() - buffer_m
        ymax = wtg_xy[:, 1].max() + buffer_m
        nx   = max(10, int(round((xmax - xmin) / grid_spacing_m)) + 1)
        ny   = max(10, int(round((ymax - ymin) / grid_spacing_m)) + 1)
        xi   = np.linspace(xmin, xmax, nx)
        yi   = np.linspace(ymin, ymax, ny)
        xx, yy = np.meshgrid(xi, yi)

        st.write(f"Grid: {nx}×{ny} pts @ {grid_spacing_m:.0f} m  ·  "
                 f"{len(wtg_xy)} turbines  ·  year {year}")

        # Site lat/lon
        site_lat, site_lon = get_site_latlon(wtg_xy, int(epsg_code))

        # Progress bar
        prog_bar = st.progress(0.0, text="Computing solar positions and flicker…")

        def _progress(frac):
            prog_bar.progress(min(frac, 1.0),
                              text=f"Processing days… {frac*100:.0f}%")

        flicker_annual, flicker_max_day, flicker_by_month = compute_shadow_flicker(
            wtg_xy, rotor_diameter, hub_height, xx, yy,
            site_lat, site_lon,
            year=int(year),
            receiver_height=float(receiver_height),
            min_sun_el_deg=float(min_sun_el),
            progress_cb=_progress,
        )
        prog_bar.progress(1.0, text="Done!")

        # Cloud correction
        flicker_corrected = None
        cloud_data = None
        if apply_cloud:
            try:
                with st.spinner("Fetching cloud data from NASA POWER…"):
                    cloud_data = fetch_cloud_correction(site_lat, site_lon, int(year))
                flicker_corrected = apply_cloud_correction(flicker_by_month, cloud_data)
                mean_kt = sum(cloud_data.values()) / len(cloud_data)
                st.success(
                    f"Cloud correction applied — mean annual clearness index: "
                    f"{mean_kt:.2f} (1.0 = always clear)")
            except Exception as e:
                st.warning(f"Could not fetch cloud data: {e}. Showing worst-case only.")

        # Receptor flicker
        receptor_annual  = None
        receptor_max_day_arr = None
        if receptor_xy is not None:
            st.write("Calculating receptor flicker levels…")
            receptor_annual, receptor_max_day_arr = compute_receptor_flicker(
                wtg_xy, rotor_diameter, hub_height, receptor_xy,
                site_lat, site_lon,
                year=int(year),
                receiver_height=float(receiver_height),
                min_sun_el_deg=float(min_sun_el),
            )

        # Plot
        fig = plot_flicker_results(
            wtg_xy, flicker_annual, flicker_max_day, xx, yy,
            contour_levels, int(epsg_code),
            use_satellite=use_satellite,
            alpha_fill=alpha_fill,
            hub_height=hub_height,
            rotor_diameter=rotor_diameter,
            year=int(year),
            receptor_xy=receptor_xy,
            receptor_annual=receptor_annual,
            receptor_max_day=receptor_max_day_arr,
            receptor_names=receptor_names,
        )

        fig.set_dpi(120)
        st.pyplot(fig, use_container_width=True)

        # PNG download
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
        png_buf.seek(0)
        st.download_button("⬇️ Download figure (PNG)", png_buf,
                           file_name="shadow_flicker.png", mime="image/png")
        plt.close(fig)

        # ── Cloud-corrected map (if available) ───────────────────────────────
        if flicker_corrected is not None:
            st.divider()
            st.markdown("**Cloud-corrected estimate (indicative, not NEPC compliance)**")
            fig_c = plot_flicker_results(
                wtg_xy, flicker_corrected, flicker_max_day * (sum(cloud_data.values())/12),
                xx, yy, contour_levels, int(epsg_code),
                use_satellite=use_satellite, alpha_fill=alpha_fill,
                hub_height=hub_height, rotor_diameter=rotor_diameter, year=int(year),
                receptor_xy=receptor_xy,
                receptor_annual=(receptor_annual * (sum(cloud_data.values())/12)
                                 if receptor_annual is not None else None),
                receptor_max_day=receptor_max_day_arr,
                receptor_names=receptor_names,
            )
            fig_c.set_dpi(120)
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)

            # Monthly breakdown table
            months = ["Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec"]
            st.markdown("**Monthly clearness index (NASA POWER)**")
            kt_rows = [{"Month": months[m-1],
                        "Clearness index (KT)": round(cloud_data.get(m, 1.0), 3)}
                       for m in range(1, 13)]
            st.dataframe(pd.DataFrame(kt_rows), use_container_width=True, hide_index=True)

        # ── Receptor results table ────────────────────────────────────────────
        if receptor_annual is not None:
            st.divider()
            st.markdown("**Sensitive Receptor Shadow Flicker Results**")
            rows = []
            for i, name in enumerate(receptor_names):
                ann  = float(receptor_annual[i])
                mday = float(receptor_max_day_arr[i]) * 60  # → minutes
                ann_ok  = "✅" if ann  <= 30  else "❌"
                day_ok  = "✅" if mday <= 30  else "❌"
                rows.append({
                    "Receptor":        name,
                    "Annual (hr/yr)":  round(ann, 1),
                    "≤30 hr/yr":       ann_ok,
                    "Max day (min)":   round(mday, 0),
                    "≤30 min/day":     day_ok,
                })
            st.dataframe(pd.DataFrame(rows).set_index("Receptor"),
                         use_container_width=True)

        # ── Grid stats ────────────────────────────────────────────────────────
        st.divider()
        nepc_dist = blade_chord * NEPC_CHORD_MULTIPLIER
        st.markdown(f"**Contour Radii from Layout Centroid**  "
                    f"*(NEPC assessment boundary: {nepc_dist:.0f} m)*")
        centroid   = wtg_xy.mean(axis=0)
        flat       = flicker_annual.ravel()
        grid_pts   = np.column_stack([xx.ravel(), yy.ravel()])
        r_from_cen = np.sqrt(((grid_pts - centroid) ** 2).sum(axis=1))
        stat_rows  = []
        for lv in contour_levels:
            m    = flat >= lv
            r_lv = float(r_from_cen[m].max()) if m.any() else 0.0
            stat_rows.append({"Contour (hr/yr)": lv,
                               "Max radius (m)": round(r_lv, 0)})
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True,
                     hide_index=True)

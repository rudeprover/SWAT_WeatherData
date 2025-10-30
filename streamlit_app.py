"""
🌦 Climate Data Utility — SWAT Builder & NetCDF Converter
Created by Praveen Kalura (2025)
"""

import streamlit as st
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import os, tempfile, zipfile, io, shutil
from shapely.geometry import Polygon
from geopandas import GeoSeries
from datetime import date
import warnings
warnings.filterwarnings("ignore")

try:
    import rarfile
except ImportError:
    rarfile = None


# =============================================================================
# COMMON HELPERS
# =============================================================================
def insert_header(originalfile, string):
    """Insert a header line at the beginning of a file."""
    with open(originalfile, "r") as f:
        data = f.read()
    with open(originalfile, "w") as f:
        f.write(string + "\n" + data)


def extract_boundary(uploaded_file, temp_dir):
    """Handle .zip, .geojson, or .rar shapefile uploads with reprojection."""
    ext = uploaded_file.name.lower()

    if ext.endswith(".geojson"):
        path = os.path.join(temp_dir, "boundary.geojson")
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        shp = gpd.read_file(path)

    elif ext.endswith(".zip"):
        shp_dir = os.path.join(temp_dir, "shp")
        os.makedirs(shp_dir, exist_ok=True)
        with zipfile.ZipFile(uploaded_file, "r") as z:
            z.extractall(shp_dir)
        shp_files = [os.path.join(shp_dir, f) for f in os.listdir(shp_dir) if f.endswith(".shp")]
        if not shp_files:
            st.error("No .shp found in ZIP.")
            st.stop()
        shp = gpd.read_file(shp_files[0])

    elif ext.endswith(".rar") and rarfile:
        shp_dir = os.path.join(temp_dir, "shp_rar")
        os.makedirs(shp_dir, exist_ok=True)
        rf = rarfile.RarFile(uploaded_file)
        rf.extractall(shp_dir)
        shp_files = [os.path.join(shp_dir, f) for f in os.listdir(shp_dir) if f.endswith(".shp")]
        if not shp_files:
            st.error("No .shp found in RAR.")
            st.stop()
        shp = gpd.read_file(shp_files[0])
    else:
        st.error("Unsupported file format. Please upload .zip, .geojson, or .rar")
        st.stop()

    # Auto reproject if needed
    if shp.crs is not None and shp.crs.to_epsg() != 4326:
        shp = shp.to_crs(epsg=4326)
    return shp


def normalize_lon_if_needed(da):
    """If longitudes are 0–360, convert to -180–180."""
    try:
        if float(da.lon.max()) > 180:
            da = da.assign_coords(lon=((da.lon + 180) % 360) - 180)
    except Exception:
        pass
    return da


def human_metadata(ds, var_name):
    """Generate plain-text metadata summary for NetCDF file."""
    lines = []
    lines.append("NetCDF File Summary\n")
    lines.append("[1] Global Attributes:")
    if ds.attrs:
        for k, v in ds.attrs.items():
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("  (none)")
    lines.append("\n[2] Coordinates:")
    for cname, coord in ds.coords.items():
        try:
            rng = f"{coord.values[0]} … {coord.values[-1]} (n={coord.size})" if coord.size > 1 else f"{coord.values} (n=1)"
        except Exception:
            rng = f"n={coord.size}"
        units = getattr(coord, "attrs", {}).get("units", "N/A")
        lines.append(f"  - {cname} [units={units}]: {rng}")
    lines.append("\n[3] Variables:")
    for vname, dv in ds.data_vars.items():
        dims = ", ".join(dv.dims)
        units = dv.attrs.get("units", "N/A")
        lname = dv.attrs.get("long_name", dv.attrs.get("standard_name", vname))
        lines.append(f"  - {vname} (dims: {dims}; units: {units}; name: {lname})")
    lines.append(f"\n[4] Selected Variable: {var_name}")
    return "\n".join(lines)


# =============================================================================
# APP SHELL
# =============================================================================

st.set_page_config(page_title="🌦 Climate Data Utility — SWAT Builder & NetCDF Converter",
                   page_icon="🌦️", layout="wide")

st.title("🌦 Climate Data Utility")
st.caption("SWAT Builder (RF & TMP) • NetCDF → CSV + Metadata")

mode = st.radio("Choose Mode:", ["SWAT Builder", "NC → CSV + Metadata"], horizontal=True)

# =============================================================================
# SWAT BUILDER MODE
# =============================================================================
if mode == "SWAT Builder":
    st.sidebar.header("📁 Input Data")
    shape_upload = st.sidebar.file_uploader("Upload Boundary (.zip / .geojson / .rar)", type=["zip", "geojson", "rar"])

    st.sidebar.header("📅 Date Range")
    start_date = st.sidebar.date_input("Start Date", date(1990, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(1995, 12, 31))

    st.subheader("🌦 Generate SWAT-Compatible Weather Data")

    # Fixed NetCDF paths (in same folder as app)
    RAINFALL_PATH = "RF25_ind1901_2024_rfp25.nc"
    MINTEMP_PATH = "Mintemp_MinT_1951_2024.nc"
    MAXTEMP_PATH = "Maxtemp_MaxT_1951_2024.nc"

    if not all(os.path.exists(p) for p in [RAINFALL_PATH, MINTEMP_PATH, MAXTEMP_PATH]):
        st.error("❌ Required NetCDF files not found in the app directory.")
        st.stop()

    if shape_upload and st.button("🚀 Generate SWAT Weather Data", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory() as temp_dir:
            shp = extract_boundary(shape_upload, temp_dir)

            out_rf = os.path.join(temp_dir, "SWAT", "RF")
            out_tmp = os.path.join(temp_dir, "SWAT", "TMP")
            os.makedirs(out_rf, exist_ok=True)
            os.makedirs(out_tmp, exist_ok=True)

            st.write("Processing rainfall and temperature data... please wait...")

            # ---------------- RAINFALL ----------------
            rain = xr.open_dataarray(RAINFALL_PATH)
            rain = normalize_lon_if_needed(rain)
            rain = rain.sel(time=slice(str(start_date), str(end_date)))

            res = abs(float(rain.lat[1] - rain.lat[0]))
            bbox = shp.total_bounds
            min_lat, max_lat = bbox[1] - res/2, bbox[3] + res/2
            min_lon, max_lon = bbox[0] - res/2, bbox[2] + res/2
            sliced_rain = rain.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))

            RF_stations = {"ID": [], "Name": [], "Lat": [], "Lon": []}
            sid = 1
            for i in range(sliced_rain.shape[1]):
                for j in range(sliced_rain.shape[2]):
                    lat = float(sliced_rain.lat[i].values)
                    lon = float(sliced_rain.lon[j].values)
                    cell = Polygon([
                        (lon - res/2, lat - res/2),
                        (lon + res/2, lat - res/2),
                        (lon + res/2, lat + res/2),
                        (lon - res/2, lat + res/2)
                    ])
                    gcell = gpd.GeoDataFrame({"geometry": [cell]}, crs="EPSG:4326")
                    try:
                        inter = gpd.overlay(shp, gcell, how="intersection")
                        if not inter.empty and inter.area[0] / gcell.area[0] > 0:
                            df = sliced_rain[:, i, j].to_dataframe(name="rf")
                            fname = os.path.join(out_rf, f"RF{sid}.csv")
                            df.to_csv(fname, header=False, index=False)
                            insert_header(fname, str(start_date).replace("-", ""))
                            RF_stations["ID"].append(sid)
                            RF_stations["Name"].append(f"RF{sid}")
                            RF_stations["Lat"].append(lat)
                            RF_stations["Lon"].append(lon)
                            sid += 1
                    except:
                        continue

            if RF_stations["ID"]:
                pd.DataFrame(RF_stations).to_csv(os.path.join(out_rf, "RF_station.csv"), index=False, header=False)

            # ---------------- TEMPERATURE ----------------
            tmax = xr.open_dataarray(MAXTEMP_PATH).sel(time=slice(str(start_date), str(end_date)))
            tmin = xr.open_dataarray(MINTEMP_PATH).sel(time=slice(str(start_date), str(end_date)))
            tmax = normalize_lon_if_needed(tmax)
            tmin = normalize_lon_if_needed(tmin)

            tres = abs(float(tmax.lat[1] - tmax.lat[0]))
            bbox_t = shp.total_bounds
            min_lat_t, max_lat_t = bbox_t[1] - tres/2, bbox_t[3] + tres/2
            min_lon_t, max_lon_t = bbox_t[0] - tres/2, bbox_t[2] + tres/2
            tmax_s = tmax.sel(lat=slice(min_lat_t, max_lat_t), lon=slice(min_lon_t, max_lon_t))
            tmin_s = tmin.sel(lat=slice(min_lat_t, max_lat_t), lon=slice(min_lon_t, max_lon_t))

            TMP_stations = {"ID": [], "Name": [], "Lat": [], "Lon": []}
            sid = 1
            for i in range(tmax_s.shape[1]):
                for j in range(tmax_s.shape[2]):
                    lat = float(tmax_s.lat[i].values)
                    lon = float(tmax_s.lon[j].values)
                    cell = Polygon([
                        (lon - tres/2, lat - tres/2),
                        (lon + tres/2, lat - tres/2),
                        (lon + tres/2, lat + tres/2),
                        (lon - tres/2, lat + tres/2)
                    ])
                    gcell = gpd.GeoDataFrame({"geometry": [cell]}, crs="EPSG:4326")
                    try:
                        inter = gpd.overlay(shp, gcell, how="intersection")
                        if not inter.empty and inter.area[0] / gcell.area[0] > 0:
                            df_max = tmax_s[:, i, j].to_dataframe(name="tmax")
                            df_min = tmin_s[:, i, j].to_dataframe(name="tmin")
                            df = pd.concat([df_max, df_min], axis=1).round(2)
                            fname = os.path.join(out_tmp, f"TMP{sid}.csv")
                            df.to_csv(fname, header=False, index=False)
                            insert_header(fname, str(start_date).replace("-", ""))
                            TMP_stations["ID"].append(sid)
                            TMP_stations["Name"].append(f"TMP{sid}")
                            TMP_stations["Lat"].append(lat)
                            TMP_stations["Lon"].append(lon)
                            sid += 1
                    except:
                        continue

            if TMP_stations["ID"]:
                pd.DataFrame(TMP_stations).to_csv(os.path.join(out_tmp, "Temp_station.csv"), index=False, header=False)

            # ---------------- ZIP OUTPUT ----------------
            zip_path = os.path.join(temp_dir, "SWAT_Data.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(os.path.join(temp_dir, "SWAT")):
                    for f in files:
                        fp = os.path.join(root, f)
                        zipf.write(fp, os.path.relpath(fp, os.path.join(temp_dir, "SWAT")))

            st.success("✅ SWAT weather data generated successfully!")
            with open(zip_path, "rb") as f:
                st.download_button("⬇️ Download SWAT_Data.zip", f.read(),
                                   file_name=f"SWAT_Data_{start_date}_{end_date}.zip",
                                   mime="application/zip",
                                   use_container_width=True)


# =============================================================================
# NC → CSV + METADATA MODE (UNCHANGED)
# =============================================================================
else:
    st.markdown("<style>[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.header("Upload NetCDF → CSV + Metadata")
        uploaded = st.file_uploader("Upload one or more NetCDF files", type=["nc", "nc4", "cdf"], accept_multiple_files=True)
        if uploaded:
            items = []
            for i, uf in enumerate(uploaded, 1):
                try:
                    ds = xr.open_dataset(uf)
                    vars_list = list(ds.data_vars.keys())
                    v = st.selectbox(f"Select variable for {uf.name}", vars_list, key=f"var_{i}")
                    preview = ds[v].to_dataframe(name=v).reset_index().head(8)
                    st.dataframe(preview, use_container_width=True)
                    items.append((uf.name, ds, v))
                except Exception as e:
                    st.error(f"{uf.name}: {e}")

            if items and st.button("📦 Convert & Download ZIP", type="primary", use_container_width=True):
                with tempfile.TemporaryDirectory() as td:
                    zpath = os.path.join(td, "NetCDF_to_CSV_Metadata.zip")
                    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for fname, ds, var in items:
                            base = os.path.splitext(os.path.basename(fname))[0]
                            df = ds[var].to_dataframe(name=var).reset_index()
                            for c in df.columns:
                                if np.issubdtype(df[c].dtype, np.datetime64):
                                    df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%dT%H:%M:%S")
                            buf = io.StringIO()
                            df.to_csv(buf, index=False)
                            zf.writestr(f"{base}__{var}.csv", buf.getvalue())
                            zf.writestr(f"{base}__{var}__metadata.txt", human_metadata(ds, var))

                    with open(zpath, "rb") as f:
                        st.download_button("⬇️ Download CSV + Metadata ZIP", f.read(),
                                           file_name="NetCDF_to_CSV_Metadata.zip",
                                           mime="application/zip", use_container_width=True)

    with col2:
        st.header("ℹ️ About — NetCDF → CSV + Metadata")
        st.markdown("""
        Convert any NetCDF variable to a tidy **CSV**, plus a plain-English **TXT metadata** summary.

        **Steps**
        1. Upload `.nc` files.
        2. Select variable for each file.
        3. Download a ZIP with both CSV and metadata.

        **Output**
        ```
        NetCDF_to_CSV_Metadata/
        ├── file__variable.csv
        └── file__variable__metadata.txt
        ```
        """)

import streamlit as st
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import os, tempfile, zipfile, io
from shapely.geometry import Polygon, Point
from geopandas import GeoSeries
from datetime import date
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Shared helpers
# ---------------------------

def detect_nc_type(filename: str) -> str:
    n = filename.lower()
    if any(x in n for x in ["rf", "rain", "precip", "pr", "ppt"]):
        return "rainfall"
    if any(x in n for x in ["tmin", "mint", "min_temp"]):
        return "tmin"
    if any(x in n for x in ["tmax", "maxt", "max_temp"]):
        return "tmax"
    return "unknown"

def insert_date_header(file_path, start_date):
    with open(file_path, "r") as f:
        content = f.read()
    with open(file_path, "w") as f:
        f.write(str(start_date).replace("-", "") + "\n" + content)

def normalize_lon_if_needed(da):
    # If longitudes are 0..360, convert to -180..180
    try:
        if float(da.lon.max()) > 180:
            da = da.assign_coords(lon=((da.lon + 180) % 360) - 180)
    except Exception:
        pass
    return da

def create_zip(directories, zipname):
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        for d in directories:
            for root, _, files in os.walk(d):
                for f in files:
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, os.path.dirname(d)))

# ---------------------------
# SWAT extraction helpers
# ---------------------------

def extract_nc_data(nc_path, shape_file, start_date, end_date, var_name, out_dir, prefix):
    """Extract a variable from a NetCDF into SWAT-style per-station CSVs."""
    ds = xr.open_dataset(nc_path)
    var = normalize_lon_if_needed(ds[var_name])

    # Bound the crop to a buffered bbox
    bounds = shape_file.total_bounds  # [minx, miny, maxx, maxy] in EPSG:4326
    cell_res = abs(float(var.lat[1] - var.lat[0])) if var.sizes.get("lat", 0) > 1 else 0.05
    buffer = cell_res * 2
    min_lat, max_lat = bounds[1] - buffer, bounds[3] + buffer
    min_lon, max_lon = bounds[0] - buffer, bounds[2] + buffer

    var = var.sel(lat=slice(min_lat, max_lat),
                  lon=slice(min_lon, max_lon),
                  time=slice(start_date, end_date))

    os.makedirs(out_dir, exist_ok=True)
    total = int(var.sizes.get("lat", 0) * var.sizes.get("lon", 0))
    done = 0
    progress = st.progress(0.0)
    stations = []

    if total == 0:
        progress.progress(1.0)
        return 0

    # Iterate cells
    for i in range(var.sizes["lat"]):
        for j in range(var.sizes["lon"]):
            lat_c = float(var.lat[i].values)
            lon_c = float(var.lon[j].values)

            # Point-in-polygon test (fast) using unary_union
            pt_gdf = gpd.GeoDataFrame({"geometry": [Point(lon_c, lat_c)]}, crs="EPSG:4326")
            if not pt_gdf.within(shape_file.unary_union).any():
                done += 1
                progress.progress(done / total)
                continue

            # Export series
            ts = var[:, i, j].to_pandas()
            out_csv = os.path.join(out_dir, f"{prefix}{len(stations)+1}.csv")
            ts.to_csv(out_csv, header=False, index=False)
            insert_date_header(out_csv, start_date)
            stations.append((len(stations)+1, f"{prefix}{len(stations)+1}", lat_c, lon_c))

            done += 1
            progress.progress(done / total)

    if stations:
        pd.DataFrame(stations, columns=["ID", "Name", "Lat", "Lon"]).to_csv(
            os.path.join(out_dir, f"{prefix}_station.csv"), index=False
        )

    progress.progress(1.0)
    return len(stations)

def merge_tmin_tmax(tmin_dir, tmax_dir, output_dir, start_date):
    """Combine TMIN & TMAX pairs to TMP# with 2 columns; write Temp_station.csv."""
    if not (os.path.isdir(tmin_dir) and os.path.isdir(tmax_dir)):
        return False
    os.makedirs(output_dir, exist_ok=True)

    # Match by ordinal — both lists were created in the same scanning order
    tmin_files = sorted([f for f in os.listdir(tmin_dir) if f.lower().startswith("tmin") and f.lower().endswith(".csv")])
    tmax_files = sorted([f for f in os.listdir(tmax_dir) if f.lower().startswith("tmax") and f.lower().endswith(".csv")])
    if not tmin_files or not tmax_files:
        return False

    # Load station lists (for lat/lon)
    tmin_list = pd.read_csv(os.path.join(tmin_dir, "TMIN_station.csv"))
    tmax_list = pd.read_csv(os.path.join(tmax_dir, "TMAX_station.csv"))
    if tmin_list.shape[1] == 4 and "Lat" not in tmin_list.columns:
        tmin_list.columns = ["ID", "Name", "Lat", "Lon"]
    if tmax_list.shape[1] == 4 and "Lat" not in tmax_list.columns:
        tmax_list.columns = ["ID", "Name", "Lat", "Lon"]

    n = min(len(tmin_files), len(tmax_files))
    temp_stations = []

    for k in range(n):
        fmin = os.path.join(tmin_dir, tmin_files[k])
        fmax = os.path.join(tmax_dir, tmax_files[k])
        df_min = pd.read_csv(fmin, header=None)
        df_max = pd.read_csv(fmax, header=None)
        df_combined = pd.concat([df_min, df_max], axis=1)  # col0=TMIN, col1=TMAX
        out_csv = os.path.join(output_dir, f"TMP{k+1}.csv")
        df_combined.to_csv(out_csv, header=False, index=False)
        insert_date_header(out_csv, start_date)

        # choose lat/lon from tmin (or average)
        lat = float(tmin_list.loc[k, "Lat"]) if "Lat" in tmin_list.columns else np.nan
        lon = float(tmin_list.loc[k, "Lon"]) if "Lon" in tmin_list.columns else np.nan
        temp_stations.append((k+1, f"TMP{k+1}", lat, lon))

    pd.DataFrame(temp_stations, columns=["ID", "Name", "Lat", "Lon"]).to_csv(
        os.path.join(output_dir, "Temp_station.csv"), index=False
    )
    return True

# ---------------------------
# NC→CSV helpers
# ---------------------------

def human_metadata(ds, var_name):
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

# ---------------------------
# App shell
# ---------------------------

st.set_page_config(page_title="🌦 Climate Data Utility — SWAT Builder & NetCDF Converter",
                   page_icon="🌦️", layout="wide")

st.title("🌦 Climate Data Utility")
st.caption("SWAT Builder (RF & TMP) • NetCDF → CSV + Metadata")

mode = st.radio("Choose Mode:", ["SWAT Builder", "NC → CSV + Metadata"], horizontal=True)

# ================= SWAT BUILDER =================
if mode == "SWAT Builder":
    st.sidebar.header("📁 Input Data")
    shape_upload = st.sidebar.file_uploader("Upload Boundary (ZIP shapefile or GeoJSON)", type=["zip", "geojson"])
    st.sidebar.header("📅 Date Range")
    start_date = st.sidebar.date_input("Start Date", date(1990, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(1995, 12, 31))

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Generate SWAT-Compatible Data")
        nc_files = [f for f in os.listdir() if f.lower().endswith(".nc")]
        if not nc_files:
            st.warning("No NetCDF files found in the app directory.")
        elif shape_upload:
            if st.button("🚀 Generate SWAT-Compatible Weather Data", type="primary", use_container_width=True):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # --- Read boundary
                    if shape_upload.name.endswith(".geojson"):
                        gpath = os.path.join(temp_dir, "boundary.geojson")
                        with open(gpath, "wb") as f:
                            f.write(shape_upload.getbuffer())
                        shp = gpd.read_file(gpath)
                    else:
                        shp_dir = os.path.join(temp_dir, "shp")
                        os.makedirs(shp_dir, exist_ok=True)
                        with zipfile.ZipFile(shape_upload, "r") as z:
                            z.extractall(shp_dir)
                        shp_files = []
                        for r, _, files in os.walk(shp_dir):
                            for x in files:
                                if x.lower().endswith(".shp"):
                                    shp_files.append(os.path.join(r, x))
                        if not shp_files:
                            st.error("No .shp file found in the ZIP.")
                            st.stop()
                        shp = gpd.read_file(shp_files[0])

                    if shp.crs is not None and shp.crs.to_epsg() != 4326:
                        st.warning("Reprojecting shapefile to EPSG:4326.")
                        shp = shp.to_crs(epsg=4326)

                    rf_dir   = os.path.join(temp_dir, "RF")
                    tmin_dir = os.path.join(temp_dir, "TMIN")
                    tmax_dir = os.path.join(temp_dir, "TMAX")
                    out_dirs = []

                    # Process each nc by type
                    for nc in nc_files:
                        kind = detect_nc_type(nc)
                        ds = xr.open_dataset(nc)
                        var_name = list(ds.data_vars.keys())[0]
                        if kind == "rainfall":
                            c = extract_nc_data(nc, shp, start_date, end_date, var_name, rf_dir, "RF")
                            if c > 0: out_dirs.append(rf_dir)
                        elif kind == "tmin":
                            c = extract_nc_data(nc, shp, start_date, end_date, var_name, tmin_dir, "TMIN")
                            if c > 0: out_dirs.append(tmin_dir)
                        elif kind == "tmax":
                            c = extract_nc_data(nc, shp, start_date, end_date, var_name, tmax_dir, "TMAX")
                            if c > 0: out_dirs.append(tmax_dir)
                        else:
                            st.info(f"Skipping (unknown type): {nc}")

                    # Merge Tmin/Tmax → TMP
                    if os.path.isdir(tmin_dir) and os.path.isdir(tmax_dir):
                        tmp_dir = os.path.join(temp_dir, "TMP")
                        if merge_tmin_tmax(tmin_dir, tmax_dir, tmp_dir, start_date):
                            out_dirs.append(tmp_dir)

                    if out_dirs:
                        zip_path = os.path.join(temp_dir, "SWAT_Data.zip")
                        create_zip(out_dirs, zip_path)
                        with open(zip_path, "rb") as f:
                            st.download_button("⬇️ Download SWAT Data", f.read(),
                                               file_name=f"SWAT_Data_{start_date}_{end_date}.zip",
                                               mime="application/zip", use_container_width=True)
                        st.success("✅ SWAT data generation complete.")
                    else:
                        st.warning("No data extracted. Check that boundary overlaps your NetCDF grid.")

    with col2:
        st.header("ℹ️ About — SWAT Builder")
        st.markdown("""
        Convert **rainfall** and **temperature (TMIN & TMAX)** NetCDFs into SWAT-ready inputs.

        **Steps**
        1. Upload a boundary (ZIP shapefile or GeoJSON).
        2. Select date range.
        3. Click *Generate SWAT-Compatible Weather Data*.

        **Output**
        ```
        SWAT_Data/
        ├── RF/
        │   ├── RF1.csv
        │   ├── RF2.csv
        │   └── RF_station.csv
        └── TMP/
            ├── TMP1.csv          # columns: TMIN, TMAX
            ├── TMP2.csv
            └── Temp_station.csv
        ```
        """)

# ================= NC → CSV + METADATA =================
else:
    # Hide sidebar in this mode
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
                            # ISO datetimes for CSV
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

        **How it works**
        - Upload one or more `.nc` files.
        - Pick a variable per file.
        - Download a single ZIP containing `filename__variable.csv` and `filename__variable__metadata.txt`.

        **Output**
        ```
        NetCDF_to_CSV_Metadata/
        ├── file__variable.csv
        └── file__variable__metadata.txt
        ```
        """)

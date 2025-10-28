import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from geopandas import GeoSeries
import os
import zipfile
import tempfile
from datetime import date
import io
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Helper functions
# ------------------------------

def build_human_metadata(ds, var_name):
    """Generate plain-language metadata summary."""
    lines = []
    lines.append("NetCDF File Summary\n")
    lines.append("[1] Global Attributes:")
    if ds.attrs:
        for k, v in ds.attrs.items():
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("  (none found)")
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
        dims = ", ".join(list(dv.dims))
        units = dv.attrs.get("units", "N/A")
        lname = dv.attrs.get("long_name", dv.attrs.get("standard_name", vname))
        lines.append(f"  - {vname} (dims: {dims}; units: {units}; name: {lname})")
    lines.append(f"\n[4] Selected Variable: {var_name}")
    return "\n".join(lines)


def insert_date_header(file_path, date_string):
    """Insert date header to SWAT CSV."""
    with open(file_path, "r") as f:
        content = f.read()
    with open(file_path, "w") as f:
        f.write(date_string + "\n" + content)


def process_large_climate_data(data_path, shape_file, start_date, end_date, data_type, output_dir):
    """Simplified SWAT data extraction with single progress indicator."""
    try:
        ds = xr.open_dataset(data_path, chunks={"time": 365, "lat": 50, "lon": 50})
        var_name = list(ds.data_vars)[0]
        data_array = ds[var_name]

        # Ensure longitude is -180 to 180 if 0-360
        if data_array.lon.max() > 180:
            data_array = data_array.assign_coords(lon=(((data_array.lon + 180) % 360) - 180))

        # Spatial bounds
        bounds = shape_file.total_bounds  # [minx, miny, maxx, maxy]
        buffer = abs(float(data_array.lat[1] - data_array.lat[0])) * 2
        min_lat, max_lat = bounds[1] - buffer, bounds[3] + buffer
        min_lon, max_lon = bounds[0] - buffer, bounds[2] + buffer

        sliced_data = data_array.sel(
            lat=slice(min_lat, max_lat),
            lon=slice(min_lon, max_lon),
            time=slice(start_date, end_date)
        )

        os.makedirs(output_dir, exist_ok=True)
        total_cells = sliced_data.shape[1] * sliced_data.shape[2]
        processed = 0
        station_details = []
        progress = st.progress(0)

        for i in range(sliced_data.shape[1]):
            for j in range(sliced_data.shape[2]):
                lon_c = float(sliced_data[0, i, j].lon.values)
                lat_c = float(sliced_data[0, i, j].lat.values)
                cell_poly = gpd.GeoDataFrame({'geometry': GeoSeries(Polygon([
                    (lon_c - 0.05, lat_c - 0.05), (lon_c + 0.05, lat_c - 0.05),
                    (lon_c + 0.05, lat_c + 0.05), (lon_c - 0.05, lat_c + 0.05)
                ]))}, crs="EPSG:4326")

                try:
                    intersect = gpd.overlay(shape_file, cell_poly, how='intersection')
                    if len(intersect) > 0:
                        ts = sliced_data[:, i, j].to_pandas()
                        file_path = os.path.join(output_dir, f"{data_type.upper()}_{len(station_details)+1}.csv")
                        ts.to_csv(file_path, header=False, index=False)
                        insert_date_header(file_path, start_date.replace("-", ""))
                        station_details.append((data_type.upper(), lat_c, lon_c))
                except Exception:
                    pass
                processed += 1
                progress.progress(processed / total_cells)

        if station_details:
            details_path = os.path.join(output_dir, f"{data_type}_station_list.csv")
            pd.DataFrame(station_details, columns=["ID", "Lat", "Lon"]).to_csv(details_path, index=False)
        progress.progress(1.0)
        return len(station_details)

    except Exception as e:
        st.error(f"Error: {e}")
        return 0


def create_zip_file(directories, zip_filename):
    """Bundle output directories into a single ZIP."""
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for directory in directories:
            for root, _, files in os.walk(directory):
                for f in files:
                    path = os.path.join(root, f)
                    arc = os.path.relpath(path, os.path.dirname(directory))
                    zipf.write(path, arc)


# ------------------------------
# Main App
# ------------------------------

def main():
    st.set_page_config(page_title="🌦️ Climate Data Utility — SWAT Builder & NetCDF Converter",
                       page_icon="🌦️", layout="wide")

    st.title("🌦️ Climate Data Utility")
    st.caption("Convert climate NetCDFs for SWAT input or export NetCDF variables to CSV + metadata.")

    mode = st.radio("Choose Mode:", ["SWAT Builder", "NC → CSV + Metadata"], horizontal=True)

    # ---------------- SWAT BUILDER ----------------
    if mode == "SWAT Builder":
        st.sidebar.header("📁 Input Data")
        shapefile_upload = st.sidebar.file_uploader("Upload Boundary (ZIP or GeoJSON)",
                                                    type=["zip", "geojson"])
        st.sidebar.header("📅 Date Range")
        start_date = st.sidebar.date_input("Start Date", value=date(1990, 1, 1))
        end_date = st.sidebar.date_input("End Date", value=date(1995, 12, 31))

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Generate SWAT-Compatible Data")
            nc_files = [f for f in os.listdir() if f.lower().endswith(".nc")]
            if not nc_files:
                st.warning("No NetCDF files found in the app directory.")
            elif shapefile_upload:
                if st.button("🚀 Generate SWAT-Compatible Weather Data", type="primary", use_container_width=True):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        shp_path = None

                        # --- GeoJSON ---
                        if shapefile_upload.name.endswith(".geojson"):
                            shp_path = os.path.join(temp_dir, "boundary.geojson")
                            with open(shp_path, "wb") as f:
                                f.write(shapefile_upload.getbuffer())
                            shape_file = gpd.read_file(shp_path)

                        # --- Shapefile ZIP ---
                        else:
                            shp_dir = os.path.join(temp_dir, "shapefile")
                            os.makedirs(shp_dir, exist_ok=True)
                            with zipfile.ZipFile(shapefile_upload, "r") as zip_ref:
                                zip_ref.extractall(shp_dir)

                            # recursive search for .shp
                            shp_files = []
                            for root, _, files in os.walk(shp_dir):
                                for f in files:
                                    if f.lower().endswith(".shp"):
                                        shp_files.append(os.path.join(root, f))
                            if not shp_files:
                                st.error("No .shp file found inside the ZIP (even after scanning subfolders).")
                                return
                            shape_file = gpd.read_file(shp_files[0])

                        # ✅ Reproject if not in lat/lon
                        if shape_file.crs is not None and shape_file.crs.to_epsg() != 4326:
                            st.warning(f"Reprojecting shapefile from {shape_file.crs} to EPSG:4326 for compatibility.")
                            shape_file = shape_file.to_crs(epsg=4326)

                        output_dirs = []
                        st.info("⏳ Processing datasets...")
                        for nc in nc_files:
                            ds_type = "rainfall" if "rain" in nc.lower() or "precip" in nc.lower() else "temperature"
                            out_dir = os.path.join(temp_dir, ds_type)
                            count = process_large_climate_data(nc, shape_file, str(start_date), str(end_date), ds_type, out_dir)
                            if count > 0:
                                output_dirs.append(out_dir)

                        if output_dirs:
                            zip_path = os.path.join(temp_dir, "SWAT_Data.zip")
                            create_zip_file(output_dirs, zip_path)
                            with open(zip_path, "rb") as f:
                                st.download_button("⬇️ Download SWAT Data Package",
                                                   f.read(),
                                                   file_name=f"SWAT_Data_{start_date}_{end_date}.zip",
                                                   mime="application/zip")
                            st.success("✅ SWAT data generation complete!")
                        else:
                            st.warning("No data generated. Check if shapefile overlaps dataset region.")

        with col2:
            st.header("ℹ️ About — SWAT Builder")
            st.markdown("""
            This module extracts **rainfall and temperature data** from uploaded NetCDF
            climate files and converts them into **SWAT-compatible input tables**.

            **Steps:**
            1. Upload a shapefile (ZIP) or GeoJSON boundary.
            2. Select a date range of interest.
            3. Click *Generate SWAT-Compatible Weather Data*.

            **Output Structure:**
            ```
            SWAT_Data/
            ├── rainfall/
            │   ├── RAINFALL_1.csv
            │   ├── RAINFALL_2.csv
            │   └── rainfall_station_list.csv
            ├── temperature/
            │   ├── TEMPERATURE_1.csv
            │   ├── TEMPERATURE_2.csv
            │   └── temperature_station_list.csv
            ```
            """)

    # ---------------- NC → CSV MODE ----------------
    else:
        st.markdown("<style>[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("Upload NetCDF → CSV + Metadata")
            uploaded = st.file_uploader("Upload NetCDF file(s)",
                                        type=["nc", "nc4", "cdf"],
                                        accept_multiple_files=True)
            if uploaded:
                files = []
                for i, uf in enumerate(uploaded, 1):
                    try:
                        ds = xr.open_dataset(uf)
                        var = st.selectbox(f"Select variable for {uf.name}", list(ds.data_vars.keys()), key=f"var_{i}")
                        preview = ds[var].to_dataframe(name=var).reset_index().head(6)
                        st.dataframe(preview)
                        files.append((uf.name, ds, var))
                    except Exception as e:
                        st.error(f"{uf.name}: {e}")
                if files and st.button("📦 Convert & Download ZIP", type="primary"):
                    with tempfile.TemporaryDirectory() as td:
                        zip_path = os.path.join(td, "NetCDF_to_CSV_Metadata.zip")
                        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                            for fname, ds, var in files:
                                base = os.path.splitext(fname)[0]
                                df = ds[var].to_dataframe(name=var).reset_index()
                                for c in df.columns:
                                    if np.issubdtype(df[c].dtype, np.datetime64):
                                        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%dT%H:%M:%S")
                                csv_buf = io.StringIO()
                                df.to_csv(csv_buf, index=False)
                                zf.writestr(f"{base}__{var}.csv", csv_buf.getvalue())
                                zf.writestr(f"{base}__{var}__metadata.txt", build_human_metadata(ds, var))
                        with open(zip_path, "rb") as f:
                            st.download_button("⬇️ Download CSV + Metadata ZIP",
                                               f.read(),
                                               file_name="NetCDF_to_CSV_Metadata.zip",
                                               mime="application/zip")

        with col2:
            st.header("ℹ️ About — NetCDF → CSV + Metadata")
            st.markdown("""
            This mode converts **NetCDF (.nc)** datasets into simple **CSV tables**
            along with a plain-language **metadata summary**.

            **How it works:**
            - Upload one or more `.nc` files.
            - Choose which variable (e.g., precipitation, temperature) to export.
            - Download a ZIP containing both data and metadata files.

            **Output Structure:**
            ```
            NetCDF_to_CSV_Metadata/
            ├── file__variable.csv
            ├── file__variable__metadata.txt
            └── ...
            ```
            """)


if __name__ == "__main__":
    main()

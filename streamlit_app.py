import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopandas import GeoSeries
import os
import zipfile
import tempfile
from datetime import datetime, date
import io
import textwrap
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Existing helpers (unchanged)
# ------------------------------

def insert_date_header(file_path, date_string):
    """Insert date header at the beginning of CSV file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        with open(file_path, 'w') as f:
            f.write(date_string + '\n')
            f.write(content)
    except Exception as e:
        st.error(f"Error inserting date header: {str(e)}")


def find_nc_files():
    """Find all NetCDF files in the app directory"""
    app_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    nc_files = []
    for file in os.listdir(app_dir):
        if file.lower().endswith('.nc'):
            full_path = os.path.join(app_dir, file)
            file_size_mb = os.path.getsize(full_path) / (1024**2)
            nc_files.append({
                'name': file,
                'path': full_path,
                'size_mb': file_size_mb
            })
    return nc_files


def load_shapefile_data(file_path):
    """Load shapefile or geojson data"""
    try:
        if file_path.lower().endswith('.geojson'):
            return gpd.read_file(file_path)
        elif file_path.lower().endswith('.shp'):
            return gpd.read_file(file_path)
        else:
            return gpd.read_file(file_path)
    except Exception as e:
        st.error(f"Error loading shapefile/geojson: {str(e)}")
        return None


def process_large_climate_data(data_path, shape_file, start_date, end_date, data_type, output_dir, file_name):
    """Process large climate data using chunked reading and spatial cropping"""
    try:
        st.info(f"Processing {file_name}...")
        ds = xr.open_dataset(data_path, chunks={'time': 365, 'lat': 50, 'lon': 50})
        var_name = list(ds.data_vars)[0]
        st.info(f"Variable detected: {var_name}")
        data_array = ds[var_name]
        bounds = shape_file.bounds
        resolution = abs(float(data_array.lat[1] - data_array.lat[0]))
        buffer = resolution * 2
        min_lat = bounds.miny[0] - buffer
        max_lat = bounds.maxy[0] + buffer
        min_lon = bounds.minx[0] - buffer
        max_lon = bounds.maxx[0] + buffer
        st.info(f"Cropping to study area: {min_lat:.3f}°N to {max_lat:.3f}°N, {min_lon:.3f}°E to {max_lon:.3f}°E")
        sliced_data = data_array.sel(
            lat=slice(min_lat, max_lat),
            lon=slice(min_lon, max_lon),
            time=slice(start_date, end_date)
        )
        original_size = data_array.nbytes / (1024**2)
        new_size = sliced_data.nbytes / (1024**2)
        st.success(f"Data size reduced from {original_size:.1f}MB to {new_size:.1f}MB")
        return process_climate_data_from_array(sliced_data, shape_file, start_date, end_date, data_type, output_dir, var_name)
    except Exception as e:
        st.error(f"Error processing {file_name}: {str(e)}")
        return 0


def process_climate_data_from_array(data_array, shape_file, start_date, end_date, data_type, output_dir, var_name):
    """Process climate data from xarray DataArray"""
    resolution = abs(float(data_array.lat[1] - data_array.lat[0]))
    station_ids = ['ID']
    station_names = ['Name']
    lats = ['Lat']
    lons = ['Lon']
    idx = 1
    os.makedirs(output_dir, exist_ok=True)
    total_cells = data_array.shape[1] * data_array.shape[2]
    if total_cells == 0:
        st.warning("No data cells found in the cropped area")
        return 0
    progress_bar = st.progress(0)
    processed_cells = 0
    for i in range(data_array.shape[1]):
        for j in range(data_array.shape[2]):
            lon_cent = float(data_array[0, i, j].lon.values)
            lat_cent = float(data_array[0, i, j].lat.values)
            lon1, lat1 = lon_cent - resolution/2, lat_cent - resolution/2
            lon3, lat3 = lon_cent + resolution/2, lat_cent - resolution/2
            lon4, lat4 = lon_cent + resolution/2, lat_cent + resolution/2
            lon2, lat2 = lon_cent - resolution/2, lat_cent + resolution/2
            grid_poly = gpd.GeoDataFrame({'geometry': GeoSeries(Polygon([(lon1, lat1), (lon3, lat3), (lon4, lat4), (lon2, lat2)]))})
            try:
                intersect_poly = gpd.overlay(shape_file, grid_poly, how='intersection')
                if len(intersect_poly) > 0:
                    fraction = intersect_poly.area[0] / grid_poly.area[0]
                    if fraction > 0:
                        time_series = data_array[:, i, j].to_pandas()
                        filename = os.path.join(output_dir, f'{data_type.upper()}{idx}.csv')
                        time_series.to_csv(filename, header=False, index=False)
                        file_start_date = str(start_date).replace('-', '')
                        insert_date_header(filename, file_start_date)
                        station_ids.append(idx)
                        station_names.append(f'{data_type.upper()}{idx}')
                        lats.append(lat_cent)
                        lons.append(lon_cent)
                        idx += 1
            except Exception:
                pass
            processed_cells += 1
            progress_bar.progress(processed_cells / total_cells)
    station_details = pd.DataFrame({'ID': station_ids, 'station_names': station_names, 'Lat': lats, 'Lon': lons})
    station_file = os.path.join(output_dir, f'{data_type.upper()}_station.csv')
    station_details.to_csv(station_file, header=False, index=False)
    return idx - 1


def create_zip_file(directories, zip_filename):
    """Create a zip file containing all the generated data"""
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for directory in directories:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(directory))
                    zipf.write(file_path, arcname)

# ------------------------------
# NEW HELPERS: NC ➜ CSV + METADATA
# ------------------------------

def _fmt(v):
    try:
        return f"{v}"
    except Exception:
        return repr(v)


def build_human_metadata(ds: xr.Dataset, var_name: str | None) -> str:
    """Create a simple-language TXT description of the NetCDF contents."""
    lines = []
    lines.append("NetCDF file summary (human readable)\n")
    # Global
    lines.append("[1] Global attributes:")
    if ds.attrs:
        for k, v in ds.attrs.items():
            lines.append(f"  - {k}: {_fmt(v)}")
    else:
        lines.append("  (no global attributes)")

    # Coordinates
    lines.append("\n[2] Coordinates and their ranges:")
    if ds.coords:
        for cname, coord in ds.coords.items():
            rng = None
            try:
                if coord.size > 1:
                    rng = f"{coord.values[0]} … {coord.values[-1]} (n={coord.size})"
                else:
                    rng = f"{coord.values} (n=1)"
            except Exception:
                rng = f"n={coord.size}"
            units = coord.attrs.get('units', 'NA') if hasattr(coord, 'attrs') else 'NA'
            lines.append(f"  - {cname} [units={units}]: {rng}")
    else:
        lines.append("  (no coords found)")

    # Variables
    lines.append("\n[3] Variables available:")
    if ds.data_vars:
        for vname, dv in ds.data_vars.items():
            dims = ", ".join(list(dv.dims))
            units = dv.attrs.get('units', 'NA')
            long_name = dv.attrs.get('long_name', dv.attrs.get('standard_name', vname))
            lines.append(f"  - {vname} (dims: {dims}; units: {units}; long_name: {long_name})")
    else:
        lines.append("  (no data variables)")

    # Selected variable details
    if var_name and var_name in ds.data_vars:
        dv = ds[var_name]
        lines.append(f"\n[4] Selected variable: {var_name}")
        lines.append(f"  - dtype: {dv.dtype}")
        lines.append(f"  - dims: {list(dv.dims)} with sizes {[int(dv.sizes[d]) for d in dv.dims]}")
        if dv.attrs:
            lines.append("  - attributes:")
            for k, v in dv.attrs.items():
                lines.append(f"      {k}: {_fmt(v)}")
        # Simple-language note
        lines.append("\n[5] How to read the CSV:")
        lines.append("  The CSV contains one row per unique index of the selected variable.")
        lines.append("  All dimension columns (e.g., time/lat/lon/level) appear first, followed by the value column.")
    else:
        lines.append("\n[4] No specific variable selected.")

    return "\n".join(lines) + "\n"


def dataset_to_csv_bytes(ds: xr.Dataset, var_name: str) -> bytes:
    """Flatten the selected variable to a tidy CSV (dims as columns + value)."""
    if var_name not in ds.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")
    da = ds[var_name]
    # Convert to DataFrame with indexes as columns
    df = da.to_dataframe(name=var_name).reset_index()
    # Ensure datetimes formatted nicely if present
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            df[c] = pd.to_datetime(df[c]).dt.strftime('%Y-%m-%dT%H:%M:%S')
    # Return bytes
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')


# ------------------------------
# MAIN APP (with tabs)
# ------------------------------

def main():
    st.set_page_config(page_title="SWAT Weather Data Builder", page_icon="🌦️", layout="wide")
    st.title("🌦️ SWAT Weather Data Builder")
    st.markdown("*Convert gridded climate data to SWAT input format — plus a NetCDF → CSV+Metadata utility.*")

    tab1, tab2 = st.tabs(["SWAT Builder", "NC ➜ CSV + Metadata"])

    # ---------------- TAB 1: Your existing SWAT flow ----------------
    with tab1:
        st.header("📊 Processing Configuration")
        nc_files = find_nc_files()

        st.sidebar.header("📁 Input Data (SWAT Builder)")
        if nc_files:
            st.sidebar.success(f"Found {len(nc_files)} NetCDF files:")
            for nc_file in nc_files:
                st.sidebar.write(f"• {nc_file['name']} ({nc_file['size_mb']:.1f}MB)")
        else:
            st.sidebar.error("No NetCDF files found in app directory!")
            st.error("Please place your .nc files in the same directory as this app")
            # We keep the rest of the UI but disable the run button

        # Shapefile/GeoJSON input
        st.sidebar.header("🗺️ Study Area Boundary")
        app_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        default_shapefiles = []
        for ext in ['.geojson', '.shp']:
            shp_files = [f for f in os.listdir(app_dir) if f.lower().endswith(ext)]
            default_shapefiles.extend(shp_files)
        shapefile_source = st.sidebar.radio(
            "Select shapefile source:",
            ["Upload new file", "Use file from directory"] if default_shapefiles else ["Upload new file"],
            key="swat_shp_src"
        )
        if shapefile_source == "Use file from directory" and default_shapefiles:
            selected_shapefile = st.sidebar.selectbox("Select shapefile:", default_shapefiles, key="swat_shp_sel")
            shapefile_path = os.path.join(app_dir, selected_shapefile)
            shapefile_upload = None
        else:
            shapefile_upload = st.sidebar.file_uploader(
                "Upload Shapefile (ZIP) or GeoJSON",
                type=['zip', 'geojson'],
                help="Upload a ZIP file containing shapefile or a GeoJSON file",
                key="swat_shp_upload"
            )
            shapefile_path = None

        st.sidebar.header("📅 Date Range")
        start_date = st.sidebar.date_input("Start Date", value=date(1990, 1, 1), help="Start date for data extraction", key="swat_start")
        end_date = st.sidebar.date_input("End Date", value=date(1995, 12, 31), help="End date for data extraction", key="swat_end")
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        col1, col2 = st.columns([2, 1])
        with col1:
            if nc_files:
                st.subheader("Files to Process:")
                for i, nc_file in enumerate(nc_files):
                    st.write(f"{i+1}. **{nc_file['name']}** ({nc_file['size_mb']:.1f}MB)")

            run_btn = st.button("🚀 Process All NetCDF Files", type="primary", use_container_width=True, key="swat_run")
            if run_btn:
                final_shapefile = shapefile_upload if shapefile_upload else shapefile_path
                if not final_shapefile:
                    st.error("Please provide a shapefile or GeoJSON!")
                    st.stop()
                if not nc_files:
                    st.error("No NetCDF files found!")
                    st.stop()
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        st.info("📦 Loading boundary data...")
                        if isinstance(final_shapefile, str):
                            shape_file = load_shapefile_data(final_shapefile)
                        else:
                            if final_shapefile.name.endswith('.geojson'):
                                geojson_path = os.path.join(temp_dir, "boundary.geojson")
                                with open(geojson_path, "wb") as f:
                                    f.write(final_shapefile.getbuffer())
                                shape_file = gpd.read_file(geojson_path)
                            else:
                                shp_dir = os.path.join(temp_dir, "shapefile")
                                os.makedirs(shp_dir, exist_ok=True)
                                with zipfile.ZipFile(final_shapefile, 'r') as zip_ref:
                                    zip_ref.extractall(shp_dir)
                                shp_files = [f for f in os.listdir(shp_dir) if f.endswith('.shp')]
                                if not shp_files:
                                    st.error("No .shp file found in the uploaded ZIP!")
                                    st.stop()
                                shp_path = os.path.join(shp_dir, shp_files[0])
                                shape_file = gpd.read_file(shp_path)
                        if shape_file is None:
                            st.error("Failed to load boundary data!")
                            st.stop()
                        st.success(f"✅ Boundary data loaded: {len(shape_file)} features")
                        bounds = shape_file.bounds
                        st.info(f"📍 Study area: {bounds.minx[0]:.3f}°E to {bounds.maxx[0]:.3f}°E, {bounds.miny[0]:.3f}°N to {bounds.maxy[0]:.3f}°N")

                        output_base = os.path.join(temp_dir, "SWAT_Data")
                        processed_datasets = []
                        total_stations = 0
                        for i, nc_file in enumerate(nc_files):
                            st.subheader(f"Processing File {i+1}/{len(nc_files)}: {nc_file['name']}")
                            file_output_dir = os.path.join(output_base, f"File_{i+1}_{nc_file['name'].replace('.nc', '')}")
                            filename_lower = nc_file['name'].lower()
                            if any(k in filename_lower for k in ['rain', 'precip', 'pcp', 'rf']):
                                data_type = "rainfall"
                            elif any(k in filename_lower for k in ['temp', 'tmax', 'tmin']):
                                data_type = "temperature"
                            else:
                                data_type = f"climate_var_{i+1}"
                            stations_created = process_large_climate_data(
                                nc_file['path'], shape_file, start_date_str, end_date_str,
                                data_type, file_output_dir, nc_file['name']
                            )
                            if stations_created > 0:
                                st.success(f"✅ Created {stations_created} stations from {nc_file['name']}")
                                processed_datasets.append(file_output_dir)
                                total_stations += stations_created
                            else:
                                st.warning(f"No stations created from {nc_file['name']}")
                        if processed_datasets:
                            st.info("📦 Creating download package...")
                            zip_path = os.path.join(temp_dir, "SWAT_Weather_Data.zip")
                            create_zip_file(processed_datasets, zip_path)
                            with open(zip_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download SWAT Data Package",
                                    data=f.read(),
                                    file_name=f"SWAT_Weather_Data_{start_date_str}_{end_date_str}.zip",
                                    mime="application/zip",
                                    use_container_width=True,
                                )
                            st.balloons()
                            st.success(f"🎉 Processing complete! Created {total_stations} total weather stations from {len(nc_files)} files.")
                        else:
                            st.warning("No weather stations were created from any files.")
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)

        with col2:
            st.header("ℹ️ About")
            st.markdown(
                """
                This app automatically processes all NetCDF files in the app directory and converts them to SWAT format.

                **Required:**
                - Place .nc files in app directory
                - Provide shapefile/GeoJSON boundary

                **Output:**
                - Separate folder for each NetCDF file
                - Station CSV files and location files
                - SWAT-compatible format
                """
            )
            st.header("📋 Instructions")
            st.markdown(
                """
                1. Place all .nc files in app directory
                2. Upload shapefile or use local file
                3. Select date range
                4. Click "Process All NetCDF Files"
                5. Download results
                """
            )
            st.header("🔧 File Organization")
            st.markdown(
                """
                **Input directory:**
                ```
                app_folder/
                ├── swat_app.py
                ├── file1.nc
                ├── file2.nc
                └── boundary.geojson
                ```
                **Output structure:**
                ```
                SWAT_Data/
                ├── File_1_file1/
                ├── File_2_file2/
                └── ...
                ```
                """
            )

    # ---------------- TAB 2: NC ➜ CSV + METADATA ----------------
    with tab2:
        st.header("Upload NetCDF → get CSV and a human-readable TXT metadata")
        st.markdown("Useful when you simply need a tidy CSV of a variable and a plain-English summary of what's inside the .nc.")

        uploaded = st.file_uploader(
            "Upload one or more NetCDF files",
            type=["nc", "nc4", "cdf"],
            accept_multiple_files=True,
            key="u_nc"
        )

        if uploaded:
            # Per-file controls
            file_configs = []
            for i, uf in enumerate(uploaded, start=1):
                with st.expander(f"⚙️ Options: {uf.name}", expanded=False):
                    # Open in memory
                    try:
                        ds = xr.open_dataset(uf, engine=None)
                    except Exception as e:
                        st.error(f"Could not open {uf.name}: {e}")
                        continue

                    st.caption("Select a variable to export")
                    var_list = list(ds.data_vars.keys())
                    if not var_list:
                        st.warning("No data variables found in this file.")
                        continue
                    var_sel = st.selectbox("Variable", var_list, key=f"var_{i}")

                    # Preview dims and size
                    dv = ds[var_sel]
                    dims = list(dv.dims)
                    sizes = [int(dv.sizes[d]) for d in dims]
                    n_rows = int(np.prod(sizes))
                    st.write(f"**Dims:** {dims} → **Rows (flattened):** {n_rows:,}")
                    if n_rows > 2_000_000:
                        st.warning("This will create a very large CSV. Consider subsetting or proceed with caution.")

                    # Small preview
                    try:
                        df_preview = dv.to_dataframe(name=var_sel).reset_index().head(10)
                        st.dataframe(df_preview)
                    except Exception:
                        st.info("Preview not available for this variable.")

                    file_configs.append((uf, ds, var_sel))

            if file_configs and st.button("📦 Convert & Download ZIP", type="primary"):
                with st.spinner("Converting to CSV and writing metadata..."):
                    with tempfile.TemporaryDirectory() as td:
                        out_dir = os.path.join(td, "nc_exports")
                        os.makedirs(out_dir, exist_ok=True)
                        exported = 0
                        for uf, ds, var_sel in file_configs:
                            base = os.path.splitext(os.path.basename(uf.name))[0]
                            # CSV
                            try:
                                csv_bytes = dataset_to_csv_bytes(ds, var_sel)
                                csv_path = os.path.join(out_dir, f"{base}__{var_sel}.csv")
                                with open(csv_path, 'wb') as f:
                                    f.write(csv_bytes)
                            except Exception as e:
                                st.error(f"{uf.name}: CSV export failed — {e}")
                                continue
                            # TXT metadata
                            try:
                                txt = build_human_metadata(ds, var_sel)
                                txt_path = os.path.join(out_dir, f"{base}__{var_sel}__metadata.txt")
                                with open(txt_path, 'w', encoding='utf-8') as f:
                                    f.write(txt)
                            except Exception as e:
                                st.error(f"{uf.name}: metadata write failed — {e}")
                                continue
                            exported += 1
                        if exported:
                            zip_path = os.path.join(td, "nc_csv_metadata.zip")
                            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                                for root, _, files in os.walk(out_dir):
                                    for fn in files:
                                        p = os.path.join(root, fn)
                                        zf.write(p, arcname=os.path.relpath(p, out_dir))
                            with open(zip_path, 'rb') as f:
                                st.download_button(
                                    "⬇️ Download CSV+Metadata ZIP",
                                    data=f.read(),
                                    file_name="nc_csv_metadata.zip",
                                    mime="application/zip",
                                )
                            st.success(f"Done! Exported {exported} file(s).")
                        else:
                            st.warning("Nothing was exported.")
        else:
            st.info("Upload one or more .nc files to begin.")


if __name__ == "__main__":
    main()

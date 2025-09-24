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
import warnings
warnings.filterwarnings("ignore")

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
        
        # Open dataset with chunking to handle large files
        ds = xr.open_dataset(data_path, chunks={'time': 365, 'lat': 50, 'lon': 50})
        
        # Get the first data variable
        var_name = list(ds.data_vars)[0]
        st.info(f"Variable detected: {var_name}")
        data_array = ds[var_name]
        
        # Get bounds and calculate buffer
        bounds = shape_file.bounds
        resolution = abs(float(data_array.lat[1] - data_array.lat[0]))
        buffer = resolution * 2
        
        min_lat = bounds.miny[0] - buffer
        max_lat = bounds.maxy[0] + buffer
        min_lon = bounds.minx[0] - buffer
        max_lon = bounds.maxx[0] + buffer
        
        st.info(f"Cropping to study area: {min_lat:.3f}°N to {max_lat:.3f}°N, {min_lon:.3f}°E to {max_lon:.3f}°E")
        
        # Spatial and temporal slice
        sliced_data = data_array.sel(
            lat=slice(min_lat, max_lat),
            lon=slice(min_lon, max_lon),
            time=slice(start_date, end_date)
        )
        
        # Show size reduction
        original_size = data_array.nbytes / (1024**2)
        new_size = sliced_data.nbytes / (1024**2)
        st.success(f"Data size reduced from {original_size:.1f}MB to {new_size:.1f}MB")
        
        # Process the sliced dataset
        return process_climate_data_from_array(sliced_data, shape_file, start_date, end_date, data_type, output_dir, var_name)
        
    except Exception as e:
        st.error(f"Error processing {file_name}: {str(e)}")
        return 0

def process_climate_data_from_array(data_array, shape_file, start_date, end_date, data_type, output_dir, var_name):
    """Process climate data from xarray DataArray"""
    
    # Get spatial resolution
    resolution = abs(float(data_array.lat[1] - data_array.lat[0]))
    
    # Initialize station data
    station_ids = ['ID']
    station_names = ['Name']
    lats = ['Lat']
    lons = ['Lon']
    
    idx = 1
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Progress bar
    total_cells = data_array.shape[1] * data_array.shape[2]
    if total_cells == 0:
        st.warning("No data cells found in the cropped area")
        return 0
        
    progress_bar = st.progress(0)
    processed_cells = 0
    
    # Extract data within shapefile
    for i in range(data_array.shape[1]):
        for j in range(data_array.shape[2]):
            
            lon_cent = float(data_array[0, i, j].lon.values)
            lat_cent = float(data_array[0, i, j].lat.values)
            
            # Create grid polygon
            lon1, lat1 = lon_cent - resolution/2, lat_cent - resolution/2
            lon2, lat2 = lon_cent - resolution/2, lat_cent + resolution/2
            lon3, lat3 = lon_cent + resolution/2, lat_cent - resolution/2
            lon4, lat4 = lon_cent + resolution/2, lat_cent + resolution/2
            
            grid_poly = gpd.GeoDataFrame({
                'geometry': GeoSeries(Polygon([(lon1, lat1), (lon3, lat3), 
                                             (lon4, lat4), (lon2, lat2)]))
            })
            
            try:
                # Check intersection with shapefile
                intersect_poly = gpd.overlay(shape_file, grid_poly, how='intersection')
                
                if len(intersect_poly) > 0:
                    fraction = intersect_poly.area[0] / grid_poly.area[0]
                    
                    if fraction > 0:
                        # Extract time series data directly from array
                        time_series = data_array[:, i, j].to_pandas()
                        
                        # Save to CSV
                        filename = os.path.join(output_dir, f'{data_type.upper()}{idx}.csv')
                        time_series.to_csv(filename, header=False, index=False)
                        
                        # Insert date header
                        file_start_date = start_date.replace('-', '')
                        insert_date_header(filename, file_start_date)
                        
                        # Store station info
                        station_ids.append(idx)
                        station_names.append(f'{data_type.upper()}{idx}')
                        lats.append(lat_cent)
                        lons.append(lon_cent)
                        
                        idx += 1
                        
            except Exception as e:
                continue
            
            processed_cells += 1
            progress_bar.progress(processed_cells / total_cells)
    
    # Create station details file
    station_details = pd.DataFrame({
        'ID': station_ids,
        'station_names': station_names,
        'Lat': lats,
        'Lon': lons
    })
    
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

def main():
    st.set_page_config(
        page_title="SWAT Weather Data Builder",
        page_icon="🌦️",
        layout="wide"
    )
    
    st.title("🌦️ SWAT Weather Data Builder")
    st.markdown("*Convert gridded climate data to SWAT input format*")
    
    # Find NetCDF files in directory
    nc_files = find_nc_files()
    
    # Sidebar for inputs
    st.sidebar.header("📁 Input Data")
    
    # Show detected NetCDF files
    if nc_files:
        st.sidebar.success(f"Found {len(nc_files)} NetCDF files:")
        for nc_file in nc_files:
            st.sidebar.write(f"• {nc_file['name']} ({nc_file['size_mb']:.1f}MB)")
    else:
        st.sidebar.error("No NetCDF files found in app directory!")
        st.error("Please place your .nc files in the same directory as this app")
        return
    
    # Shapefile input
    st.sidebar.header("🗺️ Study Area Boundary")
    
    # Check for default shapefile
    app_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    default_shapefiles = []
    for ext in ['.geojson', '.shp']:
        shp_files = [f for f in os.listdir(app_dir) if f.lower().endswith(ext)]
        default_shapefiles.extend(shp_files)
    
    shapefile_source = st.sidebar.radio(
        "Select shapefile source:",
        ["Upload new file", "Use file from directory"] if default_shapefiles else ["Upload new file"]
    )
    
    if shapefile_source == "Use file from directory" and default_shapefiles:
        selected_shapefile = st.sidebar.selectbox(
            "Select shapefile:",
            default_shapefiles
        )
        shapefile_path = os.path.join(app_dir, selected_shapefile)
        shapefile_upload = None
    else:
        shapefile_upload = st.sidebar.file_uploader(
            "Upload Shapefile (ZIP) or GeoJSON", 
            type=['zip', 'geojson'],
            help="Upload a ZIP file containing shapefile or a GeoJSON file"
        )
        shapefile_path = None
    
    # Date selection
    st.sidebar.header("📅 Date Range")
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=date(1990, 1, 1),
        help="Start date for data extraction"
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=date(1995, 12, 31),
        help="End date for data extraction"
    )
    
    # Convert dates to strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Processing Configuration")
        
        # Show what will be processed
        if nc_files:
            st.subheader("Files to Process:")
            for i, nc_file in enumerate(nc_files):
                st.write(f"{i+1}. **{nc_file['name']}** ({nc_file['size_mb']:.1f}MB)")
        
        if st.button("🚀 Process All NetCDF Files", type="primary", use_container_width=True):
            
            # Check shapefile
            final_shapefile = shapefile_upload if shapefile_upload else shapefile_path
            if not final_shapefile:
                st.error("Please provide a shapefile or GeoJSON!")
                return
            
            if not nc_files:
                st.error("No NetCDF files found!")
                return
            
            try:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    
                    # Handle shapefile/geojson
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
                            # Handle ZIP file with shapefile
                            shp_dir = os.path.join(temp_dir, "shapefile")
                            os.makedirs(shp_dir, exist_ok=True)
                            
                            with zipfile.ZipFile(final_shapefile, 'r') as zip_ref:
                                zip_ref.extractall(shp_dir)
                            
                            shp_files = [f for f in os.listdir(shp_dir) if f.endswith('.shp')]
                            if not shp_files:
                                st.error("No .shp file found in the uploaded ZIP!")
                                return
                            
                            shp_path = os.path.join(shp_dir, shp_files[0])
                            shape_file = gpd.read_file(shp_path)
                    
                    if shape_file is None:
                        st.error("Failed to load boundary data!")
                        return
                    
                    st.success(f"✅ Boundary data loaded: {len(shape_file)} features")
                    bounds = shape_file.bounds
                    st.info(f"📍 Study area: {bounds.minx[0]:.3f}°E to {bounds.maxx[0]:.3f}°E, {bounds.miny[0]:.3f}°N to {bounds.maxy[0]:.3f}°N")
                    
                    # Create output directories
                    output_base = os.path.join(temp_dir, "SWAT_Data")
                    processed_datasets = []
                    total_stations = 0
                    
                    # Process each NetCDF file
                    for i, nc_file in enumerate(nc_files):
                        st.subheader(f"Processing File {i+1}/{len(nc_files)}: {nc_file['name']}")
                        
                        # Create subdirectory for this file
                        file_output_dir = os.path.join(output_base, f"File_{i+1}_{nc_file['name'].replace('.nc', '')}")
                        
                        # Determine data type based on filename
                        filename_lower = nc_file['name'].lower()
                        if any(keyword in filename_lower for keyword in ['rain', 'precip', 'pcp', 'rf']):
                            data_type = "rainfall"
                        elif any(keyword in filename_lower for keyword in ['temp', 'tmax', 'tmin']):
                            data_type = "temperature"
                        else:
                            data_type = f"climate_var_{i+1}"
                        
                        # Process the file
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
                    
                    # Create ZIP file
                    if processed_datasets:
                        st.info("📦 Creating download package...")
                        zip_path = os.path.join(temp_dir, "SWAT_Weather_Data.zip")
                        create_zip_file(processed_datasets, zip_path)
                        
                        # Provide download
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download SWAT Data Package",
                                data=f.read(),
                                file_name=f"SWAT_Weather_Data_{start_date_str}_{end_date_str}.zip",
                                mime="application/zip",
                                use_container_width=True
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
        st.markdown("""
        This app automatically processes all NetCDF files in the app directory and converts them to SWAT format.
        
        **Required:**
        - Place .nc files in app directory
        - Provide shapefile/GeoJSON boundary
        
        **Output:**
        - Separate folder for each NetCDF file
        - Station CSV files and location files
        - SWAT-compatible format
        """)
        
        st.header("📋 Instructions")
        st.markdown("""
        1. Place all .nc files in app directory
        2. Upload shapefile or use local file
        3. Select date range
        4. Click "Process All NetCDF Files"
        5. Download results
        """)
        
        st.header("🔧 File Organization")
        st.markdown("""
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
        """)

if __name__ == "__main__":
    main()

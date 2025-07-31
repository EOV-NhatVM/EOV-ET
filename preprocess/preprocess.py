import os 
import glob 
import numpy as np 
import xarray as xr 
import rioxarray as rxr     
from scipy.ndimage import generic_filter
import utils_climate
from datetime import datetime
import geopandas as gpd
from typing import Dict, List, Optional, Tuple


def local_mean_fill(da: xr.DataArray, size: int = 3) -> xr.DataArray:
    """
    Fill NaN values in a DataArray using local mean.
    
    Parameters:
    da (xr.DataArray): Input DataArray with NaN values.
    size (int): Size of the local neighborhood to compute the mean.
    
    Returns:
    xr.DataArray: DataArray with NaN values filled by local mean.
    """
    arr = da.values
    nan_mask = np.isnan(arr)

    def nanmean_filter(x):
        x = x[~np.isnan(x)]
        return np.mean(x) if x.size > 0 else np.nan

    local_means = generic_filter(arr, nanmean_filter, size=size, mode='mirror')
    
    filled_arr = arr.copy()
    filled_arr[nan_mask] = local_means[nan_mask]
    
    # Gán lại vào DataArray, giữ nguyên tọa độ
    return xr.DataArray(filled_arr, coords=da.coords, dims=da.dims, attrs=da.attrs)


class PreprocessStaticData:

    def __init__(self, DATA_ROOT: str):
        self.DATA_ROOT = DATA_ROOT
    
    def preprocess_static_data(self) -> xr.DataArray:
        """
        Preprocess static data from a given path.
        
        Returns:
        xr.DataArray: Preprocessed static data.
        """
        # Đọc DEM tĩnh
        dem_files = {
            "z": f"{self.DATA_ROOT}/dem/elevation_2000-01-01.tif",
            "slope": f"{self.DATA_ROOT}/dem/slope_2000-01-01.tif",
            "aspect": f"{self.DATA_ROOT}/dem/aspect_2000-01-01.tif",
            "t_amp": f"{self.DATA_ROOT}/t_amp/temp_amplitude_annual_2023-01-01.tif",  
        }

        static_vars = {}

        # Lấy template từ một file để resample các biến khác
        list_sm = glob.glob(f"{self.DATA_ROOT}/sm/sm_for_ndvi_*.tif")
        if not list_sm:
            raise FileNotFoundError("No soil moisture files found in the specified directory.")
        
        # Sử dụng file soil moisture đầu tiên làm template
        template = rxr.open_rasterio(list_sm[0], masked=True).squeeze()

        for var, path in dem_files.items():
            da = rxr.open_rasterio(path, masked=True).squeeze()

            if var == 'slope':
                da = utils_climate.degree_to_radian(da)

            da = utils_climate.resample_to_100m_match_template(da, template)

            # Fill NaN bằng local mean
            da = local_mean_fill(da)

            static_vars[var] = da

        return static_vars    

    


class ClimateDataProcessor:
    """
    A class to process climate data including NDVI, ERA5, albedo, and soil moisture data.
    """
    
    def __init__(self, data_root: str, region: str = 'TayNguyen'):
        """
        Initialize the ClimateDataProcessor.
        
        Args:
            data_root (str): Root directory path for data files
            region (str): Region name for processing
        """
        self.DATA_ROOT = data_root
        self.region = region
        self.template = None  # Should be set externally if needed
        
        # Variable mapping from internal names to file prefixes
        self.var_map = {
            "t_air_24": "temperature_2m",
            "t_air_min_24": "temperature_2m_min",
            "t_air_max_24": "temperature_2m_max",
            "t_dew_24": "dewpoint_temperature_2m",
            "p_air_0_24": "surface_pressure",
            "u2m_24": "u_component_of_wind_10m",
            "v2m_24": "v_component_of_wind_10m",
            "p_24": "total_precipitation_sum",
            "ra_flat_24": "surface_net_solar_radiation_sum"
        }
        
        # Initialize data variables structure
        self.data_vars = {
            "t_air_24": [],
            "t_air_min_24": [],
            "t_air_max_24": [],
            "t_dew_24": [],
            "p_air_0_24": [],
            "u2m_24": [],
            "v2m_24": [],
            "p_24": [],
            "ra_flat_24": [],
            "albedo": [],
            "ndvi": [],
            "se_root": []
        }
        
        self.valid_dates = []
    
        # template resample
        list_sm = glob.glob(f"{self.DATA_ROOT}/sm/sm_for_ndvi_*.tif")
        if not list_sm:
            raise FileNotFoundError("No soil moisture files found in the specified directory.")
        
        self.template = rxr.open_rasterio(list_sm[0], masked=True).squeeze()

    def get_list_date(self, folder_ndvi: str) -> List[str]:
        """
        Get list of dates from NDVI folder.
        
        Args:
            folder_ndvi (str): Path to folder containing NDVI images
            
        Returns:
            List[str]: Sorted list of dates in YYYY-MM-DD format
        """
        list_name = os.listdir(folder_ndvi)
        list_dates = []
        
        for name in list_name:
            date_str = name.split('_')[-1].split('.')[0]
            list_dates.append(date_str)
        
        # Parse strings to datetime objects for accurate sorting
        list_dates_sorted = sorted(
            list_dates, 
            key=lambda d: datetime.strptime(d, '%Y-%m-%d')
        )
        
        return list_dates_sorted
    
    def load_var(self, name: str, date: str, prefix: str = "era5") -> Optional[xr.DataArray]:
        """
        Load a variable for a specific date.
        
        Args:
            name (str): Variable name
            date (str): Date in YYYY-MM-DD format
            prefix (str): Data prefix (default: "era5")
            
        Returns:
            Optional[xr.DataArray]: Loaded and processed data array, None if not found
        """
        f = glob.glob(f"{self.DATA_ROOT}/{prefix}/{date}/{name}_{date}.tif")
        if not f:
            print(f"⚠️ Missing: {prefix}/{date}/{name}_{date}.tif")
            return None
        
        da = rxr.open_rasterio(f[0], masked=True).squeeze()
        
        # Resample if template is available
        if self.template is not None:
            da = utils_climate.resample_to_100m_match_template(da, self.template)
        
        # Apply unit conversions based on variable type
        if name in ['temperature_2m', 'temperature_2m_min', 'temperature_2m_max', 'dewpoint_temperature_2m']:
            return utils_climate.kevin_to_celcius(da)
        elif name == 'surface_net_solar_radiation_sum':
            return utils_climate.jun_to_mjun(da)
        elif name == 'surface_pressure':
            return utils_climate.pa_to_kpa(da)
        else:
            return da
    
    
    def process_all_dates(self, list_dates: List[str]) -> None:
        """
        Process all dates and load all required variables.
        
        Args:
            list_dates (List[str]): List of dates to process
        """
        
        self.valid_dates = []

        for date in list_dates:
            day_path = os.path.join(self.DATA_ROOT, "era5", date)
            if not os.path.exists(day_path):
                print(f"❌ Era5 path not found for {date}")
                continue

            skip_date = False
            for var_key, filename_prefix in self.var_map.items():
                da = self.load_var(filename_prefix, date)
                if da is not None:
                    if not isinstance(da, xr.DataArray):
                        da = xr.DataArray(da)
                    self.data_vars[var_key].append(local_mean_fill(da))
                else:
                    skip_date = True
                    break

            albedo_file = f"{self.DATA_ROOT}/albedo_s2/albedo_{date}.tif"
            if os.path.exists(albedo_file):
                albedo_da = rxr.open_rasterio(albedo_file, masked=True).squeeze()
                albedo_da = utils_climate.resample_to_100m_match_template(albedo_da, self.template)
                self.data_vars["albedo"].append(local_mean_fill(albedo_da))
            else:
                print(f"⚠️ Albedo file not found for {date}")
                skip_date = True


            ndvi_file = f"{self.DATA_ROOT}/ndvi/ndvi8days_infer_{date}.tif"
            if os.path.exists(ndvi_file):
                ndvi_da = rxr.open_rasterio(ndvi_file, masked=True).squeeze()
                ndvi_da = utils_climate.resample_to_100m_match_template(ndvi_da, self.template)
                self.data_vars["ndvi"].append(local_mean_fill(ndvi_da))
            else:
                print(f"⚠️ NDVI file not found for {date}")
                skip_date = True

            
            sm_file = f"{self.DATA_ROOT}/sm/sm_for_ndvi_{date}.tif"
            if os.path.exists(sm_file):
                sm_da = rxr.open_rasterio(sm_file, masked=True).squeeze()
                sm_da = sm_da.fillna(1)
                self.data_vars["se_root"].append(sm_da)

                # test sm ful 1
                # sm_da = rxr.open_rasterio(sm_file, masked=True).squeeze()
                # sm_da.data[:] = 1 
                # data_vars["se_root"].append(sm_da)
            else:
                print(f"⚠️ NDVI file not found for {date}")
                skip_date = True

            if not skip_date:
                self.valid_dates.append(date)
    
    def get_region_centroid(self, geojson_path: str) -> Tuple[float, float]:
        """
        Get the centroid coordinates of a region from GeoJSON file.
        
        Args:
            geojson_path (str): Path to GeoJSON file
            
        Returns:
            Tuple[float, float]: Longitude and latitude of centroid
        """
        gdf = gpd.read_file(geojson_path)
        centroid = gdf.geometry.centroid.iloc[0]
        lon, lat = centroid.x, centroid.y
        
        print(f"🧭 Tọa độ trung tâm: (lon: {lon}, lat: {lat})")
        return lon, lat
    
    def prepare_data_input(self, static_vars: Dict, index: int, date: str) -> xr.Dataset:
        """
        Prepare input dataset for a specific date and index.
        
        Args:
            static_vars (Dict): Dictionary of static variables
            index (int): Index in the data arrays
            date (str): Date in YYYY-MM-DD format
            
        Returns:
            xr.Dataset: Prepared dataset for model input
        """
        coords = {
            "x": static_vars["slope"].coords["x"],
            "y": static_vars["slope"].coords["y"],
            "time_bins": [np.datetime64(date)]
        }
        # print(self.data_vars)
        ds = xr.Dataset(
            data_vars=dict(
                doy=(("time_bins",), [utils_climate.date_to_doy(date)]),
                ndvi=(("y", "x"), self.data_vars["ndvi"][index].values),
                ra_flat_24=(("y", "x"), self.data_vars["ra_flat_24"][index].values),
                slope=(("y", "x"), static_vars["slope"].values),
                aspect=(("y", "x"), static_vars["aspect"].values),
                p_air_0_24=(("y", "x"), self.data_vars["p_air_0_24"][index].values),
                z=(("y", "x"), static_vars["z"].values),
                t_dew_24=(("y", "x"), self.data_vars["t_dew_24"][index].values),
                t_air_24=(("y", "x"), self.data_vars["t_air_24"][index].values),
                t_air_min_24=(("y", "x"), self.data_vars["t_air_min_24"][index].values),
                t_air_max_24=(("y", "x"), self.data_vars["t_air_max_24"][index].values),
                se_root=(("y", "x"), self.data_vars["se_root"][index].values),
                u2m_24=(("y", "x"), self.data_vars["u2m_24"][index].values),
                v2m_24=(("y", "x"), self.data_vars["v2m_24"][index].values),
                p_24=(("y", "x"), self.data_vars["p_24"][index].values),
                t_amp=(("y", "x"), static_vars["t_amp"].values),
                r0=(("y", "x"), self.data_vars["albedo"][index].values)
            ),
            coords=coords,
        )
        
        ds.encoding["source"] = f"/mnt/storage/code/pywapor-clms/res_{self.region}/{date}/res_{date}"
        
        return ds
    
    def set_template(self, template):
        """
        Set the template for resampling operations.
        
        Args:
            template: Template for resampling
        """
        self.template = template

# if __name__ == "__main__":
#     DATA_DIR = '/mnt/storage/code/pywapor-clms/dataset/dataset_TayNguyen'
#     static_processor = PreprocessStaticData(DATA_DIR)
#     static_vars = static_processor.preprocess_static_data()
#     climate_preprocessor = ClimateDataProcessor(data_root=DATA_DIR)

#     folder_ndvi = f"{DATA_DIR}/ndvi/"
#     list_dates = climate_preprocessor.get_list_date(folder_ndvi)

#     climate_preprocessor.process_all_dates(list_dates)
#     dates = climate_preprocessor.valid_dates

#     from pywapor_folder.et_look import main

#     # for idx,date in enumerate(dates):
#     date = '2023-01-01'
#     idx = 0 
#     print(f"Processing date: {date}")
#     ds_input = climate_preprocessor.prepare_data_input(static_vars, idx, date)
#     print(ds_input.values())
#     # main(ds_input, et_look_version='v3', export_vars="default", chunks={})
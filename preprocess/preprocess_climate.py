import utils_climate
import os
import xarray as xr
import rioxarray as rxr

class Preprocess_Climate:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.ds = xr.Dataset() 

        self.t_air_24 = 'temperature_2m'
        self.vp_24 = 'dewpoint_temperature_2m'
        self.p_air_24 = 'surface_pressure'
        self.u2m_24 = 'u_component_of_10m'
        self.v2m_24 = 'v_component_of_10m'
        self.t_air_min_24 = 'temperature_2m_min'
        self.t_air_max_24 = 'temperature_2m_max'
        self.ra_flat_24 = 'surface_net_solar_radiation_sum'
        self.r0 = 'surface_solar_radiation_downwards_sum'
    
    def resample_to_100m(raster_path, template_ds):
        ds = rxr.open_rasterio(raster_path, masked=True).squeeze()
        return ds.rio.reproject_match(template_ds)
    


    def solve(self):

        # list folder by date
        list_folder = os.listdir(self.folder_path)
        # for date in list_folder:
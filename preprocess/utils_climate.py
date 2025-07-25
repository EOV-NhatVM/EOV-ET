import numpy as np
from datetime import datetime
import rioxarray
from rasterio.enums import Resampling
import xarray as xr



def kevin_to_celcius(t_air):
    """convert kevin to celcius

    Args:
        t_air (float): temperature in K

    Returns:
        np.float: temperature in C
    """
    return t_air - 273.15

def jun_to_mjun(solar_radiation):
    """J to MJ

    Args:
        solar_radiation (float): value of solar radiation

    Returns:
        float: return value of solar radiation from J to MJ
    """
    return solar_radiation / 1000000.0


def jun_square_m_to_SI_unit(solar_radiation_downwards):
    """J/m^2 to W/m^2

    Args:
        solar_radiation_downwards (float): value of solar radiation shortwave

    Returns:
        float: return value of solar radiation shortwave from J/m^2 to W/m^2
    """
    return solar_radiation_downwards / 86400.0


def pa_to_kpa(surface_pressure):
    """onvert Pa to kPa for surface pressure

    Args:
        surface_pressure (float): value of surface pressure

    Returns:
        float: surface pressure in hPa
    """
    return surface_pressure / 1000.0


def date_to_doy(date: str): 
    """date string to doy

    Args:
        date (str): date of image

    Returns:
        int: return date of year
    """
    date = datetime.strptime(date, "%Y-%m-%d")
    return date.timetuple().tm_yday

def degree_to_radian(value):
    """degree to radian

    Args:
        value (array or numeric): values in degree

    Returns:
        float: values in radian
    """
    return np.pi * value / 180.0



def resample_to_100m_match_template(da: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    """
    Resample ảnh `da` theo ảnh `template` (100m) sử dụng reproject_match.
    
    Parameters
    ----------
    da : xr.DataArray
        Ảnh raster đầu vào cần resample.
    template : xr.DataArray
        Ảnh raster gốc có độ phân giải 100m dùng làm mẫu.

    Returns
    -------
    xr.DataArray
        Ảnh raster đã được resample theo template.
    """
    # Kiểm tra nếu thiếu CRS thì gán theo template
    if not da.rio.crs:
        da = da.rio.write_crs(template.rio.crs)

    # Nếu nhiều band thì lấy band đầu tiên
    if "band" in da.dims:
        da = da.isel(band=0)

    # Squeeze để bỏ chiều dư thừa
    da = da.squeeze()

    # Reproject để resample
    da_resampled = da.rio.reproject_match(template)

    return da_resampled

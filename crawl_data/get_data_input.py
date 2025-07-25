import ee
import geopandas as gpd
from shapely.geometry import Polygon
import time
import shutil
import os 
import rasterio
import logging 
import requests
import zipfile
from datetime import datetime
from pyproj import Transformer


def get_list_date(folder_ndvi):
    """get list date of NDVI in series 

    Args:
        folder_ndvi (type: str): path to folder contain list NDVI image. 

    Returns:
        dict: list of all date follow to NDVI date.
    """
    list_name = os.listdir(folder_ndvi)
    list_dates = []
    for name in list_name:
        date_str = name.split('_')[-1].split('.')[0]
        list_dates.append(date_str)
    # Parse strings to datetime objects for accurate sorting
    list_dates_sorted = sorted(list_dates, key=lambda d: datetime.strptime(d, '%Y-%m-%d'))
    return list_dates_sorted



def is_valid_tif(file_path):
    """checking if image is tiff file

    Args:
        file_path (str): path to file tiff

    Returns:
        boolean: return True if it's a tiff file, and False otherwise.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024:
        logging.error(f"Invalid: {file_path} - File missing or too small")
        return False
    try:
        with rasterio.open(file_path) as src:
            if src.count == 0 or src.width == 0 or src.height == 0:
                logging.error(f"Invalid: {file_path} - No bands or zero dimensions")
                return False
            return True
    except rasterio.errors.RasterioIOError as e:
        logging.error(f"Invalid: {file_path} - Rasterio error: {e}")
        return False
    



def download_band(image, band_name, roi, date_str, out_folder, max_retries=4):
    """_summary_

    Args:
        image (_type_): _description_
        band_name (_type_): _description_
        roi (_type_): _description_
        date_str (_type_): _description_
        out_folder (_type_): _description_
        max_retries (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    local_path = os.path.join(out_folder, f"{band_name}_{date_str}.tif")
    temp_dir = os.path.join(out_folder, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            url = image.select([band_name]).getDownloadURL({
                'scale': 10,
                'region': roi,
                'fileFormat': 'GeoTIFF',
                'maxPixels': 1e13,
                'expires': 3600
            })
            # print(f"Attempt {attempt+1} for {band_name} {date_str}: {url}")

            temp_zip_path = os.path.join(temp_dir, "download.zip")
            response = requests.get(url, stream=True, timeout=300, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
       

            response.raise_for_status()

            with open(temp_zip_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                # print(f"Zip file contents: {zip_ref.namelist()}")
                zip_ref.extractall(temp_dir)

            # print(f"Files in temp dir: {os.listdir(temp_dir)}")
            tif_files = [f for f in os.listdir(temp_dir) if f.endswith('.tif')]

            if len(tif_files) == 1:
                tif_file = tif_files[0]
                src_path = os.path.join(temp_dir, tif_file)
                shutil.copy(src_path, local_path)  # Use shutil.copy instead of os.replace to handle cross-device links
                if is_valid_tif(local_path):
                    # logging.info(f"Successfully downloaded {band_name} for {date_str}")
                    # Clean up temp directory after successful download
                    shutil.rmtree(temp_dir)
                    return local_path
                else:
                    logging.warning(f"⚠️ Invalid file for {band_name} {date_str}, retrying...")
                    time.sleep(1 * (attempt + 1))
            else:
                logging.warning(f"⚠️ Unexpected number of .tif files in zip: {len(tif_files)}")
                time.sleep(1 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            logging.error(f"⚠️ Request error for {band_name} {date_str}: {e}")
            time.sleep(1 * (attempt + 1))
        except Exception as e:
            logging.error(f"Error downloading {band_name} for {date_str}: {e}")
            time.sleep(1 * (attempt + 1))
    
    print(f"Failed to download {band_name} for {date_str} after {max_retries} attempts.")
    # Clean up temp directory on failure
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    return None


def monitor_tasks(tasks):
    """monitor tasking 

    Args:
        tasks (): get status of tasks
    """
    while any(task.active() for task in tasks):
        for task in tasks:
            status = task.status()
            print(f"Task '{status['description']}' is {status['state']}")
        time.sleep(10)
    for task in tasks:
        status = task.status()
        print(f"Task '{status['description']}' finished with state: {status['state']}")

def get_s2_albedo(start_date, end_date, region):
    """_summary_

    Args:
        start_date (_type_): _description_
        end_date (_type_): _description_
        region (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Sentinel-2 Surface Reflectance
    s2_sr = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterDate(start_date, end_date) \
        .filterBounds(region) \
        .map(lambda img: img.clip(region))

    def compute_albedo(image):
        albedo = image.expression(
            "0.356*B2 + 0.130*B3 + 0.373*B4 + 0.085*B8 + 0.072",
            {
                'B2': image.select('B2').divide(10000),
                'B3': image.select('B3').divide(10000),
                'B4': image.select('B4').divide(10000),
                'B8': image.select('B8').divide(10000)
            }
        ).rename('albedo')
        return image.addBands(albedo)

    return s2_sr.map(compute_albedo).select('albedo')

def is_within_days(date_str, list_dates, delta=4):
    """_summary_

    Args:
        date_str (_type_): _description_
        list_dates (_type_): _description_
        delta (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    target = datetime.strptime(date_str, "%Y-%m-%d")
    for d in list_dates:
        ref = datetime.strptime(d, "%Y-%m-%d")
        if abs((target - ref).days) <= delta:
            return True
    return False

def get_srtm_variables(region, out_folder, overwrite=False):
    """_summary_

    Args:
        region (_type_): _description_
        out_folder (_type_): _description_
        overwrite (bool, optional): _description_. Defaults to False.
    """
    os.makedirs(out_folder, exist_ok=True)

    # Lấy DEM từ SRTM
    dem = ee.Image("USGS/SRTMGL1_003")

    # Tính slope và aspect
    terrain = ee.Terrain.products(dem)
    slope = terrain.select("slope")
    aspect = terrain.select("aspect")
    elevation = dem.select("elevation")

    date_str = "2000-01-01"  # vì SRTM chỉ có 1 ảnh duy nhất, có thể gán tạm

    # Tải về 3 lớp
    for band, img in zip(["slope", "aspect", "elevation"], [slope, aspect, elevation]):
        out_path = download_band(
            img, band, region, date_str, out_folder
        )
        if out_path:
            print(f"✅ Downloaded {band} to {out_path}")
        else:
            print(f"❌ Failed to download {band}")


def get_temperature_amplitude(region, out_folder, year="2023", crs='EPSG:4326', scale=1000, overwrite=False):
    """_summary_

    Args:
        region (_type_): _description_
        out_folder (_type_): _description_
        year (str, optional): _description_. Defaults to "2023".
        crs (str, optional): _description_. Defaults to 'EPSG:4326'.
        scale (int, optional): _description_. Defaults to 1000.
        overwrite (bool, optional): _description_. Defaults to False.
    """
    os.makedirs(out_folder, exist_ok=True)

    # Lấy dữ liệu nhiệt độ hàng tháng từ ERA5 LAND MONTHLY
    dataset = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
        .filterBounds(region) \
        .filterDate(f"{year}-01-01", f"{year}-12-31") \
        .select("temperature_2m")

    # Tính nhiệt độ tối đa và tối thiểu trung bình trong năm
    max_temp = dataset.max().reproject(crs=crs, scale=scale).resample('bilinear')
    min_temp = dataset.min().reproject(crs=crs, scale=scale).resample('bilinear')

    # Biên độ nhiệt độ
    amplitude = max_temp.subtract(min_temp).rename("temp_amplitude_annual")

    # Tải ảnh về
    date_str = f"{year}-01-01"
    out_path = download_band(amplitude, "temp_amplitude_annual", region, date_str, out_folder)
    if out_path:
        print(f"✅ Downloaded temperature amplitude to {out_path}")
    else:
        print(f"❌ Failed to download temperature amplitude")



def main():
    ee.Authenticate()
    ee.Initialize(project='ee-minhnhat8dc')

    folder_ndvi = '/mnt/storage/code/pywapor-clms/dataset/ndvi/'
    list_dates = get_list_date(folder_ndvi)
    ndvi_file = os.path.join(folder_ndvi, f"ndvi8days_infer_{list_dates[0]}.tif")

    with rasterio.open(ndvi_file) as src:
        bounds = src.bounds
        crs = src.crs

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    ll = transformer.transform(bounds.left, bounds.bottom)
    lr = transformer.transform(bounds.right, bounds.bottom)
    ur = transformer.transform(bounds.right, bounds.top)
    ul = transformer.transform(bounds.left, bounds.top)
    coords = [ul, ur, lr, ll, ul]
    region = ee.Geometry.Polygon([coords])

    start_date = '2023-01-01'
    end_date = '2023-12-31'

    # ============================ ERA5 DOWNLOAD ================================
    era5_daily = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")\
        .filterDate(start_date, end_date)\
        .select([
            "temperature_2m", "dewpoint_temperature_2m", "surface_pressure",
            "u_component_of_wind_10m", "v_component_of_wind_10m",
            "temperature_2m_min", "temperature_2m_max",
            "surface_net_solar_radiation_sum", "surface_solar_radiation_downwards_sum",
            "total_precipitation_sum"
        ])
    out_era5 = "/mnt/storage/code/pywapor-clms/dataset/era5"
    os.makedirs(out_era5, exist_ok=True)

    images = era5_daily.toList(era5_daily.size())

    for i in range(images.size().getInfo()):
        image = ee.Image(images.get(i))
        date = image.date().format('YYYY-MM-dd').getInfo()

        if date in list_dates:
            output_folder = os.path.join(out_era5, date)
            os.makedirs(output_folder, exist_ok=True)
            print(f"📦 Processing ERA5 for date: {date}")

            for band_name in era5_daily.first().bandNames().getInfo():
                downloaded = download_band(image, band_name, region, date, output_folder)
                if downloaded:
                    print(f"✅ Downloaded: {downloaded}")
                else:
                    print(f"❌ Failed: {band_name} on {date}")

    # ============================ ALBEDO DOWNLOAD ================================
    s2_albedo_ic = get_s2_albedo(start_date, end_date, region)
    out_s2_albedo = "/mnt/storage/code/pywapor-clms/dataset/albedo_s2"
    os.makedirs(out_s2_albedo, exist_ok=True)

    albedo_list = s2_albedo_ic.toList(s2_albedo_ic.size())

    for i in range(albedo_list.size().getInfo()):
        image = ee.Image(albedo_list.get(i))
        date = image.date().format('YYYY-MM-dd').getInfo()

        if is_within_days(date, list_dates, delta=4):
            print(f"🌞 Processing S2 Albedo for date: {list_dates[i]}")
            downloaded = download_band(image.select('albedo'), 'albedo', region, list_dates[i], out_s2_albedo)
            if downloaded:
                print(f"✅ Downloaded S2 Albedo: {downloaded}")
            else:
                print(f"❌ Failed to download S2 Albedo for {date}")

    # =========================== DEM ============================================
    out_dem_folder = "/mnt/storage/code/pywapor-clms/dataset/dem"
    os.makedirs(out_dem_folder, exist_ok=True)
    get_srtm_variables(region, out_dem_folder)


    # =========================== T_AMP ==========================================
    year = "2023"
    output_folder = "/mnt/storage/code/pywapor-clms/dataset/t_amp"
    os.makedirs(output_folder, exist_ok=True)

    # Lấy biên độ nhiệt độ năm
    get_temperature_amplitude(region, output_folder, year=year, scale=1000, crs='EPSG:4326')


if __name__ == "__main__":
    main()

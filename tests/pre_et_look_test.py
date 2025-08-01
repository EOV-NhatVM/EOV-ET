import pywapor_folder
import datetime
from pywapor_folder.general.logger import adjust_logger
import numpy as np
import xarray as xr

def test_small_et_look(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    sources = dict()
    sources["ndvi"] = {'products': [{'source': 'SENTINEL2',
        'product_name': 'S2MSI2A_R60m',
        'enhancers': 'default',
        'is_example': True}],
    'temporal_interp': 'linear',
    'spatial_interp': 'nearest',
    'composite_type': 'mean'}

    sources["r0"] = {'products': [{'source': 'SENTINEL2',
        'product_name': 'S2MSI2A_R60m',
        'enhancers': 'default'}],
    'temporal_interp': 'linear',
    'spatial_interp': 'nearest',
    'composite_type': 'min'}

    sources["z"] = {'products': [{'source': 'COPERNICUS',
        'product_name': 'GLO30',
        'enhancers': 'default'}],
    'temporal_interp': 'linear',
    'spatial_interp': 'nearest',
    'composite_type': 'max'}

    latlim = [29.4, 29.7]
    lonlim = [30.7, 31.0]
    today = datetime.datetime.now().date()
    timelim = [today - datetime.timedelta(21), today - datetime.timedelta(21)]
    bin_length = 1
    ds = pywapor_folder.pre_et_look.main(folder, latlim, lonlim, timelim, sources, bin_length = bin_length)
    assert ds.rio.crs.to_epsg() == 4326
    assert "ndvi" in ds.data_vars
    assert "r0" in ds.data_vars
    assert "ndvi" in ds.data_vars
    assert ds.ndvi.min().values >= -1.
    assert ds.ndvi.max().values <= 1.

def test_et_look_level_2_v3(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    today = datetime.datetime.now().date()
    timelim = [today - datetime.timedelta(71), today - datetime.timedelta(71)]

    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]
    sources = "level_2_v3"
    bin_length = 1
    input_data = pywapor_folder.pre_et_look.main(folder, latlim, lonlim, timelim, 
                                            sources, bin_length = bin_length)
    assert input_data.rio.crs.to_epsg() == 4326
    assert np.all([x in input_data.data_vars for x in ["ndvi", "r0", "se_root"]])
    assert input_data.ndvi.min().values >= -1.
    assert input_data.ndvi.max().values <= 1.
    assert input_data.r0.min().values >= 0.
    assert input_data.r0.max().values <= 1.
    assert input_data.se_root.min().values >= 0.
    assert input_data.se_root.max().values <= 1.

    ds = pywapor_folder.et_look.main(input_data, et_look_version = "v3")
    assert ds.rio.crs.to_epsg() == 4326
    assert "aeti_24_mm" in ds.data_vars
    assert ds.aeti_24_mm.min().values >= 0.
    assert ds.aeti_24_mm.max().values <= 16.

def test_et_look_level_3_v3(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    today = datetime.datetime.now().date()
    timelim = [today - datetime.timedelta(35), today - datetime.timedelta(30)]
    # timelim = [datetime.date(2023, 2, 1), datetime.date(2023, 2, 11)]

    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]
    sources = "level_3_v3"
    bin_length = 2
    input_data = pywapor_folder.pre_et_look.main(folder, latlim, lonlim, timelim, 
                                            sources, bin_length = bin_length)
    assert input_data.rio.crs.to_epsg() == 4326
    assert np.all([x in input_data.data_vars for x in ["ndvi", "r0", "se_root"]])
    assert input_data.ndvi.min().values >= -1.
    assert input_data.ndvi.max().values <= 1.
    assert input_data.r0.min().values >= 0.
    assert input_data.r0.max().values <= 1.
    assert input_data.se_root.min().values >= 0.
    assert input_data.se_root.max().values <= 1.

    ds = pywapor_folder.et_look.main(input_data, et_look_version = "v3")
    assert ds.rio.crs.to_epsg() == 4326
    assert "aeti_24_mm" in ds.data_vars
    assert ds.aeti_24_mm.min().values >= 0.
    assert ds.aeti_24_mm.max().values <= 16.

def test_et_look_level_1_v2(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    timelim = [datetime.date(2021, 7, 1), datetime.date(2021, 7, 11)]
    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]
    sources = "level_1"
    input_data = pywapor_folder.pre_et_look.main(folder, latlim, lonlim, timelim, 
                                            sources)
    assert input_data.rio.crs.to_epsg() == 4326
    assert np.all([x in input_data.data_vars for x in ["ndvi", "r0", "se_root"]])
    assert input_data.ndvi.min().values >= -1.
    assert input_data.ndvi.max().values <= 1.
    assert input_data.r0.min().values >= 0.
    assert input_data.r0.max().values <= 1.
    assert input_data.se_root.min().values >= 0.
    assert input_data.se_root.max().values <= 1.

    ds = pywapor_folder.et_look.main(input_data, et_look_version = "v2")
    assert ds.rio.crs.to_epsg() == 4326
    assert "aeti_24_mm" in ds.data_vars
    assert ds.aeti_24_mm.min().values >= 0.
    assert ds.aeti_24_mm.max().values <= 16.

def test_et_look_level_2_v2(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    timelim = [datetime.date(2019, 4, 1), datetime.date(2019, 4, 11)]
    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]
    sources = "level_2"
    bin_length = "DEKAD"
    input_data = pywapor_folder.pre_et_look.main(folder, latlim, lonlim, timelim, sources, bin_length = bin_length)
    assert input_data.rio.crs.to_epsg() == 4326
    assert np.all([x in input_data.data_vars for x in ["ndvi", "r0", "se_root"]])
    assert input_data.ndvi.min().values >= -1.
    assert input_data.ndvi.max().values <= 1.
    assert input_data.r0.min().values >= 0.
    assert input_data.r0.max().values <= 1.
    assert input_data.se_root.min().values >= 0.
    assert input_data.se_root.max().values <= 1.

    ds = pywapor_folder.et_look.main(input_data, et_look_version = "v2")
    assert ds.rio.crs.to_epsg() == 4326
    assert "aeti_24_mm" in ds.data_vars
    assert ds.aeti_24_mm.min().values >= 0.
    assert ds.aeti_24_mm.max().values <= 16.

def test_et_look_level_3_v2(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    timelim = [datetime.date(2019, 4, 11), datetime.date(2019, 4, 21)]
    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]
    sources = "level_3"
    bin_length = "DEKAD"
    input_data = pywapor_folder.pre_et_look.main(folder, latlim, lonlim, timelim, sources, bin_length = bin_length)
    assert input_data.rio.crs.to_epsg() == 4326
    assert np.all([x in input_data.data_vars for x in ["ndvi", "r0", "se_root"]])
    assert input_data.ndvi.min().values >= -1.
    assert input_data.ndvi.max().values <= 1.
    assert input_data.r0.min().values >= 0.
    assert input_data.r0.max().values <= 1.
    assert input_data.se_root.min().values >= 0.
    assert input_data.se_root.max().values <= 1.

    ds = pywapor_folder.et_look.main(input_data, et_look_version = "v2")
    assert ds.rio.crs.to_epsg() == 4326
    assert "aeti_24_mm" in ds.data_vars
    assert ds.aeti_24_mm.min().values >= 0.
    assert ds.aeti_24_mm.max().values <= 16.

def test_appending(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    timelim = [datetime.date(2019, 4, 11), datetime.date(2019, 4, 15)]
    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]

    sources = "level_1"
    
    et_look_sources_lvl1 = pywapor_folder.general.levels.pre_et_look_levels(level = sources, bin_length = "DEKAD")
    sources1 = {k: v for k,v in et_look_sources_lvl1.items() if v["products"][0]["source"] == "GEOS5" and k in ["t_air", "u2m"]}
    ds1, _ = pywapor_folder.collect.downloader.collect_sources(folder,sources1,latlim, lonlim, timelim)

    sources2 = {k: v for k,v in et_look_sources_lvl1.items() if v["products"][0]["source"] == "GEOS5" and k in ["t_air", "qv", "v2m"]}
    ds2, _ = pywapor_folder.collect.downloader.collect_sources(folder,sources2,latlim, lonlim, timelim)

    assert "qv" in xr.open_dataset(ds2[('GEOS5', 'inst3_2d_asm_Nx')]).data_vars

if __name__ == "__main__":

    tmp_path = r"/Users/hmcoerver/Local/test_et_look_level_2_v30"

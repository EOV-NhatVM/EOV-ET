import pywapor_folder
import inspect
import datetime
import warnings
import rasterio
import numpy as np
from pywapor_folder.general.logger import adjust_logger, log
import pytest

def get_source_details(mod):
    # Parse the `default_vars`-function code content.
    lines = inspect.getsourcelines(mod.default_vars)

    # Read the lines from `default_vars` to substract the availabel products and variables.
    idx1 = [i for i, x in enumerate(lines[0]) if "variables = " in x][0]
    idx2 = [i - 1 for i, x in enumerate(lines[0]) if "req_dl_vars = " in x][0]
    ldict = {}
    cont = True
    iteration = 0
    while cont and iteration < 10:
        try:
            exec("".join(lines[0][idx1:idx2]).replace("  ", ""), globals(), ldict)
            cont = False
        except NameError as e:
            exec(f"{e.name} = lambda x: None", globals())
        iteration += 1
    variables = ldict["variables"]

    # Read the lines from `default_vars` to subtract the `req_dl_vars`
    idx1 = [i for i, x in enumerate(lines[0]) if "req_dl_vars = " in x][0]
    idx2 = [i - 1 for i, x in enumerate(lines[0]) if "out = " in x][0]
    ldict = {}
    exec("".join(lines[0][idx1:idx2]).replace("  ", ""), globals(), ldict)
    req_dl_vars = ldict["req_dl_vars"]

    # List the available products for this `source`.
    products = variables.keys()

    return req_dl_vars, products

def download_mod(mod, workdir, timelim, latlim, lonlim, products = None, req_dl_vars = None):

    if isinstance(products, type(None)) or isinstance(req_dl_vars, type(None)):
        req_dl_vars, products = get_source_details(mod)

    args = {
        "folder": workdir,
        "latlim": latlim,
        "lonlim": lonlim,
        "timelim": timelim, 
    }

    dss = {}

    for product_name in products:

        req_vars = list(req_dl_vars[product_name].keys())
        
        args.update({
            "product_name": product_name,
            "req_vars": req_vars,
        })

        for k, v in args.items():
            log.info(f"{k} = {v}")

        dss[product_name] = mod.download(**args)

    return dss

def has_geotransform(ds):
    varis = ds.data_vars
    for var in varis:
        with warnings.catch_warnings(record=True) as w:
            _ = rasterio.open(f"netcdf:{ds.encoding['source']}:{var}")
            if len(w) > 0:
                for warning in w:
                    no_geot = "Dataset has no geotransform, gcps, or rpcs." in str(warning.message)
                    if no_geot:
                        return False
    return True

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

sources = {
    'GEOS5':        [pywapor_folder.collect.product.GEOS5,     [datetime.date(2022, 3, 1), datetime.date(2022, 3, 3)]], # opendap.download_xarray
    'STATICS':      [pywapor_folder.collect.product.STATICS,   None], # cog.download
    'MODIS':        [pywapor_folder.collect.product.MODIS,     [datetime.date(2019, 3, 1), datetime.date(2019, 4, 1)]], # opendap.download
    'MERRA2':       [pywapor_folder.collect.product.MERRA2,    [datetime.date(2022, 3, 1), datetime.date(2022, 3, 3)]], # opendap.download
    'GLOBCOVER':    [pywapor_folder.collect.product.GLOBCOVER, None], # cog.download
    'CHIRPS':       [pywapor_folder.collect.product.CHIRPS,    [datetime.date(2022, 3, 1), datetime.date(2022, 3, 3)]], # opendap.download
    'SRTM':         [pywapor_folder.collect.product.SRTM,      None], # opendap.download
    'ERA5':         [pywapor_folder.collect.product.ERA5,      [datetime.date(2022, 3, 1), datetime.date(2022, 3, 3)]], # cds.download
    'SENTINEL2':    [pywapor_folder.collect.product.SENTINEL2, [datetime.date(2023, 3, 1), datetime.date(2023, 3, 3)]],    
    'SENTINEL3':    [pywapor_folder.collect.product.SENTINEL3, [datetime.date(2023, 3, 1), datetime.date(2023, 3, 3)]],
    'VIIRSL1':      [pywapor_folder.collect.product.VIIRSL1,   [datetime.date(2022, 3, 1), datetime.date(2022, 3, 2)]],
    'COPERNICUS':   [pywapor_folder.collect.product.COPERNICUS, None], # cog.download
    'TERRA':        [pywapor_folder.collect.product.TERRA,     [datetime.date(2020, 7, 2), datetime.date(2020, 7, 9)]],
    'LANDSAT': [pywapor_folder.collect.product.LANDSAT, [datetime.date(2022, 3, 1), datetime.date(2022, 3, 12)]]
}

@pytest.mark.parametrize("product_name", list(sources.keys()))
def test_dl(product_name, tmp_path):

    folder = workdir = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    (mod, timelim) = sources[product_name]

    log.info(f"==> {product_name} <==")

    latlim = [29.4, 29.7]
    lonlim = [30.7, 31.0]

    if product_name == "LANDSAT":
        products = ["LE07_SR", "LC08_SR", "LC09_SR"]
        req_dl_vars = {k: {"ndvi":None} for k in products}
    else:
        req_dl_vars = None
        products = None

    dss = download_mod(mod, workdir, timelim, latlim, lonlim, req_dl_vars=req_dl_vars, products=products)
    
    for ds in dss.values():
        assert ds.rio.crs.to_epsg() == 4326
        assert "spatial_ref" in ds.coords
        assert strictly_increasing(ds.x.values)
        assert strictly_decreasing(ds.y.values)
        if "time" in ds.dims:
            assert strictly_increasing(ds.time.values)
        assert has_geotransform(ds)
        assert np.all([int(ds[var].notnull().sum().values) > 0 for var in ds.data_vars])

if __name__ == "__main__":

    tmp_path = r"/Users/hmcoerver/Local/test_dl_GEOS5_0"
    product_name = "GEOS5"
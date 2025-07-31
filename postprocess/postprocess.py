from typing import List, Optional, Union
from pathlib import Path
import rasterio
from rasterio.transform import from_origin, Affine
import numpy as np
import xarray as xr


class PostProcessing:
    """
    A class for post-processing operations on geospatial data.
    
    This class handles the conversion of NetCDF files to GeoTIFF format
    for a specific region.
    
    Attributes:
        region (str): The region identifier for processing data.
    """
    
    def __init__(self, region: str) -> None:
        """
        Initialize the PostProcessing instance.
        
        Args:
            region: The region identifier used for file paths and naming.
        """
        self.region: str = region

    def export_nc_to_tiff(self, list_date: List[str]) -> None:
        """
        Export NetCDF files to GeoTIFF format for specified dates.
        
        This method processes NetCDF files containing AETI (Actual Evapotranspiration
        and Interception) data and converts them to GeoTIFF format with proper
        geospatial referencing.
        
        Args:
            list_date: A list of date strings in the format expected
                      by the NetCDF file naming convention.
                            
        Raises:
            FileNotFoundError: If the NetCDF file for a specific date doesn't exist.
            ValueError: If the data cannot be processed or written.
            
        Note:
            - Input NetCDF files should be located at:
              `/mnt/storage/code/pywapor-clms/res/res_{region}/{date}/res_{date}`
            - Output GeoTIFF files will be saved to:
              `/mnt/storage/code/pywapor-clms/res_image/result_image_{region}/`
            - The method assumes EPSG:4326 coordinate reference system.
        """
        for date in list_date:
            # Open NetCDF dataset
            nc_file_path: str = (f"/mnt/storage/code/pywapor-clms/res/res_{self.region}/"
                                f"{date}/res_{date}")
            ds1: xr.Dataset = xr.open_dataset(nc_file_path)
            
            # Extract AETI values
            aeti: np.ndarray = ds1["aeti_24_mm"].isel(time_bins=0).values  # Shape: (y, x)
            aeti = np.squeeze(aeti)

            # Extract coordinate information
            x: np.ndarray = ds1.coords["x"].values
            y: np.ndarray = ds1.coords["y"].values

            # Calculate spatial resolution
            res_x: float = abs(x[1] - x[0])
            res_y: float = abs(y[1] - y[0])
            
            # Create affine transformation
            # Adjust for pixel-center to pixel-corner registration
            transform: Affine = from_origin(
                x[0] - res_x / 2, 
                y[0] + res_y / 2, 
                res_x, 
                res_y
            )

            # Define output file path
            output_path: str = (f"/mnt/storage/code/pywapor-clms/res_image/"
                               f"result_image_{self.region}/aete_res_{date}.tif")

            # Write GeoTIFF file
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=aeti.shape[0],
                width=aeti.shape[1],
                count=1,
                dtype=aeti.dtype,
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(aeti, 1)
                
            # Close the dataset to free memory
            ds1.close()
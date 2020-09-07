import collections

import numpy as np
import pandas  as pd
import xarray as xr


def make_test_store() -> collections.MutableMapping:
    return {
        ".zgroup": bytes('{"zarr_format": 2}', 'utf8'),
        ".zattrs": bytes('{}', 'utf8'),
        "var1/.zarray": bytes('{"zarr_format": 2}', 'utf8'),
        "var1/.zattrs": bytes('{}', 'utf8'),
        "var1/0": bytes(np.array([1, 2, 3])),
        "var2/.zarray": bytes('{"zarr_format": 2}', 'utf8'),
        "var2/.zattrs": bytes('{}', 'utf8'),
        "var2/0": bytes(np.array([4, 5, 6])),
    }


def make_test_cube(num_lats=180, num_times=10, num_vars=3) -> xr.Dataset:
    width = 2 * num_lats
    height = num_lats
    spatial_res = 180.0 / num_lats
    lon = xr.DataArray(np.linspace(-180 + spatial_res, 180 - spatial_res, width), dims=['lon'])
    lat = xr.DataArray(np.linspace(-90 + spatial_res, 90 - spatial_res, height), dims=['lat'])
    time = xr.DataArray(pd.date_range(pd.to_datetime('2020-01-01 10:00:00'), periods=num_times), dims=['time'])
    coords = dict(time=time, lat=lat, lon=lon)
    return xr.Dataset({f'v{i + 1}': xr.DataArray(
        np.random.logistic(size=width * height * num_times).reshape((num_times, height, width)),
        dims=['time', 'lat', 'lon']) for i in range(num_vars)}, coords=coords)

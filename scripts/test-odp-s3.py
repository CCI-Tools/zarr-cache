import collections
import os
import os.path

import xarray as xr
import zarr.storage

from xcube_cci.cciodp import CciOdp
from xcube_cci.chunkstore import CciChunkStore
from zarr_cache import CachedStore
from zarr_cache.indexes import MemoryStoreIndex

odp = CciOdp()

ds_id = "esacci.OZONE.mon.L3.NP.multi-sensor.multi-platform.MERGED.fv0002.r1"

original_store = CciChunkStore(odp,
                               ds_id,
                               dict(
                                   # Note, crash, if I use tuple for time_range
                                   time_range=["2010-01-01", "2011-01-01"],
                                   variable_names=[
                                       "O3_du",
                                       "O3_du_tot",
                                       "surface_pressure",
                                   ]))


def store_opener(store_id: str) -> collections.MutableMapping:
    if not os.path.isdir('store_cache'):
        os.mkdir('store_cache')
    return zarr.storage.DirectoryStore(f'store_cache/{store_id}.zarr')


store_index = MemoryStoreIndex()

ozone_1_store = CachedStore(original_store, "ozone_1", store_index=store_index, store_opener=store_opener)
ozone_2_store = CachedStore(original_store, "ozone_2", store_index=store_index, store_opener=store_opener)
ozone_3_store = CachedStore(original_store, "ozone_3", store_index=store_index, store_opener=store_opener)


ozone_1_ds = xr.open_zarr(ozone_1_store)
ozone_2_ds = xr.open_zarr(ozone_2_store)
ozone_3_ds = xr.open_zarr(ozone_3_store)

# print(ozone_1_ds)

ozone_1_ds.to_zarr('ozone_1a.zarr')
ozone_3_ds.to_zarr('ozone_3a.zarr')
ozone_2_ds.to_zarr('ozone_2a.zarr')
ozone_3_ds.to_zarr('ozone_3b.zarr')
ozone_2_ds.to_zarr('ozone_2b.zarr')
ozone_1_ds.to_zarr('ozone_1b.zarr')


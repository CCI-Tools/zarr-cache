import collections
import os
import os.path

import s3fs
import xarray as xr
import zarr.storage

from xcube_cci.cciodp import CciOdp
from xcube_cci.chunkstore import CciChunkStore
from zarr_cache import CachedStore
from zarr_cache import DefaultStoreCache
from zarr_cache.indexes import MemoryStoreIndex

ds_id = "esacci.OZONE.mon.L3.NP.multi-sensor.multi-platform.MERGED.fv0002.r1"


def get_cciodp_store():
    odp = CciOdp()
    return CciChunkStore(odp,
                         ds_id,
                         dict(
                             # Note, crash, if I use tuple for time_range
                             time_range=["2010-01-01", "2011-01-01"],
                             variable_names=[
                                 "O3_du",
                                 "O3_du_tot",
                                 "surface_pressure",
                             ]))


new_ds_id = ds_id.replace('.mon.', '.2010.')

# original_store = get_cciodp_store()
# ds = xr.open_zarr(original_store)
# ds.to_zarr(f'{new_ds_id}.zarr')
# exit(0)
######################################################################################################


# with open('jasmin-os-credentials.json') as fp:
#     credentials = json.load(fp)
# s3 = s3fs.S3FileSystem(anon=False,
#                        key=credentials['aws_access_key_id'],
#                        secret=credentials['aws_secret_access_key'],
#                        client_kwargs=dict(endpoint_url=credentials['endpoint_url']))

s3 = s3fs.S3FileSystem()
s3.mkdir("cciodp-cache-v1")
s3map = s3fs.S3Map(f"cciodp-cache-v1/{ds_id}.zarr", s3, check=False, create=True)

ds = xr.open_zarr(f'{new_ds_id}.zarr')
ds.to_zarr(s3map)

exit(0)


######################################################################################################


def store_opener(store_id: str) -> collections.MutableMapping:
    if not os.path.isdir('store_cache'):
        os.mkdir('store_cache')
    return zarr.storage.DirectoryStore(f'store_cache/{store_id}.zarr')


store_index = MemoryStoreIndex()

store_cache = DefaultStoreCache(store_index=store_index, store_opener=store_opener)
ozone_1_store = CachedStore(original_store, "ozone_1", store_cache)
ozone_2_store = CachedStore(original_store, "ozone_2", store_cache)
ozone_3_store = CachedStore(original_store, "ozone_3", store_cache)

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

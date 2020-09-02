import collections

import s3fs
import xarray as xr

from xcube_cci.cciodp import CciOdp
from xcube_cci.chunkstore import CciChunkStore
from zarr_cache import CachedStore
from zarr_cache import DefaultStoreCache
from zarr_cache.indexes import MemoryStoreIndex

odp = CciOdp()

bucket_name = "cciodp-cache-v1"

# with open('jasmin-os-credentials.json') as fp:
#     credentials = json.load(fp)
# s3 = s3fs.S3FileSystem(anon=False,
#                        key=credentials['aws_access_key_id'],
#                        secret=credentials['aws_secret_access_key'],
#                        client_kwargs=dict(endpoint_url=credentials['endpoint_url']))

s3 = s3fs.S3FileSystem()
if not s3.isdir(bucket_name):
    s3.mkdir(bucket_name)


def store_opener(store_id: str) -> collections.MutableMapping:
    return s3fs.S3Map(f"{bucket_name}/{store_id}.zarr", s3, check=False, create=True)


store_index = MemoryStoreIndex()

store_cache = DefaultStoreCache(store_index=store_index, store_opener=store_opener)


def open_cached_dataset(ds_id):
    md = odp.get_dataset_metadata(ds_id)
    time_range = md['temporal_coverage_start'], md['temporal_coverage_end']
    variable_names = odp.var_names(ds_id)
    original_store = CciChunkStore(odp, ds_id, dict(variable_names=variable_names, time_range=time_range))
    cached_store = CachedStore(original_store, ds_id, store_cache)
    return xr.open_zarr(cached_store)

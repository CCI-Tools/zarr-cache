import s3fs
import os.path
import xarray as xr

from xcube_cci.cciodp import CciOdp
from xcube_cci.chunkstore import CciChunkStore
from zarr_cache import CachedStore, S3StoreOpener
from zarr_cache import IndexedCacheStorage
from zarr_cache import MemoryStoreIndex

odp = CciOdp()

dataset_names_path = 'dataset_names.txt'
if not os.path.exists(dataset_names_path):
    with open(dataset_names_path, 'w') as fp:
        fp.writelines(map(lambda s: s + '\n', sorted(odp.dataset_names)))

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

store_index = MemoryStoreIndex()

store_cache = IndexedCacheStorage(store_index=store_index,
                                  store_opener=S3StoreOpener(bucket_name + "/{store_id}.zarr", s3=s3))

def open_cached_dataset(ds_id):
    original_store = CciChunkStore(odp, ds_id)
    cached_store = CachedStore(original_store, ds_id, store_cache)
    return xr.open_zarr(cached_store)

# ds = open_cached_dataset('esacci.OZONE.mon.L3.NP.multi-sensor.multi-platform.MERGED.fv0002.r1')
# ds = open_cached_dataset('esacci.CLOUD.mon.L3C.CLD_PRODUCTS.multi-sensor.multi-platform.AVHRR-PM.3-0.r1')
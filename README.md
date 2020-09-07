# zarr-cache

An experimental cache for multiple Zarr datasets.

## Idea

Suppose we read Zarr chunks from some slow API. The API access is 
implemented as a `collections.abc.MutableMapping`.

Assumed we have much faster storage, for example some high performance 
S3-compatible object storage. 

## Usage

Programming model:

```python
    from collections.abc import MutableMapping

    from zarr_cache import CachedStore
    from zarr_cache import S3StoreOpener
    from zarr_cache import RedisStoreIndex
    from zarr_cache import IndexedCacheStorage

    def open_my_slow_store(store_id: str, ...) -> MutableMapping:
        return ...

    def wrap_store(original_store: MutableMapping, store_id: str) -> MutableMapping:
        store_index = RedisStoreIndex(...)
        store_opener = S3StoreOpener('s3://my_bucket/{store_id}.zarr', ...)
        cache_storage = IndexedCacheStorage(store_index, store_opener)
        return CachedStore(original_store, store_id, cache_storage)

    my_slow_store = open_my_slow_store(my_store_id, ...); 
    my_faster_store = wrap_store(my_slow_store, my_store_id)
    
    
```


## Installation

### Get code

    $ git clone https://github.com/CCI-Tools/zarr-cache.git
    $ cd zarr-cache

### Install Python environment

If you already have an existing environment that is supposed to use `zarr-cache`:    

    $ conda env update
     
If you don't have an environment yet: 
    
    $ conda env create
    $ conda activate zarr-cache
    
### Install package

    $ python setup.py install 



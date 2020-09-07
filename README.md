# zarr-cache

An experimental cache for multiple Zarr datasets.

## Idea

Suppose we read Zarr array chunks from some slow data API. The API access is 
implemented as store of type `collections.abc.MutableMapping`. And 
assumed we have much faster storage, for example some high performance 
S3-compatible object storage. Then it would make sense to use the 
object storage as a cache for the array chunks retrieved by the slow API store.

However, caching requires efficient management of the possibly very large number 
of keys in the cache. Enumerating and sorting of keys, pushing and popping of 
keys should be very fast operations. We'd also like to associate cache-specific
values which each key e.g. access frequency, access duration, chunk sizes, etc. 
The object storage may not be an ideal candidate provide that capabilities very well.

The design used in this library therefore splits the cache storage into 
 
1. an index of cached keys, e.g. a Redis database; 
2. a storage for the cached array chunks, e.g. S3.

## Usage

Programming model:

```python
    from collections.abc import MutableMapping

    from zarr_cache import CachedStore
    from zarr_cache import S3StoreOpener
    from zarr_cache import MemoryStoreIndex
    from zarr_cache import IndexedCacheStorage

    def open_my_slow_store(store_id: str, ...) -> MutableMapping:
        return ...

    def wrap_store(original_store: MutableMapping, store_id: str) -> MutableMapping:
        # Coming soon:
        # store_index = RedisStoreIndex(...)
        store_index = MemoryStoreIndex()
        store_opener = S3StoreOpener('s3://my_bucket/{store_id}.zarr', ...)
        cache_storage = IndexedCacheStorage(store_index, store_opener)
        return CachedStore(original_store, store_id, cache_storage)
    
    my_store_id = "..."
    my_slow_store = open_my_slow_store(my_store_id, ...) 
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



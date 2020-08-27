import xarray as xr

from xcube_cci.cciodp import CciOdp
from xcube_cci.chunkstore import CciChunkStore

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
                                       "air_pressure",
                                       "surface_pressure",
                                   ]))

dataset = xr.open_zarr(original_store)

dataset.to_zarr(ds_id + '.zarr')

print(dataset)

import xarray as xr
import pyremap

data_directory = r"/oscar/home/pvankatw/data/pvankatw/CMIP/"
data_path = r"pr_day_MPI-ESM1-2-HR_ssp585_r1i1p1f1_gn_20150101-20191231.nc"
in_filename = data_directory + data_path
ds = xr.open_dataset(in_filename)

inDescriptor = pyremap.LatLonGridDescriptor.read(in_filename, latVarName='lat', lonVarName='lon')

out_filename = f"/users/pvankatw/research/ise/AIS_sectors_8km.nc"
outDescriptor = pyremap.get_polar_descriptor_from_file(out_filename, projection='antarctic')

remapper = pyremap.Remapper(inDescriptor, outDescriptor, mappingFileName='map.nc')

esmf_path = r"/oscar/home/pvankatw/data/pvankatw/ESMF/esmf-8.7.0/install/"
remapper.build_mapping_file(method='conserve', esmf_path=esmf_path)



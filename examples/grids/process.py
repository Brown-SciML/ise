import sys

sys.path.append("../..")
from ise.data.process import DatasetMerger, DimensionalityReducer, ProjectionProcessor
from ise.models.HybridEmulator import PCAModel, WeakPredictor

ice_sheet = "AIS"
print(f"ice sheet: {ice_sheet}")

# all filepaths...
forcing_directory = r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing/"
projections_directory = (
    r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-AIS/"
    if ice_sheet == "AIS"
    else r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-GrIS/"
)
scalefac_fp = (
    r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/af2_el_ismip6_ant_01.nc"
    if ice_sheet == "AIS"
    else r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Projection-GrIS/af2_ISMIP6_GrIS_05000m.nc"
)
densities_path = (
    r"/users/pvankatw/research/current/supplemental/AIS_densities.csv"
    if ice_sheet == "AIS"
    else r"/users/pvankatw/research/current/supplemental/GIS_densities.csv"
)
output_dir = f"/oscar/home/pvankatw/scratch/pca/{ice_sheet}"
converted_forcing_dir = f"/oscar/home/pvankatw/scratch/pca/{ice_sheet}/forcings/"
converted_projection_dir = f"/oscar/home/pvankatw/scratch/pca/{ice_sheet}/projections/"
experiment_file = r"/users/pvankatw/research/current/supplemental/ismip6_experiments_updated.csv"
# df = get_model_densities(r"/gpfs/data/kbergen/pvankatw/pvankatw-bfoxkemp/v7_CMIP5_pub", r"/users/pvankatw/research/current/supplemental/")

# Create IVAF files from raw outputs from ice sheet models
processor = ProjectionProcessor(
    ice_sheet, forcing_directory, projections_directory, scalefac_fp, densities_path
)
processor.process()

# Take both the forcing files and the projections, train PCA models, and convert forcings and projections to PCA space
pca = DimensionalityReducer(
    forcing_dir=forcing_directory,
    projection_dir=projections_directory,
    output_dir=output_dir,
    ice_sheet=ice_sheet,
)
pca.generate_pca_models()
pca.convert_forcings(num_pcs="95%", pca_model_directory=f"{output_dir}/pca_models/")
pca.convert_projections(num_pcs="99%", pca_model_directory=f"{output_dir}/pca_models/")

# Merge the converted forcings and projections into a single dataset
merger = DatasetMerger(
    ice_sheet, converted_forcing_dir, converted_projection_dir, experiment_file, output_dir
)
merger.merge_dataset()

print("Done!")

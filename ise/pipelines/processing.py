from ise.data.processors.forcings import process_forcings
from ise.data.processors.ismip6 import process_ismip6_outputs
from ise.data.processors.merge import merge_datasets

def process_data(forcing_directory, ismip6_output_directory, export_directory):
    process_forcings(forcing_directory, export_directory, to_process='all', verbose=False)
    process_ismip6_outputs(ismip6_output_directory, export_directory)
    master, inputs, outputs = merge_datasets(export_directory, export_directory, export_directory, include_icecollapse=False)
    return master, inputs, outputs
    
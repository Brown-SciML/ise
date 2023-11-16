r"""
# [forcings](https://brown-sciml.github.io/ise/ise/sectors/data/processors/forcings.html)
Processing functions for ISMIP6 atmospheric, oceanic, and ice-collapse forcings found in the Globus ISMIP6 Archive

# [ismip6](https://brown-sciml.github.io/ise/ise/sectors/data/processors/ismip6.html)
Processing functions for ismip6 ice sheet model outputs.

# [merge](https://brown-sciml.github.io/ise/ise/sectors/data/processors/merge.html)
Processing functions for joining the processed inputs from the forcing directory and the outputs from the ismip6 ice sheet models to create a master dataset.

# [control](https://brown-sciml.github.io/ise/ise/sectors/data/processors/control.html)
Processing functions for the control experiments.
"""


from ise.sectors.data.processors.forcings import (
    process_forcings,
    AtmosphereForcing,
    GridSectors,
    IceCollapse,
    OceanForcing,
    aggregate_atmosphere,
    aggregate_by_sector,
    aggregate_icecollapse,
    aggregate_ocean,
)

from ise.sectors.data.processors.ismip6 import (
    process_ismip6_outputs,
    _get_sector,
    process_experiment,
    process_repository,
    process_single_file,
)

from ise.sectors.data.processors.merge import combine_datasets, exp_to_attributes, format_aogcms

from ise.sectors.data.processors.control import create_control_dataset

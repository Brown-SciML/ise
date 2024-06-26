r"""

# ISE

ISE, or ice-sheet emulators, is a package for end-to-end creation and analysis of ice-sheet emulators.

The main features of ISE include loading and processing of ISMIP6 sea level contribution simulations,
data preparation and feature engineering for machine learning, and training and testing of trained neural network emulators.
The package is divided into two sections: `sectors` and `grids`. The sectors module provides all necessary functions for
creating and training emulators based on the 18 ISMIP6 sectors, while the grids module provides the same functionality
for smaller kilometer-scale grids.

# Quickstart

To get started, you must first have access to the Globus Archive containing the ISMIP6 climate
forcings and ISMIP6 model outputs. For information on gaining access to these datasets, see the [ISMIP
wiki page](https://theghub.org/groups/ismip6/wiki).

Next, clone the repository by running the following command in your terminal:
```shell
git clone https://github.com/Brown-SciML/ise.git
```

To use it as a package, navigate to the cloned directory and run the following command:
```shell
pip install -e .
```

*This repository is a work in progress that is actively being updated and improved. Feel free to contact Peter Van Katwyk, Ph.D. Candidate @ Brown University at peter_van_katwyk@brown.edu with further questions.*

"""

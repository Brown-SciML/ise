r"""

# ISE

ISE, or ice-sheet emulators, is a package for end-to-end creation of ice-sheet emulators.

The main features of ISE include loading and processing of ISMIP6 sea level contribution simulations,
data preparation and feature engineering for machine learning, and training and testing of trained neural network emulators.

# Quickstart

To get started, you must first have access to the Globus Archive containing the ISMIP6 climate
forcings and ice-sheet model outputs located at [GHub-ISMIP6-Forcing](https://app.globus.org/file-manager?origin_id=ad1a6ed8-4de0-4490-93a9-8258931766c7&origin_path=%2F).
Do not change the file structure or directory tree.

Next, clone the repository by running the following command in your terminal:
```shell
git clone https://github.com/Brown-SciML/ise.git
```

To use it as a package, navigate to the cloned directory and run the following command:
```shell
pip install -e .
```

*This repository is a work in progress that is actively being updated and improved. Feel free to contact Peter Van Katwyk, Ph.D. student @ Brown University at peter_van_katwyk@brown.edu with further questions.*

"""
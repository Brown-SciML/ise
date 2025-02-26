ISE Documentation
=================

ISE, or ice-sheet emulators, is a package for end-to-end creation and analysis of ice-sheet emulators.

The main features of ISE include loading and processing of ISMIP6 sea level contribution simulations,
data preparation and feature engineering for machine learning, and training and testing of trained neural network emulators.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   docs/source/ise.rst


Quickstart
==========

To get started, you must first have access to the Globus Archive containing the ISMIP6 climate
forcings and ISMIP6 model outputs. For information on gaining access to these datasets, see the 
`ISMIP wiki page <https://theghub.org/groups/ismip6/wiki>`_.

Next, clone the repository by running the following command in your terminal:

.. code-block:: shell

   git clone https://github.com/Brown-SciML/ise.git

To use it as a package, navigate to the cloned directory and run:

.. code-block:: shell

   pip install -e .

About
=====

*This repository is a work in progress that is actively being updated and improved. Feel free to contact
Peter Van Katwyk, Ph.D. Candidate @ Brown University at peter_van_katwyk@brown.edu with further questions.*

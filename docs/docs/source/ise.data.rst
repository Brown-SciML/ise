ise.data
========

Data loading, processing, and feature engineering for ice sheet emulation.

This package covers the full data pipeline:

- **ForcingFile** / **GridFile** — load and sector-average climate forcing and
  grid NetCDF files.
- **ISEFlowAISInputs** / **ISEFlowGrISInputs** — validated input dataclasses
  for running pretrained ISEFlow emulators.
- **FeatureEngineer** — train/val/test splitting, scaling, lag variables,
  outlier handling, and ISM characteristic merging.
- **ProjectionProcessor** / **DatasetMerger** — IVAF calculation from raw
  ISMIP6 NetCDF outputs and sector-level forcing/projection merging.
- **EmulatorDataset** / **PyTorchDataset** / **TSDataset** / **ScenarioDataset**
  — PyTorch ``Dataset`` subclasses for LSTM and normalizing-flow training.
- **StandardScaler** / **RobustScaler** / **LogScaler** — GPU-compatible
  ``nn.Module`` scalers for use in training loops.

Submodules
----------

ise.data.forcings
-----------------

.. automodule:: ise.data.forcings
   :members:
   :undoc-members:
   :show-inheritance:

ise.data.grids
--------------

.. automodule:: ise.data.grids
   :members:
   :undoc-members:
   :show-inheritance:

ise.data.inputs
---------------

.. automodule:: ise.data.inputs
   :members:
   :undoc-members:
   :show-inheritance:

ise.data.feature\_engineer
--------------------------

.. automodule:: ise.data.feature_engineer
   :members:
   :undoc-members:
   :show-inheritance:

ise.data.process
----------------

.. automodule:: ise.data.process
   :members:
   :undoc-members:
   :show-inheritance:

ise.data.dataclasses
--------------------

.. automodule:: ise.data.dataclasses
   :members:
   :undoc-members:
   :show-inheritance:

ise.data.scaler
---------------

.. automodule:: ise.data.scaler
   :members:
   :undoc-members:
   :show-inheritance:

ise.data.utils
--------------

.. automodule:: ise.data.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: ise.data
   :members:
   :undoc-members:
   :show-inheritance:

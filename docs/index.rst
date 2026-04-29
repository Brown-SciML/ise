ISE Documentation
=================

.. image:: https://readthedocs.org/projects/ise/badge/?version=latest
   :target: https://ise.readthedocs.io/en/latest/

**ISE** (Ice Sheet Emulator) is a Python package for training and running machine
learning emulators of ice sheet models, with a focus on **ISEFlow** — a hybrid
normalizing-flow + deep-ensemble neural network that produces sea level projections
with full **epistemic and aleatoric uncertainty quantification** for the Antarctic
Ice Sheet (AIS) and Greenland Ice Sheet (GrIS).

Supported ice sheets
~~~~~~~~~~~~~~~~~~~~

- **AIS** (Antarctic Ice Sheet): 18 sectors, 8 km standard resolution
- **GrIS** (Greenland Ice Sheet): 6 regions, 5 km standard resolution

Projection period: **2015-2100** (86 annual timesteps).

About ISEFlow
=============

ISEFlow combines two sub-models trained in sequence:

1. **NormalizingFlow** — autoregressive masked affine flow trained via maximum
   likelihood on ISMIP6 outputs.  Captures **aleatoric** (data) uncertainty via
   posterior sampling.
2. **DeepEnsemble** — ensemble of LSTM networks trained on the original features
   concatenated with NormalizingFlow latent representations.  Captures
   **epistemic** (model) uncertainty as disagreement across members.

Total uncertainty = epistemic + aleatoric.

This codebase has been used in peer-reviewed research, including:

- *"A Variational LSTM Emulator of Sea Level Contribution From the Antarctic Ice
  Sheet"*
- *"ISEFlow: A Flow-Based Neural Network Emulator for Improved Sea Level
  Projections and Uncertainty Quantification"*

For replication details see the `Releases <https://github.com/Brown-SciML/ise/releases>`_
section.

Installation
============

ISE uses `uv <https://github.com/astral-sh/uv>`_ for dependency management.
Set up the environment with:

.. code-block:: shell

   uv venv
   uv pip install -r requirements.txt

or using **pip** directly:

.. code-block:: shell

   pip install -r requirements.txt

For development (editable install):

.. code-block:: shell

   pip install -e .

Quickstart — pretrained ISEFlow-AIS
=====================================

.. code-block:: python

   import numpy as np
   from ise.models import ISEFlow_AIS
   from ise.data.inputs import ISEFlowAISInputs

   # Build the input dataclass for a single sector (sector 1)
   year = np.arange(2015, 2101)          # 86 years
   sector = np.ones(86, dtype=int)       # sector 1

   inputs = ISEFlowAISInputs(
       year=year,
       sector=sector,
       pr_anomaly=np.zeros(86),
       evspsbl_anomaly=np.zeros(86),
       smb_anomaly=np.zeros(86),
       ts_anomaly=np.zeros(86),
       ocean_thermal_forcing=np.zeros(86),
       ocean_salinity=np.zeros(86),
       ocean_temperature=np.zeros(86),
       # ISM configuration (use model_configs shortcut or set individually)
       model_configs="AWI_PISM1",
       ice_shelf_fracture=False,
       ocean_sensitivity="medium",
       ocean_forcing_type="standard",
       standard_melt_type="local",
   )

   # Load pretrained model (v1.1.0) and run inference
   model = ISEFlow_AIS(version="v1.1.0")
   predictions, uncertainties = model.predict(inputs)

   print(predictions.shape)          # (86, 1)  — SLE in mm
   print(uncertainties["epistemic"]) # epistemic uncertainty per timestep
   print(uncertainties["aleatoric"]) # aleatoric uncertainty per timestep

Package layout
==============

.. code-block:: text

   ise/
   ├── data/
   │   ├── forcings.py          ForcingFile: load/process climate NetCDF data
   │   ├── grids.py             GridFile: sector boundary definitions
   │   ├── inputs.py            ISEFlowAISInputs, ISEFlowGrISInputs
   │   ├── feature_engineer.py  FeatureEngineer: split, scale, lag, outliers
   │   ├── process.py           ProjectionProcessor, DatasetMerger, sector utils
   │   ├── dataclasses.py       EmulatorDataset, PyTorchDataset, TSDataset
   │   ├── scaler.py            PyTorch StandardScaler, RobustScaler, LogScaler
   │   └── utils.py             convert_and_subset_times()
   ├── models/
   │   ├── iseflow.py           ISEFlow, ISEFlow_AIS, ISEFlow_GrIS
   │   ├── deep_ensemble.py     DeepEnsemble: ensemble of LSTM models
   │   ├── lstm.py              LSTM: single LSTM network
   │   ├── normalizing_flow.py  NormalizingFlow: autoregressive masked affine flow
   │   ├── loss.py              WeightedGridLoss, WeightedMSELoss, and variants
   │   ├── training.py          CheckpointSaver, EarlyStoppingCheckpointer
   │   ├── pretrained/          Pretrained weight paths + expected variable lists
   │   └── _experimental/       GP, PCA, ScenarioPredictor, VariationalLSTMEmulator
   ├── evaluation/
   │   └── metrics.py           Point, probabilistic, and distribution metrics
   └── utils/
       ├── functions.py         get_X_y, get_data, to_tensor, unscale_output, …
       └── io.py                check_type() runtime type validation

Contributing
============

We welcome contributions!  To get started:

1. **Fork the repository** on GitHub.
2. **Create a new branch** for your feature or bugfix.
3. **Submit a pull request** (PR) for review.

Run tests before submitting:

.. code-block:: shell

   pytest tests/

Contact & Support
=================

This repository is maintained by **Peter Van Katwyk**, Ph.D. student at
**Brown University**.

- **Email:** `pvankatwyk@gmail.com <mailto:pvankatwyk@gmail.com>`_
- **GitHub Issues:** `Report a bug <https://github.com/Brown-SciML/ise/issues>`_

If you use ISE in research, please consider citing our work.  See
``CITATION.md`` for details.

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   docs/source/ise

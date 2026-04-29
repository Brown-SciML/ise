# Ice Sheet Emulator (ISE)

[![Documentation Status](https://readthedocs.org/projects/ise/badge/?version=latest)](https://ise.readthedocs.io/en/latest/)

**ISE** is a Python package for training and analyzing **ice sheet emulators**, including **ISEFlow** — a hybrid flow-based neural network emulator for improved **sea level projections** and **uncertainty quantification**.

ISEFlow supports emulation for both the **Antarctic Ice Sheet (AIS)** and the **Greenland Ice Sheet (GrIS)**, producing projections of ice volume above flotation (IVAF) changes driven by ISMIP6 climate forcings. Uncertainty is decomposed into **epistemic** (model) and **aleatoric** (data) components.

This codebase has been used in peer-reviewed research, including:

- *"A Variational LSTM Emulator of Sea Level Contribution From the Antarctic Ice Sheet"*
- *"ISEFlow: A Flow-Based Neural Network Emulator for Improved Sea Level Projections and Uncertainty Quantification"*

For replication details see the [Releases](https://github.com/Brown-SciML/ise/releases) section.

**Documentation:** <https://ise.readthedocs.io/>

---

## Installation

Install in editable mode:

```sh
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```sh
uv venv
uv pip install -e .
```

---

## Project Structure

```text
ise/
├── examples/                   Example scripts for using ISEFlow
├── ise/                        Main package
│   ├── data/                   Forcing/grid loading, feature engineering, dataset classes
│   │   ├── anomaly.py          AnomalyConverter: raw forcing → ISMIP6 anomalies
│   │   ├── dataclasses.py      EmulatorDataset, PyTorchDataset, TSDataset, ScenarioDataset
│   │   ├── feature_engineer.py FeatureEngineer: split, scale, lag, outliers
│   │   ├── forcings.py         ForcingFile: load/process climate NetCDF data
│   │   ├── grids.py            GridFile: sector boundary definitions
│   │   ├── inputs.py           ISEFlowAISInputs, ISEFlowGrISInputs
│   │   ├── process.py          ProjectionProcessor, DatasetMerger, sector helpers
│   │   ├── scaler.py           PyTorch StandardScaler, RobustScaler, LogScaler
│   │   └── utils.py            convert_and_subset_times()
│   ├── evaluation/             Metrics
│   │   └── metrics.py          Point, probabilistic, and distribution metrics
│   ├── models/                 Model architectures
│   │   ├── iseflow.py          ISEFlow, ISEFlow_AIS, ISEFlow_GrIS
│   │   ├── deep_ensemble.py    DeepEnsemble
│   │   ├── lstm.py             LSTM
│   │   ├── normalizing_flow.py NormalizingFlow
│   │   ├── training.py         CheckpointSaver, EarlyStoppingCheckpointer
│   │   ├── loss.py             WeightedGridLoss, WeightedMSELoss, and variants
│   │   ├── _experimental/      Legacy models from prior manuscripts (deprecated)
│   │   └── pretrained/         Pretrained weights (v1.0.0, v1.1.0)
│   └── utils/                  Data helpers and tensor utilities
│       ├── functions.py        get_X_y, get_data, to_tensor, unscale_output, …
│       └── io.py               check_type() runtime type validation
├── manuscripts/                Research paper scripts
├── tests/                      Unit tests
├── pyproject.toml
└── uv.lock
```

---

## Usage

### Loading and Running a Pretrained ISEFlow-AIS Model

```python
import numpy as np
from ise.models import ISEFlow_AIS
from ise.data.inputs import ISEFlowAISInputs

year = np.arange(2015, 2101)  # 86 annual timesteps

# Option A: use a known ISMIP6 ISM configuration shortcut
inputs = ISEFlowAISInputs(
    year=year,
    sector=np.ones(86, dtype=int),      # sector 1 of 18
    pr_anomaly=np.zeros(86),
    evspsbl_anomaly=np.zeros(86),
    smb_anomaly=np.zeros(86),
    ts_anomaly=np.zeros(86),
    ocean_thermal_forcing=np.zeros(86),
    ocean_salinity=np.zeros(86),
    ocean_temperature=np.zeros(86),
    model_configs="AWI_PISM1",          # loads all ISM config fields automatically
    ice_shelf_fracture=False,
    ocean_sensitivity="medium",
    ocean_forcing_type="standard",
    standard_melt_type="local",
)

# Option B: provide all ISM parameters individually
inputs = ISEFlowAISInputs(
    year=year,
    sector=np.ones(86, dtype=int),
    pr_anomaly=np.zeros(86),
    evspsbl_anomaly=np.zeros(86),
    smb_anomaly=np.zeros(86),
    ts_anomaly=np.zeros(86),
    ocean_thermal_forcing=np.zeros(86),
    ocean_salinity=np.zeros(86),
    ocean_temperature=np.zeros(86),
    initial_year=1980,
    numerics="fd",
    stress_balance="ho",
    resolution="16",
    init_method="da",
    melt_in_floating_cells="floating condition",
    icefront_migration="str",
    ocean_forcing_type="open",
    ocean_sensitivity="low",
    ice_shelf_fracture=False,
    open_melt_type="picop",
    standard_melt_type=None,
)

# Option C: if you have raw (non-anomaly) forcing values, use from_raw_values()
inputs = ISEFlowAISInputs.from_raw_values(
    year=year,
    sector=10,
    pr=pr_array,
    evspsbl=evspsbl_array,
    smb=smb_array,
    ts=ts_array,
    ocean_thermal_forcing=otf_array,
    ocean_salinity=sal_array,
    ocean_temperature=temp_array,
    aogcm="noresm1-m_rcp85",           # or custom_climatology={...}
    model_configs="AWI_PISM1",
    ice_shelf_fracture=False,
    ocean_sensitivity="medium",
    ocean_forcing_type="standard",
    standard_melt_type="local",
)

# Load the pretrained v1.1.0 model and run inference
model = ISEFlow_AIS(version="v1.1.0")
predictions, uncertainties = model.predict(inputs)

print(predictions.shape)           # (86, 1)  — SLE in mm, 2015-2100
print(uncertainties["epistemic"])  # epistemic uncertainty per timestep
print(uncertainties["aleatoric"])  # aleatoric uncertainty per timestep
print(uncertainties["total"])      # total uncertainty (epistemic + aleatoric)
```

### Running the Pretrained GrIS Emulator

```python
import numpy as np
from ise.models import ISEFlow_GrIS
from ise.data.inputs import ISEFlowGrISInputs

inputs = ISEFlowGrISInputs(
    year=np.arange(2015, 2101),
    sector=np.ones(86, dtype=int),      # basin 1 of 6
    aST=np.zeros(86),                   # surface temperature anomaly
    aSMB=np.zeros(86),                  # SMB anomaly
    ocean_thermal_forcing=np.zeros(86),
    basin_runoff=np.zeros(86),
    model_configs="AWI_ISSM1",
    ice_shelf_fracture=False,
    ocean_sensitivity="medium",
    standard_ocean_forcing=True,
)

model = ISEFlow_GrIS(version="v1.1.0")
predictions, uncertainties = model.predict(inputs)
```

### Training ISEFlow from Scratch

```python
from ise.models import ISEFlow, DeepEnsemble, NormalizingFlow

nf = NormalizingFlow(input_size=93, output_size=1, num_flow_transforms=5)
de = DeepEnsemble(input_size=93, num_ensemble_members=5, output_sequence_length=86)
model = ISEFlow(deep_ensemble=de, normalizing_flow=nf)

# X: (N, n_features), y: (N,) — pre-scaled ISMIP6 data
model.fit(
    X_train, y_train,
    nf_epochs=100,
    de_epochs=100,
    X_val=X_val,
    y_val=y_val,
    early_stopping=True,
    patience=15,
)

model.save("./ISEFlow/", input_features=list(X_train.columns))
```

### Evaluating Model Performance

```python
from ise.models import ISEFlow
from ise.evaluation import metrics as m
from ise.utils import functions as f

model = ISEFlow.load("./ISEFlow/")

predictions, uncertainties = model.predict(X_val)
y_val_unscaled = f.unscale_output(y_val.reshape(-1, 1), "./ISEFlow/scaler_y.pkl")

mse = m.mean_squared_error(y_val_unscaled, predictions)
print(f"MSE: {mse:.4f}")
```

---

## Contributing

We welcome contributions! To get started:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request (PR) for review.

Run tests before submitting:

```sh
pytest tests/
```

---

## Contact & Support

Developed by **Peter Van Katwyk** (Ph.D., Brown University).

- **Email:** [pvankatwyk@gmail.com](mailto:pvankatwyk@gmail.com)
- **GitHub Issues:** [Report a bug](https://github.com/Brown-SciML/ise/issues)

If you use ISE in research, please consider citing our work. See [CITATION.md](CITATION.md) for details.

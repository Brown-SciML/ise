[![Documentation Status](https://readthedocs.org/projects/ise/badge/?version=latest)](https://ise.readthedocs.io/en/latest/)

# Ice Sheet Emulator (ISE) for Emulation of Sea Level Rise

**ISE** is a Python package for training and analyzing **ice sheet emulators**, including **ISEFlow**, a flow-based neural network emulator designed for improved **sea level projections** and **uncertainty quantification**. 

This repository supports emulation for both the **Antarctic** and **Greenland ice sheets**, enabling efficient predictions of **ice volume above flotation (IVAF)** changes using machine learning models.

## 🌍 **About**
ISEFlow and other emulators in this package process **climate forcings** and **IVAF projections** from the **[ISMIP6 simulations](https://app.globus.org/file-manager?origin_id=ad1a6ed8-4de0-4490-93a9-8258931766c7&origin_path=%2FAIS%2F)**.

This codebase has been used in **peer-reviewed research**, including:
- **"A Variational LSTM Emulator of Sea Level Contribution From the Antarctic Ice Sheet"**
- **"ISEFlow: A Flow-Based Neural Network Emulator for Improved Sea Level Projections and Uncertainty Quantification"**  

🔎 **For details on replication**, refer to the [Releases](https://github.com/Brown-SciML/ise/releases) section.

📚 **Documentation:** <https://ise.readthedocs.io/>

---

## 🚀 **Installation**

Install in **editable mode**:
```sh
pip install -e .
```

Or with **[uv](https://github.com/astral-sh/uv)**:
```sh
uv venv
uv pip install -e .
```

---

## 📂 **Project Structure**
```
ise/
├── examples/                # Example scripts for using ISEFlow
├── ise/                     # Main package
│   ├── data/                # Forcing/grid loading, feature engineering, dataset classes
│   │   ├── dataclasses.py   # EmulatorDataset, PyTorchDataset, TSDataset, ScenarioDataset
│   │   ├── feature_engineer.py
│   │   ├── forcings.py      # ForcingFile
│   │   ├── grids.py         # GridFile
│   │   ├── inputs.py        # ISEFlowAISInputs, ISEFlowGrISInputs
│   │   ├── process.py       # ProjectionProcessor, sector processing helpers
│   │   ├── scaler.py        # PyTorch StandardScaler, RobustScaler, LogScaler
│   │   └── utils.py         # convert_and_subset_times()
│   ├── evaluation/          # Metrics
│   │   └── metrics.py
│   ├── models/              # Model architectures
│   │   ├── iseflow.py       # ISEFlow, ISEFlow_AIS, ISEFlow_GrIS
│   │   ├── deep_ensemble.py # DeepEnsemble
│   │   ├── lstm.py          # LSTM
│   │   ├── normalizing_flow.py
│   │   ├── training.py      # CheckpointSaver, EarlyStoppingCheckpointer
│   │   ├── loss.py          # WeightedGridLoss, WeightedMSELoss, and variants
│   │   ├── experimental/    # Legacy models (deprecated)
│   │   └── pretrained/      # Pretrained weights (v1.0.0, v1.1.0)
│   └── utils/               # Data helpers and tensor utilities
├── manuscripts/             # Research paper scripts
├── tests/                   # Unit tests
├── pyproject.toml
└── uv.lock
```

---

## 🏠 **Usage**
### **1️⃣ Loading a Pretrained ISEFlow-AIS Model**
```python
from ise.models import ISEFlow_AIS

# Load the pretrained v1.1.0 AIS emulator
iseflowais = ISEFlow_AIS(version="v1.1.0")
```

### **2️⃣ Running Predictions**
```python
import numpy as np

# Identify Climate Forcings
year = np.arange(2015, 2101)
pr_anomaly = np.array([...])
evspsbl_anomaly = np.array([...])
smb_anomaly = np.array([...])
ts_anomaly = np.array([...])
ocean_thermal_forcing = np.array([...])
ocean_salinity = np.array([...])
ocean_temp = np.array([...])

# Ice Sheet Model Characteristics for projection (see Table A1 Seroussi et al. 2020)
initial_year = 1980
numerics = 'fd'
stress_balance = 'ho'
resolution = 16
init_method = "da"
melt_in_floating_cells = "floating condition"
icefront_migration = "str"
ocean_forcing_type = "open"
ocean_sensitivity = "low"
ice_shelf_fracture = False
open_melt_type = "picop"
standard_melt_type = "nonlocal"

inputs = ISEFlowAISInputs(
    year=year,
    sector=sector,
    pr_anomaly=pr_anomaly,
    evspsbl_anomaly=evspsbl_anomaly,
    smb_anomaly=smb_anomaly,
    ts_anomaly=ts_anomaly,
    ocean_thermal_forcing=ocean_thermal_forcing,
    ocean_salinity=ocean_salinity,
    ocean_temperature=ocean_temp,
    ocean_forcing_type=ocean_forcing_type,
    ocean_sensitivity=ocean_sensitivity,
    ice_shelf_fracture=ice_shelf_fracture,
    stress_balance=stress_balance,
    resolution=resolution,
    numerics=numerics,
    init_method=init_method,
    melt_in_floating_cells=melt_in_floating_cells,
    icefront_migration=icefront_migration,
    open_melt_type=open_melt_type,
    standard_melt_type=standard_melt_type,
    initial_year=initial_year,
)

# OR, TO RUN AS A PARTICULAR MODEL...
# inputs = ISEFlowAISInputs(
#     year=year,
#     sector=sector,
#     pr_anomaly=pr_anomaly,
#     evspsbl_anomaly=evspsbl_anomaly,
#     smb_anomaly=smb_anomaly,
#     ts_anomaly=ts_anomaly,
#     ocean_thermal_forcing=ocean_thermal_forcing,
#     ocean_salinity=ocean_salinity,
#     ocean_temperature=ocean_temp,
#     model_configs="AWI_PISM1"
# )

pred, uq = iseflowais.predict(inputs,)

print(pred)
print(uq['aleatoric'])
print(uq['epistemic'])
```

### **3️⃣ Training a New Model**
```python
from ise.models import ISEFlow, DeepEnsemble, NormalizingFlow

# Load training data
data_directory = r"./ISMIP6-data/"
X_train, y_train, X_val, y_val, X_test, y_test = get_data(data_directory, return_format='numpy')

# Initialize emulator with ISEFlow architecture
de = DeepEnsemble(num_ensemble_members=5, input_size=X_train.shape[1])
nf = NormalizingFlow(input_size=X_train.shape[1])
emulator = ISEFlow(de, nf)

# Fit the model
emulator.fit(X_train, y_train, X_val=X_val, y_val=y_val, )

# Save the model
emulator.save("./ISEFlow/")
```

### **4️⃣ Evaluating Model Performance**
```python
from ise.models import ISEFlow
from ise.evaluation import metrics as m
from ise.utils import functions as f

# Load the previously trained model
emulator = ISEFlow.load("./ISEFlow/")

# Evaluate the model on validation data
predictions, uncertainties = emulator.predict(X_val, output_scaler=f"{data_directory}/scaler_y.pkl")
y_val = f.unscale_output(y_val.reshape(-1,1), f"{data_directory}/scaler_y.pkl")

# Calculate MSE
mse = m.mean_squared_error(y_val, predictions)
print(f"MSE: {mse:0.4f}")     
```

---

## 🛠 **Contributing**
We welcome contributions! To get started:
1. **Fork the repository** on GitHub.
2. **Create a new branch** for your feature or bugfix.
3. **Submit a pull request** (PR) for review.

Run tests before submitting:
```sh
pytest tests/
```

---

## 📌 **Known Issues**

- Test coverage is limited to data and evaluation modules; model-level tests are not yet included.
- Some docstrings were written with the assistance of Generative AI. Please report any inaccuracies via GitHub Issues.

---

## 📧 **Contact & Support**
Developed by **Peter Van Katwyk** (Ph.D., Brown University).

📩 **Email:** [peter_van_katwyk@brown.edu](mailto:peter_van_katwyk@brown.edu)  
🐙 **GitHub Issues:** [Report a bug](https://github.com/Brown-SciML/ise/issues)  

---

If you use this in research, please consider citing our work. See [CITATION.md](CITATION.md) for details.

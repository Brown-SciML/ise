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
ISE uses **[uv](https://github.com/astral-sh/uv)** for dependency management. To set up the environment:

```sh
uv venv
uv pip install -r requirements.txt
```
or using **pip** directly:
```sh
pip install -r requirements.txt
```

To install in **editable mode** (for development):
```sh
pip install -e .
```

---

## 📂 **Project Structure**
```
ise/
├── examples                 # Example scripts for using ISEFlow
├── ise                      # Main ISEFlow package
│   ├── data                 # Data handling and preprocessing
│   ├── evaluation           # Model evaluation
│   ├── models               # ISEFlow model architectures
│   └── utils                # Utility functions
├── LICENSE.md               # License information
├── manuscripts              # Related research papers
├── pyproject.toml           # Project metadata
├── README.md                # ISEFlow documentation
├── requirements.txt         # Required Python dependencies
├── setup.py                 # Installation script
├── tests                    # Unit tests
└── uv.lock                  # Dependency lock file

```

---

## 🏠 **Usage**
### **1️⃣ Loading a Pretrained ISEFlow-AIS Model**
```python
from ise.models.ISEFlow import ISEFlow_AIS

# Load v1.0.0 of ISEFlow-AIS
iseflowais = ISEFlow_AIS.load(version="v1.0.0", )
```

### **2️⃣ Running Predictions**
```python
import numpy as np

# Identify Climate Forcings
year = np.arange(2015, 2101)
pr_anomaly = np.array([-7.0884660e-07,  3.3546070e-06,  ...])
evspsbl_anomaly = np.array([-1.7997656e-06,  8.4536487e-07, ...])
mrro_anomaly = np.array([ 9.14532450e-09, -1.04553575e-08,  ....])
smb_anomaly = np.array([ 1.0817737e-06,  2.5196978e-06,  ....])
ts_anomaly = np.array([-0.6466742 ,  0.00770213, ...  ])
ocean_thermal_forcing = np.array([3.952802, 3.952802, ...  ])
ocean_thermal_forcing = np.array([4.052609 , 4.048029 , ...])
ocean_salinity = np.array([34.538155, 34.54216 , ...])
ocean_temp = np.array([1.4597255, 1.454916 , ...])

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

prediction, uq = iseflowais.predict(
    year, pr_anomaly, evspsbl_anomaly, mrro_anomaly, smb_anomaly, ts_anomaly, ocean_thermal_forcing, ocean_salinity, ocean_temp, initial_year, numerics, stress_balance, resolution, init_method,  melt_in_floating_cells, icefront_migration, ocean_forcing_type, ocean_sensitivity, ice_shelf_fracture, open_melt_type, standard_melt_type
)

print(prediction)
print(uq['aleatoric'])
print(uq['epistemic'])
```

### **3️⃣ Training a New Model**
```python
from ise.models.ISEFlow import ISEFlow, DeepEnsemble, NormalizingFlow

# Load trianing data
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
from ise.models.ISEFlow import ISEFlow
from ise.evaluation import metrics as m
from ise.utils import functions as f

# Load the previously trained model
emulator = ISEFlow.load("./ISEFlow/")

# Evaluate the model on validation data
predictions, uncertainties = emulator.predict(X_val, output_scaler=f"{data_directory}/scaler_y.pkl")
y_val = f.unscale(y_val.reshape(-1,1), f"{data_directory}/scaler_y.pkl")

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

## 📌 **Known Issues & Future Work**
- Creating more unit tests. I know, maybe one day I'll get around it.
- Expanding **support for additional climate scenarios** and additional ISM runs (ISMIP7).
- Better documentation and improvements to the readthedocs page.

---

## 📧 **Contact & Support**
This repository is actively maintained by **Peter Van Katwyk**, Ph.D. student at **Brown University**.

📩 **Email:** [peter_van_katwyk@brown.edu](mailto:peter_van_katwyk@brown.edu)  
🐙 **GitHub Issues:** [Report a bug](https://github.com/Brown-SciML/ise/issues)  

---

🚀 **ISE is a work in progress!** If you use this in research, please consider citing our work. See [CITATION.md](CITATION.md) for details.

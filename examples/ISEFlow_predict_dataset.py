"""Example: run ISEFlow-AIS predictions over a full preprocessed dataset.

This script loads a preprocessed training/test dataset from a directory,
runs ISEFlow-AIS inference on the test split, and saves results to CSV.

Expected directory layout (produced by FeatureEngineer):
    data_directory/
        X_test.csv
        y_test.csv
        scaler_y.pkl
"""

from ise.models.iseflow import ISEFlow_AIS
from ise.utils import get_data, unscale_output
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
data_directory = "/path/to/your/ml/data/"   # update this path
output_csv = "iseflow_ais_preds.csv"
# ──────────────────────────────────────────────────────────────────────────────

iseflowais = ISEFlow_AIS(version="v1.1.0")

X_train, y_train, X_val, y_val, X_test, y_test = get_data(data_directory, return_format="pandas")

y_test_unscaled = unscale_output(y_test.values.reshape(-1, 1), f"{data_directory}/scaler_y.pkl")

preds, uq = iseflowais.predict(X_test)

X_test = X_test.copy()
X_test["aleatoric"] = np.asarray(uq["aleatoric"]).squeeze()
X_test["epistemic"] = np.asarray(uq["epistemic"]).squeeze()
X_test["preds"] = np.asarray(preds).squeeze()
X_test["true"] = y_test_unscaled.squeeze()

X_test.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")
print(f"MSE: {np.mean((y_test_unscaled.squeeze() - X_test['preds'].values) ** 2):.4f}")

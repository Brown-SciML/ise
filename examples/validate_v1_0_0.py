"""Validate ISEFlow v1.0.0 weights against paper-reported test-set MSE.

This script reproduces the held-out test-set evaluation reported for ISEFlow
v1.0.0. It downloads the v1.0.0 test splits and scalers from HuggingFace Hub
(cached after first download) and evaluates against pretrained v1.0.0 weights.

Key difference from v1.1.0
--------------------------
ISEFlow v1.0.0 includes ``mrro_anomaly`` (meltwater runoff anomaly) as an
active input feature for AIS. This variable was removed in v1.1.0 due to
inconsistent availability across CMIP models. The v1.0.0 test splits and
scalers therefore have a different column count than v1.1.0.

Data
----
Test splits and scalers are downloaded from:
  https://huggingface.co/datasets/pvankatwyk/iseflow-datasets  (v1.0.0/)

Weights are resolved via ``get_model_dir("v1.0.0", ice_sheet)`` and
downloaded from ``pvankatwyk/ISEFlow`` on HuggingFace Hub if not cached.
"""

import os
import tempfile

import numpy as np
from huggingface_hub import hf_hub_download

from ise.models.deep_ensemble import DeepEnsemble
from ise.models.iseflow import ISEFlow
from ise.models.normalizing_flow import NormalizingFlow
from ise.models.pretrained import get_model_dir
from ise.utils.functions import get_data, unscale_output

DATASET_REPO = "pvankatwyk/iseflow-datasets"

# ── 1. Evaluate each ice sheet ────────────────────────────────────────────────
#
# For each ice sheet we download the held-out test split and scaler from
# HuggingFace, stage them in a temp directory that matches the layout expected
# by get_data(), then run inference with the v1.0.0 weights and print MSE.

for ice_sheet in ("AIS", "GrIS"):
    prefix = f"v1.0.0/{ice_sheet}"

    # ── 2. Download test split and scaler from HuggingFace Hub ────────────────

    with tempfile.TemporaryDirectory() as data_dir:
        for filename, dest in [
            (f"{prefix}/test.csv", "test.csv"),
            (f"{prefix}/train.csv", "train.csv"),  # get_data needs all splits present
            (f"{prefix}/val.csv", "val.csv"),
            (f"{prefix}/scalers/scaler_X.pkl", "scaler_X.pkl"),
            (f"{prefix}/scalers/scaler_y.pkl", "scaler_y.pkl"),
        ]:
            local_path = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=filename,
                repo_type="dataset",
            )
            os.symlink(local_path, os.path.join(data_dir, dest))

        # ── 3. Load test split ────────────────────────────────────────────────

        _, _, _, _, X_test, y_test = get_data(data_dir, return_format="numpy")
        y_test = unscale_output(y_test.reshape(-1, 1), os.path.join(data_dir, "scaler_y.pkl"))

        # ── 4. Load pretrained v1.0.0 weights ─────────────────────────────────

        model_dir = get_model_dir("v1.0.0", ice_sheet)
        de = DeepEnsemble.load(f"{model_dir}/deep_ensemble.pth")
        nf = NormalizingFlow.load(f"{model_dir}/normalizing_flow.pth")
        model = ISEFlow(de, nf)
        model.model_dir = model_dir

        # ── 5. Predict and compute MSE ────────────────────────────────────────

        preds, _ = model.predict(X_test)
        mse = np.mean((y_test - preds) ** 2)
        print(f"{ice_sheet} MSE: {mse:.4f}  (expected: V1.0.0 AIS ~1.20, GrIS ~1.02)")

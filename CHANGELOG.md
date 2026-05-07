# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses **two independent version numbers**:

- **Package version** (`ise-py` on PyPI) — follows [Semantic Versioning](https://semver.org/).
- **Model version** (ISEFlow weights on HuggingFace Hub) — `v1.0.0`, `v1.1.0`, etc.
  Model versions only change when new pretrained weights are released.

---

## [Unreleased]

---

## [1.0.0] — 2026-05-07 (package) | Model: v1.1.0

> First release of `ise-py` on PyPI. This version represents a full rewrite of the
> original `ise` package, introducing ISEFlow as the primary model, GrIS support,
> HuggingFace Hub weight distribution, validated input dataclasses, anomaly conversion,
> and a complete test suite and CI/CD pipeline.

### Changed
- **Breaking:** Package renamed from `ise` to `ise-py` on PyPI; import name stays `ise`.
- **Breaking:** `from_raw_values()` renamed to `from_absolute_forcings()` on both
  `ISEFlowAISInputs` and `ISEFlowGrISInputs`; old name kept as a deprecated alias
  (emits `DeprecationWarning`) until v3.0.0.
- **Breaking:** Model classes restructured — `ise.models.ISEFlow.ISEFlow`,
  `ise.models.predictors.deep_ensemble`, and `ise.models.density_estimators.normalizing_flow`
  replaced by top-level `ISEFlow`, `DeepEnsemble`, `NormalizingFlow`, and `LSTM` in
  `ise.models`.
- Pretrained weights moved to HuggingFace Hub (`pvankatwyk/ISEFlow`); downloaded
  automatically on first use via `huggingface_hub`. Falls back to bundled local weights
  when HuggingFace is unavailable (air-gapped HPC / local dev).
- Build system switched from `setup.py` + `setuptools` to Hatchling (`pyproject.toml`);
  runtime dependencies cleaned up — removed transitive pins, moved dev/docs/GPU deps to
  optional extras. Removed heavy deps from core: `seaborn`, `cartopy`, `geopandas`,
  `pyproj`, `xesmf`, `clisops`, `owslib`, `statsmodels`.
- Replaced `flake8` + `isort` + `black` with `ruff`; updated `pyproject.toml` and
  pre-commit config.
- mypy fixes: replaced `= None` defaults with `| None` unions across `forcings.py`,
  `grids.py`, `inputs.py`, `process.py`, `feature_engineer.py`, `training.py`,
  `functions.py`, and `anomaly.py`; explicit type annotations on `ForcingFile` and
  `GridFile` instance attributes.
- `ISEFlow_AIS.predict` / `ISEFlow_GrIS.predict` marked `# type: ignore[override]` to
  satisfy mypy while preserving the intentional signature narrowing.
- Variable shadowing fixed in `ProjectionProcessor` and `get_model_densities`.
- `backfill_outliers` docstring corrected (bfill, not ffill).
- `AnomalyConverter` updated to support GrIS MAR integration; GrIS anomaly variables
  (`aSMB`, `aST`) pass through unchanged (already anomalies in ISMIP6).
- NaN handling added in NormalizingFlow sampling.
- Introduced separate package vs. model versioning (see above).

### Added
- **`ISEFlow_AIS` and `ISEFlow_GrIS`** — convenience subclasses of `ISEFlow` for
  AIS (18 sectors) and GrIS (6 basins) with pretrained weight loading.
- **`ISEFlowAISInputs` / `ISEFlowGrISInputs`** — validated input dataclasses for
  `model.predict()`, with `from_absolute_forcings()` classmethod for raw forcing values.
- **`AnomalyConverter`** — converts absolute sector-averaged forcings to ISMIP6
  anomalies using bundled climatology CSVs; called internally by `from_absolute_forcings()`.
- **`ForcingFile`** (`ise/data/forcings.py`) — loads and formats climate forcing NetCDF
  files to 86-step ISMIP6 time series, with depth aggregation and sector assignment.
- **`GridFile`** (`ise/data/grids.py`) — loads sector boundary grids.
- **`NormalizingFlow`** (`ise/models/normalizing_flow.py`) — standalone autoregressive
  masked affine flow replacing the old `density_estimators/normalizing_flow.py`.
- **`LSTM`** (`ise/models/lstm.py`) — standalone LSTM replacing `predictors/lstm.py`,
  with variable sequence length and save/load support.
- **`DeepEnsemble`** (`ise/models/deep_ensemble.py`) — rewritten ensemble, trained on
  `[X, z]` where `z` is the NF latent; captures epistemic uncertainty.
- **`ise/data/utils.py`** — `convert_and_subset_times()` for xarray time handling.
- **`ise/utils/io.py`** — `check_type()` runtime type validation helper.
- **`ise/data/data_files/ismip6_model_configs.json`** and
  `ise/data/data_files/GrIS_ismip6_model_configs.json` — ISM configuration lookup tables.
- **`ise/data/data_files/AIS_atmos_climatologies.csv`** and
  `ise/data/data_files/GrIS_atmos_climatologies.csv`** — bundled ISMIP6 climatology
  baselines used by `AnomalyConverter`.
- ISEFlow model weights (v1.1.0) for both AIS and GrIS on HuggingFace Hub.
- `ise.__version__` attribute (read from package metadata via `importlib.metadata`).
- `ise/py.typed` marker for PEP 561 type-checking support.
- `[tool.mypy]`, `[tool.pytest.ini_options]`, `[tool.coverage.run]` sections in
  `pyproject.toml`.
- `.pre-commit-config.yaml`, `.editorconfig`, `Makefile`.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `CITATION.cff`.
- GitHub Actions CI workflow (lint + test on Python 3.11 & 3.12), mypy CI job, and
  release workflow (OIDC trusted publisher).
- GitHub Actions issue templates.
- Slow and GPU pytest markers; `skip_if_no_gpu` helper in `conftest.py`.
- `__all__` exports defined in all subpackage `__init__.py` files.
- Full test suite covering anomaly conversion, input dataclasses, all model components,
  scalers, loss functions, training utilities, and metrics (244 tests).
- Example scripts: `example_ais.py`, `example_gris.py`, `example_absolute_forcings.py`,
  `process_training_data.py`.
- `wandb` added as a core dependency (used in NF and LSTM training).

### Removed
- `ise/models/ISEFlow/` directory (old monolithic `ISEFlow.py`, `de.py`, `nf.py`).
- `ise/models/predictors/` directory (`lstm.py`, `deep_ensemble.py`).
- `ise/models/density_estimators/` directory (`normalizing_flow.py`).
- `setup.py` (replaced by `pyproject.toml` + Hatchling).
- `variational_lstm_emulator.pt` legacy weight file (unused).
- Stray `ISEFlow_GrIS_v1-1-0 copy/` duplicate weights directory.
- Old example scripts `ISEFlow_from_NC.py` and `ISEFlow_predict.py`.

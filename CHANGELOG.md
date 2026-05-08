# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses **two independent version numbers**:

- **Package version** (`ise-py` on PyPI) — follows [Semantic Versioning](https://semver.org/).
- **Model version** (ISEFlow weights on HuggingFace Hub) — `v1.0.0`, `v1.1.0`, etc.
  Model versions only change when new pretrained weights are released.

---

## [Unreleased]

- Fixed `sum_by_sector` `UnboundLocalError` when `grid_file` is an `xr.Dataset`. The `xr.Dataset` branch read `grids.Description` but `grids` was only assigned in the `str` branch. Now binds `grids = grid_file` and reads `Description` via `attrs.get()` for safety.
- `ISEFlow_GrIS_DE_v1_0_0` and `ISEFlow_GrIS_NF_v1_0_0` now emit `DeprecationWarning` on instantiation, matching the behaviour of their AIS counterparts.
- Fixed four `inputs.py` validation issues: AIS/GrIS `_check_inputs` now coerces `year` to ndarray before arithmetic so passing `year=list(...)` no longer raises `TypeError`; `'False'` removed from `melt_in_floating_cells` accepted values (it was never a valid encoding and caused a delayed `KeyError` in `_map_args`); GrIS `surface_thickness` validator error message corrected (was copy-pasted from the `bed` validator); `_map_args` is now idempotent so calling `__post_init__` twice no longer `KeyError`s.
- Fixed `NormalizingFlow.save()` crash when called after `fit(save_checkpoints=False)`. `save()` read `self.best_loss` and `self.epochs_trained` directly, but those attributes are only set when a checkpoint is written. Now uses `getattr` defaults to match `DeepEnsemble.save()`.
- Fixed `EmulatorDataset.__getitem__` crash on 3-D `(N_proj, T, F)` input. `__init__` set `num_features` for the 3-D branch but `features` for the 2-D branch; `__getitem__` only used the latter, raising `AttributeError` on first index lookup. Standardised on `num_features` in both branches (`self.features` retained as an alias).

- Fixed `ISEFlow.fit()` so it can actually train. The internal call to `NormalizingFlow.fit()` was passing positional arguments that no longer matched the NF signature (`X_val`/`y_val` were inserted in front of `epochs`/`batch_size` at some point), causing every fresh training run to crash before the first epoch. Switched to keyword arguments.
- Fixed `ISEFlow.__init__` so it accepts a freshly-constructed `NormalizingFlow`. It previously read `normalizing_flow.model_dir` directly, but that attribute is only set by `NormalizingFlow.load()`, so untrained NFs could not be wrapped in an ISEFlow. Now uses `getattr(..., None)`.
- Fixed `NormalizingFlow.fit()` off-by-one loop guard (`start_epoch < epochs` → `<=`). Single-epoch fits previously skipped the entire training loop and then crashed trying to load the unwritten checkpoint. Also made the post-training checkpoint load defensive (skips if no file was written).
- Fixed `LSTM.fit()` so it writes checkpoints when training without a validation split. The checkpointer was only invoked inside `if validate:`, so train-only runs ended with no checkpoint file, then crashed on the post-training load. Now also checkpoints on training loss in the no-validation branch, and the post-training load is defensive.
- Fixed `DeepEnsemble.save()` to be repeatable and tolerant of missing per-member training attributes. It now uses `getattr` defaults for `best_loss`, `epochs_trained`, and `sequence_length`, and the leftover-checkpoint cleanup tolerates already-deleted files. Previously, a model trained with `save_checkpoints=False` couldn't be saved at all, and any model could only be saved once.
- Fixed v1.0.0 `ISEFlow_AIS.process()` `year_mean_map` off-by-one. The map was keyed `0..85` but `data["year"]` is in model encoding `1..86`, so any NaN in the last year crashed with `KeyError: 86`. Now keyed `1..86` to match.
- Fixed `ISEFlowAISInputs._convert_arrays` so optional `mrro_anomaly=None` is preserved as `None` instead of being coerced to `np.array(None, dtype=object)`. This restores the documented `inputs.mrro_anomaly is None` check used by `ISEFlow_AIS(version="v1.0.0").process()` to raise a clear `ValueError` when `mrro_anomaly` is missing.
- Substantially extended the test suite (244 → 304 tests). New coverage areas: end-to-end ISEFlow `fit`/`save`/`load` integration, real pretrained `ISEFlow_AIS`/`ISEFlow_GrIS.predict()` from synthetic inputs, `ise.models.pretrained` weight resolution and HF fallback, `add_lag_variables` cross-projection bleed, year encoding, validation edge cases, miscellaneous utility round-trips, repeatable `save()`, and v1.0.0 NaN-mrro fallback.
- Restored backward compatibility for ISEFlow v1.0.0 weights. `ISEFlow_AIS(version="v1.0.0")` and `ISEFlow_GrIS(version="v1.0.0")` now load and predict end-to-end with the v1.0.0 pretrained weights. v1.1.0 behaviour is unchanged.
  - `NormalizingFlow.load()` detects v1.0.0 metadata (missing `flow_hidden_size`/`num_flows` keys) and reconstructs the original architecture: a single-`nn.Linear` context encoder with `flow_hidden_features = output_size * 2` and `num_flow_transforms=5`.
  - `ISEFlow_AIS.process()` and `ISEFlow_GrIS.process()` gained a v1.0.0 path matching the historical preprocessing order (lag → one-hot → reindex → append `outlier=False` → positional StandardScaler.transform → drop `outlier`). v1.1.0 path is untouched.
  - Populated `ISEFlow_GrIS_v1_0_0_variables` (90 features) — was previously an empty stub.
  - `DeepEnsemble.load()` now defaults `member.sequence_length=5` when the saved metadata omits it (older v1.0.0 metadata).
- Updated install instructions in README and docs to use `pip install ise-py` and `pip install -e ".[dev]"`; removed uv/requirements.txt references.
- Fixed HuggingFace Hub URL casing in README.
- Added warning suppression for properscoring and scikit-learn dependencies around Syntax and versionning.

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
  (emits `DeprecationWarning`) until further versions.
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

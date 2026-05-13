# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses **two independent version numbers**:

- **Package version** (`ise-py` on PyPI) — follows [Semantic Versioning](https://semver.org/).
- **Model version** (ISEFlow weights on HuggingFace Hub) — `v1.0.0`, `v1.1.0`, etc.
  Model versions only change when new pretrained weights are released.

---

## [Unreleased]

### Added
- `docs/index.rst` — PyPI, Python version, License, and CI badges to match `README.md`. Previously only the ReadTheDocs badge was shown.
- `ise.models.pretrained.get_model_dir()` now prints a clear stderr message when ISEFlow weights are being downloaded from HuggingFace Hub for the first time (and another when the download finishes). When weights are already cached, the loader stays silent — previously the call could appear to hang while HF metadata sync ran.

### Fixed
- Silenced `sklearn.exceptions.InconsistentVersionWarning` package-wide. Bundled pretrained scalers were pickled with an older sklearn version but unpickle correctly; the warning was noise on every example run. Filter installed in `ise/__init__.py` next to the existing properscoring escape-sequence filter.
- `feature_engineer.scale_data()` and `FeatureEngineer.scale_data()` leaked file handles by calling `pickle.load(open(...))` without a context manager — switched to `with open(...) as f` to close deterministically and clear `ResourceWarning` noise.
- `from ise import ISEFlow, ISEFlow_AIS, ISEFlow_GrIS` now works. The top-level `__all__` listed these names but never imported them, so the imports failed with `ImportError`. Added the imports from `ise.models`.
- `ISEFlowGrISInputs._assign_model_configs` defaulted to the **AIS** model-configs JSON, so passing a GrIS-only ISM name (e.g. `model_configs="AWI-ISSM1"`) raised `ValueError: Model name ... not found`. Added a new `gris_ismip6_model_configs_path` constant in `ise/utils/__init__.py` pointing at `GrIS_ismip6_model_configs.json` and switched the GrIS dataclass to use it as the default. Regression test added.
- `unscale_output()` / `unscale_input()` in `ise/utils/functions.py` leaked file handles via `pkl.load(open(...))` — switched to a `with open(...) as f` context manager (same pattern as the recent `feature_engineer` fix).
- Replaced a bare `except:` clause in `ise.utils.functions.load_ml_data` with `except FileNotFoundError:` so that `KeyboardInterrupt`, `SystemExit`, and `MemoryError` are no longer silently swallowed.

### Changed
- `ProjectionProcessor` ocean-forcing warnings (AIS and GrIS) now list the directory, the expected NetCDF variable names, and the files that *were* found, instead of an opaque "Directory X does not contain N files." message.
- Moved the in-body `from scipy.ndimage import uniform_filter1d` to the top of `ise/models/iseflow.py` next to the other module imports (style cleanup; PEP 8 E402).

### Documentation
- Full audit pass across `ise/` docstrings. Notable fixes:
  - `ISEFlow.forward` removed a phantom `smooth_projection` argument that did not exist in the signature.
  - `ISEFlow.predict` converted a misleading `Raises: Warning` clause into a proper Sphinx `Warns` block and expanded the description of how scaler resolution and smoothing interact.
  - `ISEFlow.load` removed a stale `Raises: NotImplementedError` claim that no longer matched the implementation.
  - `NormalizingFlow.fit` documented the previously undocumented `X_val`, `y_val`, `lr`, and `wd` arguments.
  - `NormalizingFlow.aleatoric` corrected the return-shape claim (per-row std `(N,)`, not `(num_samples,)`).
  - `LSTM.fit` added the previously undocumented `wandb_run` argument.
  - `RobustScaler.save/load` and `LogScaler.save/load` gained docstrings (only `StandardScaler` had them).
  - `unscale_output` / `unscale_input` corrected: they accept any sklearn scaler with `inverse_transform`, not only `MinMaxScaler`.
- `docs/model_versions.rst`: corrected the GrIS v1.1.0 input list (`aSMB`, `aST`, `sector`, etc. — not the prior `smb_anomaly`, `st_anomaly`, `region` placeholder), and replaced an unverified `aogcm="MIROC6"` example with a real bundled name (`noresm1-m_rcp85`).
- `.gitignore`: ignore `.bugs.md` (local bug-triage scratchpad produced during the docs audit).

---

## [1.2.0] — 2026-05-11 (package) | Model: v1.1.0

### Added
- `NormalizingFlow.get_latent_representation()` now supports v1.0.0 weights via a `legacy_v1_0_0` path: pushes a zero vector through the forward transform (deterministic) rather than sampling from the conditional base distribution. `NormalizingFlow.load()` auto-detects the version from saved metadata and sets the flag automatically, so v1.0.0 and v1.1.0 inference are both correct without user intervention.
- `docs/model_versions.rst` — new reference page documenting all ISEFlow weight versions, per-version input schemas, migration guides (v1.0.0 → v1.1.0), `from_absolute_forcings()` usage, and weight resolution order. Linked from `docs/index.rst`.
- `examples/recreate_manuscript_results.py` — end-to-end script that downloads v1.0.0 test splits and scalers from HuggingFace Hub (`pvankatwyk/iseflow-datasets`), runs inference with v1.0.0 pretrained weights, and reports held-out MSE for both AIS and GrIS.

### Changed
- `NormalizingFlow.get_latent_representation()` docstring expanded to document both the legacy deterministic path (v1.0.0) and the stochastic sampling path (v1.1.0+), including the behavioural difference and its downstream effect on `DeepEnsemble` inputs.

---

## [1.1.0] — 2026-05-08 (package) | Model: v1.1.0

### Added
- Restored full backward compatibility for v1.0.0 pretrained weights. `ISEFlow_AIS(version="v1.0.0")` and `ISEFlow_GrIS(version="v1.0.0")` now load and predict end-to-end; `NormalizingFlow.load()` auto-detects the older architecture from saved metadata.
- Extended test suite from 244 to 304 tests, adding end-to-end `fit`/`save`/`load` integration, pretrained `predict()` from synthetic inputs, and v1.0.0 compatibility coverage.

### Fixed
- `ISEFlow.fit()` was broken: `NormalizingFlow.fit()` was called with mismatched positional arguments, crashing before the first epoch. Switched to keyword arguments.
- `NormalizingFlow.fit()` off-by-one loop guard caused single-epoch fits to skip training entirely.
- `LSTM.fit()` did not write checkpoints on train-only runs, causing a crash on the post-training load.
- `DeepEnsemble.save()` / `NormalizingFlow.save()` failed when trained with `save_checkpoints=False`; now uses `getattr` defaults for missing training attributes.
- `ISEFlow.fit()` silently overrode the caller's random seed via `torch.manual_seed(random_int)`; removed to restore reproducibility.
- `unscale_column` year range was off by one (2016–2100 instead of 2015–2100); sector scaler was hard-coded to AIS. Now accepts an `ice_sheet` kwarg for correct GrIS (1–6) ranges.
- `feature_engineer.py`: split results were wiped to `None` immediately after creation; `scale_data` raised `NameError` in the explicit-X branch; `split_training_data` now honours `random_state` via `np.random.default_rng`.
- `inputs.py`: `year=list(...)` raised `TypeError`; `_map_args` was not idempotent; invalid `melt_in_floating_cells` value `'False'` accepted.
- `ISEFlow.save()` now emits a `UserWarning` instead of silently swallowing a missing scaler path.
- Fixed HuggingFace Hub `snapshot_download` to use recursive glob so nested weight files are included.
- `ISEFlow_GrIS_DE/NF_v1_0_0` now emit `DeprecationWarning` on instantiation, matching AIS counterparts.

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

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses **two independent version numbers**:

- **Package version** (`ise-py` on PyPI) — follows [Semantic Versioning](https://semver.org/).
- **Model version** (ISEFlow weights on HuggingFace Hub) — `v1.0.0`, `v1.1.0`, etc.
  Model versions only change when new pretrained weights are released.

---

## [Unreleased]

### Changed
- mypy fixes: replaced `= None` defaults with `| None` unions across `forcings.py`,
  `grids.py`, `inputs.py`, `process.py`, `feature_engineer.py`, `training.py`,
  `functions.py`, and `anomaly.py`.
- `ForcingFile` and `GridFile` instance attributes given explicit type annotations; methods
  assert data is loaded before use.
- `ISEFlow_AIS.predict` / `ISEFlow_GrIS.predict` marked `# type: ignore[override]`.
- Variable shadowing fixed in `ProjectionProcessor` and `get_model_densities`.
- `calculate_distribution_metrics`: removed stale `year=2100` kwarg passed to
  `create_distribution`.
- Removed heavy optional deps from core requirements: `seaborn`, `cartopy`, `geopandas`,
  `pyproj`, `xesmf`, `clisops`, `owslib`, `statsmodels`.
- `backfill_outliers` docstring corrected (bfill, not ffill).
- `__all__` exports defined in all subpackage `__init__.py` files.
- Replaced `flake8` + `isort` + `black` with `ruff`; updated `pyproject.toml` and
  pre-commit config.

### Added
- mypy CI job in GitHub Actions.
- Slow and GPU pytest markers; `skip_if_no_gpu` helper in `conftest.py`.
- GitHub Actions issue templates and release workflow.
- `Makefile` `type` target for running mypy.

---

## [1.0.0] — 2026-05-06 (package) | Model: v1.1.0

> This is the first tracked release of `ise-py` on PyPI. Substantial development
> preceded this release (including the original AIS emulator, GrIS support, and
> multiple rounds of model training) but was not tracked in a changelog. Starting here.

### Changed
- **Breaking:** Package renamed from `ise` to `ise-py` on PyPI; import name stays `ise`.
- **Breaking:** `from_raw_values()` renamed to `from_absolute_forcings()` on both
  `ISEFlowAISInputs` and `ISEFlowGrISInputs`; old name kept as a deprecated alias
  (emits `DeprecationWarning`) until v3.0.0.
- Pretrained weights moved to HuggingFace Hub (`Brown-SciML/ISEFlow`); downloaded
  automatically on first use via `huggingface_hub`. Falls back to bundled local weights
  when HuggingFace is unavailable.
- Build system switched to Hatchling; runtime dependencies cleaned up (removed
  transitive pins, moved dev/docs/GPU deps to optional extras).
- Introduced separate package vs. model versioning (see above).

### Added
- `ise.__version__` attribute.
- `ise/py.typed` marker for PEP 561 type-checking support.
- `[tool.black]`, `[tool.isort]`, `[tool.mypy]`, `[tool.pytest.ini_options]`,
  `[tool.coverage.run]` sections in `pyproject.toml`.
- `.flake8`, `.pre-commit-config.yaml`, `.editorconfig`, `Makefile`.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`.
- GitHub Actions CI workflow (lint + test on Python 3.11 & 3.12) and release workflow.
- `conftest.py` with shared pytest fixtures.

### Removed
- Stray `ISEFlow_GrIS_v1-1-0 copy/` duplicate weights directory.
- `variational_lstm_emulator.pt` legacy file (unused).


# Contributing to ise-py

Thank you for your interest in contributing! This guide covers how to set up your development
environment, run tests, follow code style conventions, and submit pull requests.

---

## Development setup

```bash
git clone https://github.com/Brown-SciML/ise.git
cd ise

# Create a virtual environment (uv recommended)
uv venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Running tests

```bash
pytest tests/
```

To run with coverage:

```bash
pytest tests/ --cov=ise --cov-report=term-missing
```

---

## Code style

This project uses **black**, **isort**, and **flake8** with a line length of 100.

Format code automatically:

```bash
make format
# or manually:
black . && isort .
```

Check without modifying:

```bash
make lint
# or manually:
black --check . && isort --check . && flake8 .
```

Pre-commit hooks run these checks automatically before each commit once installed.

---

## Pre-commit hooks

After installing dev dependencies, enable the hooks with:

```bash
pre-commit install
```

The hooks run **black**, **isort**, and **flake8** on every commit. To run all hooks
manually at any time:

```bash
pre-commit run --all-files
```

---

## Adding a new AOGCM climatology

1. Compute the sector-averaged baseline means for the new AOGCM (1995–2014 for AIS;
   1960–1989 MAR for GrIS).
2. Append a row to the appropriate CSV file:
   - AIS: `ise/data/data_files/AIS_atmos_climatologies.csv`
   - GrIS: `ise/data/data_files/GrIS_atmos_climatologies.csv`
3. The AOGCM name must match the key users will pass as `aogcm=` in
   `ISEFlowAISInputs.from_absolute_forcings()` / `ISEFlowGrISInputs.from_absolute_forcings()`.
   See `AnomalyConverter._normalise_aogcm_name()` for the normalisation logic.
4. Add a test in `tests/ise/data/test_anomaly.py` verifying the new entry is found.

---

## Adding a new ISM configuration

1. Open `ise/data/data_files/ismip6_model_configs.json`.
2. Add a new JSON object with the model key and all required fields (see existing
   entries for the schema).
3. Add a test in `tests/ise/data/` verifying the new config is accepted by
   `ISEFlowAISInputs` or `ISEFlowGrISInputs`.

---

## Pull request process

1. Fork the repository and create a feature branch from `master`.
2. Ensure all tests pass: `pytest tests/`.
3. Ensure the code passes all linting checks: `make lint`.
4. Write or update tests for any new behaviour.
5. Update `CHANGELOG.md` under the `[Unreleased]` section.
6. Open a pull request against `master` with a clear description of the change
   and the motivation for it.

---

## Questions?

Open a [GitHub issue](https://github.com/Brown-SciML/ise/issues) or contact
[pvankatwyk@gmail.com](mailto:pvankatwyk@gmail.com).

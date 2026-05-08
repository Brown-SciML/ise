# Pre-Release Audit Plan — `ise-py` v1.0.0 → next merge

> **For the implementing agent (Claude Sonnet):**
> Read this entire "How to use this plan" section before touching any file.
> The fixes below are ordered. Work through them top-to-bottom. Each item
> is self-contained: file, exact location, reproduction, fix recipe with
> code, regression test to add, and the commit message to use. Some items
> are explicitly marked **STOP AND ASK** — do not proceed past those without
> explicit user approval.

---

## How to use this plan

### Ground rules
1. **Make one commit per item.** Each section below has a `Commit message:` line. Use it verbatim, with the body. Do not batch unrelated fixes.
2. **Every commit must update CHANGELOG.md.** Project CLAUDE.md mandates this. Add a bullet under `## [Unreleased]` describing the fix in user-facing terms (not "fixed bug 0.1").
3. **Every code fix must come with the regression test in the same commit.** Test paths are given inline. If you can't write a test for some reason, stop and ask.
4. **Do not coauthor commits with Claude.** Project CLAUDE.md is explicit about this.
5. **Do not push, tag, or amend commits.** Only `git add` + `git commit`.
6. **Do not bypass hooks** (`--no-verify`) under any circumstances.
7. **Run the affected tests after each commit.** Use `python -m pytest tests/<the test file> -q` for fast feedback. Run the full `python -m pytest tests -m "not slow and not gpu" -q` after every P0 commit.
8. **Stay within scope.** If you spot an unrelated bug while editing a file, do not fix it — note it and continue. The user wants reviewable, atomic commits.
9. **STOP AND ASK markers** mean exactly that: pause, summarize the choice, wait for the user. Don't decide for them.

### Execution order
The list below is in **dependency order**. Items in the same file are batched together so you don't reopen `inputs.py` four times. Don't reorder.

1. P0.1 — `EmulatorDataset` 3-D bug (`dataclasses.py`)
2. P0.4 — `NormalizingFlow.save()` defensive (`normalizing_flow.py`)
3. P0.5 + P0.6 + P1.3 + P2.9 — All `inputs.py` issues, single batch
4. P0.7 — GrIS deprecation warnings (`iseflow.py`)
5. P0.8 — `sum_by_sector` UnboundLocalError (`metrics.py`)
6. P0.2 + P0.3 + P1.2 + P2.3 + P2.7 — All `feature_engineer.py` issues, single batch
7. P1.1 + P3.1 — `unscale_column` + `unscale_input` (`utils/functions.py`)
8. P1.4 + P1.5 — Loss tensor handling (`loss.py`)
9. P1.6 + P1.7 + P1.8 + P3.2 — All `lstm.py` issues
10. P1.9 — `ISEFlow.fit` reseeding (`iseflow.py`)
11. P1.10 — **STOP AND ASK** before this one
12. P2.1, P2.2, P2.4, P2.5, P2.6, P2.8, P2.10 — Polish pass
13. P2.11 — HF allow_patterns hardening
14. P3.3, P3.4, P3.6 — Future polish (skip if user says so)
15. P2.12 — Version bump (last; only after all above land green)

### Verification gate
Before declaring done, all 10 verification steps at the bottom of this file must pass. Do not skip any.

---

## Audit context

The package was read end-to-end. The test suite passes (268 passed, 36 deselected with `pytest -m "not slow and not gpu"`) but covers only the happy paths — none of the bugs below are caught. All P0 items were reproduced with one-liners (notes inline).

Issues are grouped by severity. Each item has:
- **Where:** file + line range
- **What:** the bug
- **Reproduction:** one-line test
- **Fix:** before/after code
- **Test to add:** path and outline
- **Commit message:** to use verbatim

---

## P0 — Critical: must fix before publishing

### P0.1 `EmulatorDataset.__getitem__` crashes on 3-D input

- **Where:** [ise/data/dataclasses.py:79-174](ise/data/dataclasses.py#L79-L174). Constructor is lines 79-103; `__getitem__` is lines 139-174.
- **What:** `__init__` sets `self.num_features` for the `xdim == 3` branch (line 99) but `self.features` for the `xdim == 2` branch (line 101). `__getitem__` references `self.features` at line 154 and 165 unconditionally. 3-D input crashes with `AttributeError`.
- **Reproduction:**
  ```python
  import numpy as np
  from ise.data.dataclasses import EmulatorDataset
  ds = EmulatorDataset(np.random.randn(10, 86, 5), np.random.randn(860, 1), sequence_length=3)
  ds[0]   # AttributeError
  ```
- **Fix:** in `__init__` lines 98-103, set `self.features = self.X.shape[-1]` in *both* branches. Replace lines 98-103 with:
  ```python
  if self.xdim == 3:  # Batched by projection
      self.num_projections, self.num_timesteps, self.num_features = X.shape
  elif self.xdim == 2:  # Unbatched (rows of projections*timestamps)
      self.projections_and_timesteps, _ = X.shape
      self.num_timesteps = projection_length
      self.num_projections = self.projections_and_timesteps // self.num_timesteps
      self.num_features = X.shape[1]
  self.features = self.num_features
  ```
  Then change line 154 (`torch.zeros((self.sequence_length, self.features))`) to use `self.num_features` for clarity (`self.features` will still work as alias). Standardise on `self.num_features` everywhere.
- **Test to add:** in [tests/ise/data/test_dataclasses.py](tests/ise/data/test_dataclasses.py), add `test_emulator_dataset_accepts_3d_input` that builds a `(2, 86, 5)` array, instantiates `EmulatorDataset(X, y, sequence_length=3)`, and verifies `ds[0]` returns a tensor of shape `(3, 5)`.
- **Commit message:**
  ```
  fix EmulatorDataset 3-D input crash

  __init__ set num_features for the 3-D branch but features for the 2-D
  branch; __getitem__ only used the latter, so any 3-D (N_proj, T, F) input
  raised AttributeError on first index lookup. Standardised on num_features
  and added a regression test.
  ```

---

### P0.2 `FeatureEngineer.__init__` overwrites split results with `None`

> Implemented as part of the `feature_engineer.py` batch — see step 6.

---

### P0.3 `split_training_data` ignores `random_state`

> Implemented as part of the `feature_engineer.py` batch — see step 6.

---

### P0.4 `NormalizingFlow.save()` crashes after `save_checkpoints=False` train

- **Where:** [ise/models/normalizing_flow.py:418-453](ise/models/normalizing_flow.py#L418-L453). The crash is in the `metadata = {...}` dict construction at lines 431-439.
- **What:** Reads `self.best_loss` and `self.epochs_trained` directly. These are set only inside the `if save_checkpoints and os.path.exists(checkpoint_path):` branch in `fit()` (lines 326-344). A user calling `nf.fit(..., save_checkpoints=False)` then `nf.save(...)` hits `AttributeError: 'NormalizingFlow' object has no attribute 'best_loss'`.
- **Reproduction:**
  ```python
  from ise.models.normalizing_flow import NormalizingFlow
  import torch
  nf = NormalizingFlow(input_size=5, output_size=1, num_flow_transforms=2)
  nf.fit(torch.randn(86, 5), torch.randn(86, 1), epochs=1, batch_size=8,
         save_checkpoints=False, verbose=False)
  nf.save('/tmp/test_nf.pth')   # AttributeError
  ```
- **Fix:** Replace lines 431-439 with `getattr` defaults (mirror what `DeepEnsemble.save()` already does at [ise/models/deep_ensemble.py:299-318](ise/models/deep_ensemble.py#L299-L318)):
  ```python
  metadata = {
      "input_size": self.num_input_features,
      "output_size": self.num_predicted_sle,
      "device": self.device,
      "best_loss": float(getattr(self, "best_loss", float("inf"))),
      "epochs_trained": int(getattr(self, "epochs_trained", 0)),
      "flow_hidden_size": self.flow_hidden_features,
      "num_flows": self.num_flow_transforms,
  }
  ```
- **Test to add:** in [tests/ise/models/test_normalizing_flow.py](tests/ise/models/test_normalizing_flow.py), add `test_save_after_no_checkpoint_train` that mirrors the reproduction above and checks the metadata JSON file exists and has `best_loss`/`epochs_trained` keys.
- **Commit message:**
  ```
  fix NormalizingFlow.save crash after checkpoint-less training

  save() read self.best_loss / self.epochs_trained directly, but those
  attrs are only set when fit() runs with save_checkpoints=True and writes
  a checkpoint. Calling fit(save_checkpoints=False) then save() raised
  AttributeError. Now uses getattr defaults to match DeepEnsemble.save.
  ```

---

### P0.5 + P0.6 + P1.3 + P2.9 — `inputs.py` batch

This is a single commit covering four related issues in [ise/data/inputs.py](ise/data/inputs.py). Each fix is independent but they all touch validation logic, so batching is cleanest.

#### P0.5 `ISEFlowAISInputs` rejects Python list `year`

- **Where:** [ise/data/inputs.py:319-331](ise/data/inputs.py#L319-L331) (AIS) and [ise/data/inputs.py:885-897](ise/data/inputs.py#L885-L897) (GrIS).
- **What:** `_check_inputs` runs before `_convert_arrays` in `__post_init__`. Line 331 (AIS) does `self.year - 2015 + 1`, requiring `year` to already be a numpy array. Same for GrIS line 897.
- **Reproduction:**
  ```python
  from ise.data.inputs import ISEFlowAISInputs
  import numpy as np
  ISEFlowAISInputs(year=list(range(2015, 2101)), sector=10,
                   pr_anomaly=np.zeros(86), evspsbl_anomaly=np.zeros(86),
                   smb_anomaly=np.zeros(86), ts_anomaly=np.zeros(86),
                   ocean_thermal_forcing=np.zeros(86), ocean_salinity=np.zeros(86),
                   ocean_temperature=np.zeros(86),
                   ice_shelf_fracture=False, ocean_sensitivity='medium',
                   initial_year=2005, numerics='fd', stress_balance='hybrid',
                   resolution='8', init_method='eq',
                   melt_in_floating_cells='sub-grid', icefront_migration='str',
                   ocean_forcing_type='open', open_melt_type='quad')
  # TypeError: unsupported operand type(s) for -: 'list' and 'int'
  ```
- **Fix:** at the top of `_check_inputs` (both AIS and GrIS), coerce year to ndarray:
  ```python
  # AIS line 330 (before existing `if self.year[0] == 2015:`)
  self.year = np.asarray(self.year)
  # GrIS line 896 — same
  ```
  This is less invasive than reordering `__post_init__` and matches what `_convert_arrays` does later.

#### P0.6 `melt_in_floating_cells='False'` breaks `_map_args`

- **Where:** [ise/data/inputs.py:369-378](ise/data/inputs.py#L369-L378) (validator) and [ise/data/inputs.py:455-460](ise/data/inputs.py#L455-L460) (mapping).
- **What:** Validator accepts `('floating condition', 'sub-grid', 'No', 'None', 'False')` but mapping has only `{'floating condition': 'Floating_condition', 'sub-grid': 'Sub-grid', 'No': 'No', 'None': None}`. `'False'` validates then `KeyError`s in `_map_args`.
- **Decision:** **Remove `'False'` from the validator**, do not add it to the mapping. Confirmed via `AIS_model_characteristics.csv`: the training data only uses `Sub-grid`, `Floating_condition`, `No`, `None`. `'False'` was never a valid model encoding.
- **Reproduction:** same as `_check_inputs` example but with `melt_in_floating_cells='False'`.
- **Fix:** at line 374-378 change the validator tuple from:
  ```python
  if str(self.melt_in_floating_cells) not in (
      "floating condition",
      "sub-grid",
      "None",
      "False",
      "No",
  ):
  ```
  to:
  ```python
  if str(self.melt_in_floating_cells) not in (
      "floating condition",
      "sub-grid",
      "None",
      "No",
  ):
  ```
  And update the error message on line 376-378 to drop `'False'`.

#### P1.3 Wrong field name in GrIS `surface_thickness` error message

- **Where:** [ise/data/inputs.py:975-976](ise/data/inputs.py#L975-L976).
- **What:** Validation block for `surface_thickness` (line 975) raises with the message `"bed must be one of 'morlighem' or 'bamber'"` — but `bed` is a different field validated on line 972-973. This is a copy-paste error.
- **Fix:** change line 976 from `"bed must be one of 'morlighem' or 'bamber'"` to `"surface_thickness must be one of 'None' or 'morlighem'"`.

#### P2.9 `__post_init__` not idempotent

- **Where:** [ise/data/inputs.py:496-503](ise/data/inputs.py#L496-L503) (AIS) and [ise/data/inputs.py:1099-1110](ise/data/inputs.py#L1099-L1110) (GrIS).
- **What:** `_map_args` does `arg_map[key][lookup_key]`. After mapping, the value is e.g. `'FD'` which is no longer a key in the inner dict. Calling `__post_init__` twice raises `KeyError`. Edge case but easy to fix while we're here.
- **Fix:** in the AIS `_map_args` loop, before doing the lookup, skip already-mapped values. Replace lines 496-503 with:
  ```python
  for key, value in vars(self).items():
      current_value = getattr(self, key)

      if key in arg_map:
          # Skip if already mapped (idempotent post_init)
          if current_value in arg_map[key].values():
              continue
          # Normalise Python None to the string 'None' so the lookup succeeds
          lookup_key = "None" if current_value is None else current_value
          new_value = arg_map[key][lookup_key]
          setattr(self, key, new_value)
  ```
  Apply the same pattern to GrIS `_map_args`.

#### Tests for the inputs.py batch

In [tests/ise/data/test_inputs.py](tests/ise/data/test_inputs.py), add four tests:
1. `test_ais_inputs_accepts_python_list_year` — passes `year=list(range(2015, 2101))` and asserts no exception.
2. `test_ais_inputs_rejects_false_melt_value` — passes `melt_in_floating_cells='False'` and asserts a `ValueError` is raised at validation (not a `KeyError` later).
3. `test_gris_surface_thickness_error_message` — passes an invalid `surface_thickness` value and asserts the error message contains `"surface_thickness"` (not `"bed"`).
4. `test_ais_post_init_is_idempotent` — constructs a valid input, calls `inputs.__post_init__()` a second time, asserts no exception.

#### Commit message for the inputs.py batch

```
fix four inputs.py validation issues

- AIS/GrIS _check_inputs now coerces year to ndarray before arithmetic,
  so passing year=list(...) no longer raises TypeError.
- 'False' removed from melt_in_floating_cells accepted values; it was
  never a valid encoding (see AIS_model_characteristics.csv) and only
  caused a delayed KeyError in _map_args.
- GrIS surface_thickness validator error message corrected (was
  copy-pasted from the bed validator).
- _map_args is now idempotent: already-mapped values are skipped, so
  calling __post_init__ twice no longer KeyErrors.
```

---

### P0.7 GrIS v1.0.0 deprecated classes missing `DeprecationWarning`

- **Where:** [ise/models/iseflow.py:1009-1012](ise/models/iseflow.py#L1009-L1012) (`ISEFlow_GrIS_DE_v1_0_0.__init__`) and [ise/models/iseflow.py:1077-1083](ise/models/iseflow.py#L1077-L1083) (`ISEFlow_GrIS_NF_v1_0_0.__init__`).
- **What:** AIS counterparts emit `warnings.warn(..., DeprecationWarning)` (lines 965-968 and 1050-1053). GrIS classes do not — silent.
- **Reproduction:**
  ```python
  import warnings
  from ise.models.iseflow import ISEFlow_GrIS_DE_v1_0_0
  with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      ISEFlow_GrIS_DE_v1_0_0()
      assert any(issubclass(x.category, DeprecationWarning) for x in w), w
  # AssertionError
  ```
- **Fix:** at the start of `ISEFlow_GrIS_DE_v1_0_0.__init__` (line 1011, before `self.input_size = 90`):
  ```python
  warnings.warn(
      "ISEFlow_GrIS_DE_v1_0_0 is deprecated and will be removed in future versions. Please use ISEFlow_GrIS instead.",
      DeprecationWarning,
  )
  ```
  Add the same block at the start of `ISEFlow_GrIS_NF_v1_0_0.__init__` (line 1080, before `self.input_size = 90`):
  ```python
  warnings.warn(
      "ISEFlow_GrIS_NF_v1_0_0 is deprecated and will be removed in future versions. Please use ISEFlow_GrIS instead.",
      DeprecationWarning,
  )
  ```
- **Test to add:** in [tests/ise/models/test_iseflow.py](tests/ise/models/test_iseflow.py) (or wherever the AIS deprecation tests live — grep for `ISEFlow_AIS_DE_v1_0_0`), add `test_iseflow_gris_de_v1_0_0_emits_deprecation_warning` and `test_iseflow_gris_nf_v1_0_0_emits_deprecation_warning` using `pytest.warns(DeprecationWarning)`.
- **Commit message:**
  ```
  emit DeprecationWarning from GrIS v1.0.0 legacy classes

  ISEFlow_GrIS_DE_v1_0_0 and ISEFlow_GrIS_NF_v1_0_0 were silent on
  instantiation; their AIS twins already warned. Without the warning the
  deprecation policy is unenforced.
  ```

---

### P0.8 `sum_by_sector` UnboundLocalError on `xr.Dataset` input

- **Where:** [ise/evaluation/metrics.py:80-89](ise/evaluation/metrics.py#L80-L89).
- **What:** Inside the `elif isinstance(grid_file, xr.Dataset):` branch, code reads `grids.Description` — but `grids` is only assigned in the previous `if isinstance(grid_file, str):` branch.
- **Reproduction:**
  ```python
  import xarray as xr
  import numpy as np
  from ise.evaluation.metrics import sum_by_sector
  ds = xr.Dataset({'sectors': (('x', 'y'), np.zeros((100, 100)))},
                  attrs={'Description': 'AIS'})
  sum_by_sector(np.zeros((86, 100, 100)), ds)
  # UnboundLocalError: cannot access local variable 'grids'
  ```
- **Fix:** in the `elif` branch starting at line 86, assign `grids = grid_file` before reading attrs. Replace lines 83-89 with:
  ```python
  if isinstance(grid_file, str):
      grids = xr.open_dataset(grid_file)
      sector_name = "sectors" if "ais" in grid_file.lower() else "ID"
  elif isinstance(grid_file, xr.Dataset):
      grids = grid_file
      sector_name = "ID" if "Rignot" in grids.attrs.get("Description", "") else "sectors"
  else:
      raise ValueError("grid_file must be a string or an xarray Dataset.")
  ```
  Note: also changed `grids.Description` to `grids.attrs.get("Description", "")` because `Description` is in `attrs`, not a top-level attribute on the Dataset object.
- **Test to add:** in [tests/ise/evaluation/test_metrics.py](tests/ise/evaluation/test_metrics.py), add `test_sum_by_sector_accepts_xr_dataset` that constructs a small `xr.Dataset` and calls `sum_by_sector(arr, ds)` without crashing.
- **Commit message:**
  ```
  fix sum_by_sector UnboundLocalError on xr.Dataset input

  The xr.Dataset branch read `grids.Description` but `grids` was only
  assigned in the str branch above. Now binds grids = grid_file and reads
  Description via attrs.get for safety.
  ```

---

## P1 — High: should fix before publishing

### P1.1 + P3.1 — `unscale_column` and `unscale_input` in `utils/functions.py`

Single commit. Two related fixes in the same file.

#### P1.1 `unscale_column` wrong year range and AIS-only sector range

- **Where:** [ise/utils/functions.py:461-492](ise/utils/functions.py#L461-L492).
- **What:** Line 488 fits `MinMaxScaler` over `np.arange(2016, 2101)` (85 years) — should be `2015-2100` (86 years). Line 481 fits sectors over `np.arange(1, 19)` — only correct for AIS.
- **Fix:** add `ice_sheet="AIS"` parameter, default to AIS to preserve existing call sites. Replace lines 461-492 with:
  ```python
  def unscale_column(
      dataset: pd.DataFrame,
      column: str | list[str] = "year",
      ice_sheet: str = "AIS",
  ):
      """
      Unscales specified columns back to their original range using known value distributions.

      This function is specifically used to revert the normalization of 'year' and
      'sectors' columns since they have known value ranges.

      Args:
          dataset (pd.DataFrame): Dataset containing the scaled columns.
          column (str or list, optional): Column(s) to be unscaled.
              Can be 'year', 'sectors', or a list containing both. Defaults to "year".
          ice_sheet (str, optional): 'AIS' (18 sectors) or 'GrIS' (6 basins).
              Only relevant when 'sectors' is in `column`. Defaults to "AIS".

      Returns:
          pd.DataFrame: Dataset with the specified column(s) unscaled.
      """

      if isinstance(column, str):
          column = [column]

      if "sectors" in column:
          n_sectors = 18 if ice_sheet.upper() == "AIS" else 6
          sectors_scaler = MinMaxScaler().fit(np.arange(1, n_sectors + 1).reshape(-1, 1))
          dataset["sectors"] = sectors_scaler.inverse_transform(
              np.array(dataset.sectors).reshape(-1, 1)
          )
          dataset["sectors"] = round(dataset.sectors).astype(int)

      if "year" in column:
          year_scaler = MinMaxScaler().fit(np.arange(2015, 2101).reshape(-1, 1))
          dataset["year"] = year_scaler.inverse_transform(np.array(dataset.year).reshape(-1, 1))
          dataset["year"] = round(dataset.year).astype(int)

      return dataset
  ```

#### P3.1 `unscale_input` ndarray support

- **Where:** [ise/utils/functions.py:859-886](ise/utils/functions.py#L859-L886).
- **What:** Docstring claims `np.ndarray or pd.DataFrame`, but ndarray raises `NotImplementedError`.
- **Fix:** drop ndarray from the docstring (don't expand support — too risky right before release). Change the docstring "X (np.ndarray or pd.DataFrame)" to "X (pd.DataFrame)" at line 864 and the return-type accordingly.

#### Tests

In [tests/ise/utils/test_functions.py](tests/ise/utils/test_functions.py), add:
1. `test_unscale_column_year_round_trip` — scales `np.arange(2015, 2101)` with MinMaxScaler, runs through `unscale_column`, asserts the recovered values equal the originals.
2. `test_unscale_column_gris_sectors` — same for `np.arange(1, 7)` with `ice_sheet="GrIS"`.

#### Commit message

```
fix unscale_column year/sector ranges and clarify unscale_input docstring

unscale_column was fitting MinMaxScaler over np.arange(2016, 2101), one
year short of the package contract; fixed to 2015-2100. Sectors were
hard-coded to 1-18 (AIS); now takes an ice_sheet kwarg defaulting to AIS
so GrIS users get the right 1-6 range.

unscale_input docstring claimed ndarray support that was never
implemented; corrected to pd.DataFrame only.
```

---

### P1.2 + P0.2 + P0.3 + P2.3 + P2.7 — `feature_engineer.py` batch

Single commit. Five related fixes in [ise/data/feature_engineer.py](ise/data/feature_engineer.py).

#### P0.2 `__init__` overwrites splits with `None`

- **Where:** [ise/data/feature_engineer.py:132-143](ise/data/feature_engineer.py#L132-L143).
- **What:** Lines 135-138 set splits, lines 141-143 immediately reset them to `None`.
- **Fix:** delete lines 141-143 and move the initialisation *above* the `if split_dataset:` block. Replace lines 132-143 with:
  ```python
  self.train = None
  self.val = None
  self.test = None

  if fill_mrro_nans:
      self.data = self.fill_mrro_nans(method="zero")

  if split_dataset:
      self.train, self.val, self.test = self.split_data(
          data, train_size, val_size, test_size, output_directory, random_state=42
      )
  self._including_model_characteristics = False
  ```

#### P0.3 `split_training_data` ignores `random_state`

- **Where:** [ise/data/feature_engineer.py:781-822](ise/data/feature_engineer.py#L781-L822), specifically line 817.
- **Fix:** replace line 817 (`np.random.shuffle(total_ids)`) with:
  ```python
  rng = np.random.default_rng(random_state)
  rng.shuffle(total_ids)
  ```

#### P1.2 `scale_data(X=...)` references undefined `dropped_data`

- **Where:** [ise/data/feature_engineer.py:208-348](ise/data/feature_engineer.py#L208-L348).
- **Fix:** initialise `dropped_data = pd.DataFrame(index=self.data.index)` at the top of the function (just after the docstring, around line 222 before the `if X is not None:` block). When the user passes X directly, the empty DataFrame concat is a no-op.

#### P2.3 `add_lag_variables` silent partial-segment drop

- **Where:** [ise/data/feature_engineer.py:683-684](ise/data/feature_engineer.py#L683-L684).
- **Fix:** after computing `num_segments`, add:
  ```python
  if len(data) % projection_length != 0:
      warnings.warn(
          f"Data length {len(data)} is not divisible by projection_length "
          f"{projection_length}; dropping {len(data) % projection_length} trailing rows."
      )
  ```

#### P2.7 Misleading comment in standalone `scale_data`

- **Where:** [ise/data/feature_engineer.py:545-547](ise/data/feature_engineer.py#L545-L547).
- **Fix:** replace the comment `# Convert bools back to int` with `# Drop duplicate columns from the concat`.

#### Tests

In [tests/ise/data/test_feature_engineer.py](tests/ise/data/test_feature_engineer.py), add:
1. `test_feature_engineer_split_dataset_returns_non_none_splits` — constructs a FeatureEngineer with `split_dataset=True` and asserts `fe.train is not None and len(fe.train) > 0`.
2. `test_split_training_data_random_state_is_reproducible` — calls `split_training_data(df, 0.7, 0.15, 0.15, random_state=42)` twice (with different global numpy seeds in between) and asserts both return the same train ids.
3. `test_scale_data_with_explicit_X_y_does_not_raise` — calls `fe.scale_data(X=X_df, y=y_df)` and asserts no NameError.

#### Commit message

```
fix five feature_engineer.py issues

- __init__ no longer wipes split_dataset=True results to None three
  lines after creating them.
- split_training_data now uses np.random.default_rng(random_state),
  honouring the documented reproducibility guarantee.
- scale_data(X=..., y=...) initialises dropped_data so the explicit-X
  branch no longer NameErrors at the post-scale concat.
- add_lag_variables warns when the input length is not divisible by
  projection_length (was silently dropping trailing rows).
- Corrected a misleading comment in the standalone scale_data helper.
```

---

### P1.4 + P1.5 — `loss.py` tensor handling

Single commit. Two related fixes in [ise/models/loss.py](ise/models/loss.py).

#### P1.4 `WeightedGridLoss.forward` and `WeightedMSEPCALoss.forward` detach gradients

- **Where:** [ise/models/loss.py:130-131](ise/models/loss.py#L130-L131) (WeightedGridLoss) and [ise/models/loss.py:252-253](ise/models/loss.py#L252-L253) (WeightedMSEPCALoss).
- **What:** `torch.tensor(true, dtype=...)` on a tensor input detaches it from the autograd graph. Loss has no grad. Silent breakage.
- **Fix for WeightedGridLoss** (lines 130-131): replace
  ```python
  true = torch.tensor(true, dtype=torch.float32, device=self.device)
  predicted = torch.tensor(predicted, dtype=torch.float32, device=self.device)
  ```
  with:
  ```python
  true = true.to(self.device).float() if isinstance(true, torch.Tensor) else torch.as_tensor(true, dtype=torch.float32, device=self.device)
  predicted = predicted.to(self.device).float() if isinstance(predicted, torch.Tensor) else torch.as_tensor(predicted, dtype=torch.float32, device=self.device)
  ```
- **Fix for WeightedMSEPCALoss** (lines 252-253): same pattern. The existing `input.to(self.device)` is fine — but verify the function does not also wrap input in `torch.tensor(...)` elsewhere. (It doesn't — only `data_mean`/`data_std`/`weight_factor`/`custom_weights` are wrapped at construction time, which is correct.)

#### P1.5 `WeightedMSEPCALoss` mutates `self.custom_weights` on first forward

- **Where:** [ise/models/loss.py:268-275](ise/models/loss.py#L268-L275).
- **Fix:** replace lines 268-275 with:
  ```python
  if self.custom_weights is not None:
      cw = self.custom_weights
      if cw.dim() == 1:
          cw = cw.unsqueeze(0)
      if cw.shape != weights.shape:
          raise ValueError("Custom weights shape must match input/target shape.")
      weights = weights * cw
  ```
  Use a local `cw`; never mutate `self.custom_weights`.

#### Tests

In [tests/ise/models/test_loss.py](tests/ise/models/test_loss.py), add:
1. `test_weighted_grid_loss_preserves_gradients` — passes a `predicted` tensor with `requires_grad=True`, computes the loss, calls `.backward()`, asserts `predicted.grad is not None`.
2. `test_weighted_mse_pca_loss_does_not_mutate_custom_weights` — instantiates with a 1-D `custom_weights`, captures the tensor object, calls `forward()` twice with different batch shapes, asserts `loss.custom_weights` is unchanged across calls.

#### Commit message

```
fix loss tensor handling: gradient detachment and weight mutation

WeightedGridLoss.forward and WeightedMSEPCALoss.forward wrapped their
inputs in torch.tensor(...), which detaches autograd-tracked tensors.
Replaced with .to(device) for tensor inputs and torch.as_tensor for
others.

WeightedMSEPCALoss.forward also mutated self.custom_weights on first
call (in-place unsqueeze). Now uses a local copy so the registered
tensor is never modified.
```

---

### P1.6 + P1.7 + P1.8 + P3.2 — `lstm.py` batch

Single commit. Four related fixes in [ise/models/lstm.py](ise/models/lstm.py).

#### P1.6 `LSTM.save()` crashes when `sequence_length is None`

- **Where:** [ise/models/lstm.py:481](ise/models/lstm.py#L481).
- **Fix:** replace `"sequence_length": int(self.sequence_length),` with:
  ```python
  "sequence_length": int(self.sequence_length) if self.sequence_length is not None else 5,
  ```

#### P1.7 `LSTM.forward` unnecessary `requires_grad_()` on h0/c0

- **Where:** [ise/models/lstm.py:179-191](ise/models/lstm.py#L179-L191).
- **Fix:** replace lines 179-191 with the simpler:
  ```python
  _, (hn, _) = self.lstm(x)
  x = hn[-1, :, :]
  ```
  `nn.LSTM` defaults to zero h0/c0 internally — no need to construct them by hand.

#### P1.8 LSTM dropout module built but not applied

- **Where:** [ise/models/lstm.py:159](ise/models/lstm.py#L159) and forward at [ise/models/lstm.py:194-196](ise/models/lstm.py#L194-L196).
- **Decision:** apply the dropout (option a from the audit) — matches user expectation from the docstring.
- **Fix:** in `forward`, between `linear1`/`relu` and `linear_out`, apply dropout:
  ```python
  x = self.linear1(x)
  x = self.relu(x)
  if self.dropout is not None:
      x = self.dropout(x)
  x = self.linear_out(x)
  ```

#### P3.2 `LSTM.predict` toggles back to train mode

- **Where:** [ise/models/lstm.py:442](ise/models/lstm.py#L442).
- **Fix:** delete line 442 (`self.train()`). After `predict()`, the model stays in eval mode — that's the user expectation.

#### Tests

In [tests/ise/models/test_lstm.py](tests/ise/models/test_lstm.py), add:
1. `test_lstm_save_with_unset_sequence_length_does_not_crash` — instantiates LSTM, never calls fit, calls save, asserts no exception and that the metadata file exists.
2. `test_lstm_dropout_is_applied_in_forward` — instantiates LSTM with `dropout=0.5`, runs the same input twice in train mode, asserts outputs differ (proves dropout is active). Confirm the test gets the expected behaviour on a small fixed seed.
3. `test_lstm_predict_leaves_model_in_eval_mode` — calls predict, asserts `not model.training`.

#### Commit message

```
fix four LSTM correctness/cleanliness issues

- save() now defaults sequence_length to 5 when unset, instead of
  TypeErroring on int(None).
- forward() drops the manual zero h0/c0 construction (nn.LSTM does
  this internally) and the unnecessary requires_grad_() on them.
- The dropout module that was being constructed but never applied is
  now wired in between linear1/relu and linear_out, matching the
  documented behaviour.
- predict() no longer toggles the model back to train mode after
  inference; it stays in eval mode.
```

---

### P1.9 `ISEFlow.fit` reseeds with `np.random.randint`

- **Where:** [ise/models/iseflow.py:186](ise/models/iseflow.py#L186).
- **What:** `torch.manual_seed(np.random.randint(0, 100000))` clobbers any seed the caller set, with a random seed itself. Defeats reproducibility.
- **Fix:** delete line 186 entirely. Callers should set their own seeds upstream.
- **Test to add:** in [tests/ise/models/test_iseflow.py](tests/ise/models/test_iseflow.py), add `test_fit_does_not_reseed_torch_rng` — sets `torch.manual_seed(123)`, captures `torch.randint(0, 100, (1,))`, calls `model.fit(...)` with `nf_epochs=1, de_epochs=1`, sets `torch.manual_seed(123)` again, captures the same random call, asserts they're equal. (This proves `fit` did not consume any rng state to override the user's seed — but note: the *training itself* consumes rng. So this test should just check no obvious reseed happened, e.g. by setting the seed after fit and checking determinism on a fresh call. **Simplest reliable form:** assert that `iseflow.py` source does not contain `torch.manual_seed(np.random.randint`.)
- **Commit message:**
  ```
  remove rng-clobbering call in ISEFlow.fit

  fit() called torch.manual_seed(np.random.randint(0, 100000)) on entry,
  which silently overrode any seed the caller had set with a random one.
  Reproducibility is the caller's responsibility; this line was actively
  harmful.
  ```

---

### P1.10 — **STOP AND ASK**

> The audit flagged P1.10 (silent scaler-X copy failure in `ISEFlow.save`) but the suggested fix in the original audit involved an API change (making `output_scaler_path` a tuple/dict). That's a breaking change and should not be made unilaterally. Stop here and present the user with the tradeoff:
>
> - **Option A (non-breaking):** add a `warnings.warn(...)` when the inferred scaler-X path doesn't exist, leave the API alone.
> - **Option B (breaking):** accept a `(scaler_X_path, scaler_y_path)` tuple, deprecate the string form.
>
> Wait for user direction before editing.

---

## P2 — Medium: cleanup pass

Implement these as a **single polish commit** if all are quick. If any item is non-trivial, split it out.

### P2.1 Document scaler asymmetry assumption in `ISEFlow.predict`

- **Where:** [ise/models/iseflow.py:341-379](ise/models/iseflow.py#L341-L379).
- **Fix:** add a comment at line 341 explaining that the upper/lower-then-average pattern is correct only for monotone scalers (StandardScaler, MinMaxScaler, RobustScaler — all linear). One sentence.

### P2.2 Quiet `EmulatorDataset` warning on small fixtures

- **Where:** [ise/data/dataclasses.py:89-92](ise/data/dataclasses.py#L89-L92).
- **Fix:** make the warning conditional on `projection_length == 86` (the default — production case). Custom `projection_length` passed by the user means they know what they're doing.
  ```python
  if X.shape[0] < projection_length and projection_length == 86:
      warnings.warn(...)
  ```

### P2.4 Remove `y_test` from `test()` docstrings

- **Where:** [ise/models/iseflow.py:736-756](ise/models/iseflow.py#L736-L756) and [ise/models/iseflow.py:922-942](ise/models/iseflow.py#L922-L942).
- **Fix:** delete the `y_test (array-like): Test target values.` line from both docstrings.

### P2.5 `DeepEnsemble.fit(early_stopping=False)` default

- **Where:** [ise/models/deep_ensemble.py:213](ise/models/deep_ensemble.py#L213).
- **Fix:** flip default to `early_stopping=True`. Update the docstring at line 233 to match.
- **Note:** confirm no test assumes the old default. Run the affected test file after.

### P2.6 Verify v1.0.0 path handles `None`-valued config fields

- **What:** see audit description at P2.6. **No code change required if test passes.**
- **Test to add:** in [tests/ise/models/test_iseflow_pretrained.py](tests/ise/models/test_iseflow_pretrained.py), add `test_iseflow_ais_v1_0_0_predict_with_none_config_field` that calls `ISEFlow_AIS(version="v1.0.0").predict(inputs)` where `inputs` has `standard_melt_type=None` (or whichever field is legitimately optional for an open-melt config). Verify no exception.

### P2.8 Delete commented-out code blocks

- **Where:** see audit P2.8 for the list. Touch each file once, delete the dead code. Do not delete commented-out code that *documents intent* (e.g., a `# TODO:` or `# Reason: ...` block). Only delete commented-out code lines.

### P2.10 Narrow bare `except: pass` blocks in `process.py`

- **Where:** [ise/data/process.py:738](ise/data/process.py#L738), [ise/data/process.py:1035](ise/data/process.py#L1035), [ise/data/process.py:1046](ise/data/process.py#L1046).
- **Fix:** replace `except:` with `except (ValueError, KeyError, AttributeError):` based on what the surrounding code actually expects (read each block — they're all in CSV/JSON parse paths, so `(ValueError, KeyError)` is usually correct).

### Commit message for the polish commit

```
polish pass: docstring fixes, dead code removal, narrower excepts

Cleanup pass for the upcoming release:
- ISEFlow.predict: documented the monotone-scaler assumption.
- EmulatorDataset: small-fixture warning now only fires for the
  default 86-step projection length, not user-set lengths.
- ISEFlow_AIS.test/ISEFlow_GrIS.test: removed the unused y_test arg
  from docstrings.
- DeepEnsemble.fit: default early_stopping flipped to True so direct
  callers match the ISEFlow.fit default.
- Removed commented-out code blocks in lstm.py, feature_engineer.py,
  normalizing_flow.py, loss.py, iseflow.py.
- Narrowed bare except: pass blocks in process.py to specific
  exception types.
- Added regression test for v1.0.0 predict with a None config field.
```

---

### P2.11 Harden HF allow_patterns

- **Where:** [ise/models/pretrained/__init__.py:53](ise/models/pretrained/__init__.py#L53).
- **What:** `allow_patterns=[f"{subfolder}/*"]` works today (verified via fnmatch) but is fragile if the matcher behaviour ever changes. `**` is unambiguous.
- **Fix:** change to `allow_patterns=[f"{subfolder}/**"]`.
- **Test to add:** in [tests/ise/models/test_pretrained_loader.py](tests/ise/models/test_pretrained_loader.py), add an integration test that mocks `snapshot_download` to capture the kwargs and asserts the pattern includes a recursive glob (`**`). Don't actually hit the network.
- **Commit message:**
  ```
  use recursive glob in HuggingFace allow_patterns

  Switched allow_patterns from "<subfolder>/*" to "<subfolder>/**" so
  ensemble_members/*.pth are unambiguously included regardless of the
  matcher's behaviour around the slash character.
  ```

---

## P3 — Future polish (skip if user wants minimal changes)

### P3.3 Mark `ScenarioDataset` deprecated

- Add a `DeprecationWarning` to `ScenarioDataset.__init__` recommending `PyTorchDataset`. One-line change.

### P3.4 `sum_by_sector` magic number

- Add explicit `ice_sheet="AIS"|"GrIS"|None` arg with autodetect fallback. Document the autodetect heuristic. Optional polish.

### P3.6 Move v1.0.0 `mrro_means` array to a CSV

- Move the inline 86-element array from [ise/models/iseflow.py:533-622](ise/models/iseflow.py#L533-L622) to `ise/data/data_files/mrro_year_means_v1_0_0.csv`. Load lazily, cache module-level.

**Skip P3 items unless the user asks for them. They're polish, not bugs.**

---

## P2.12 — Version bump (last commit)

> Only do this after **all** P0/P1 commits land, the full test suite passes, the verification gate below passes, and the user has approved (or skipped) the P1.10 ASK.

- **Where:** [pyproject.toml:8](pyproject.toml#L8) and [CHANGELOG.md:14](CHANGELOG.md#L14).
- **Decision:** bump to `1.1.0` (minor). Justification: restored v1.0.0 weight compatibility plus several behavioural bug fixes warrant minor, not patch. **Stop and ask if user disagrees** before tagging.
- **Fix:**
  1. Edit [pyproject.toml](pyproject.toml) line 8: `version = "1.1.0"`.
  2. Edit [CHANGELOG.md](CHANGELOG.md): replace the `## [Unreleased]` heading on line 14 with `## [1.1.0] — 2026-05-07 (package) | Model: v1.1.0`. Add a fresh empty `## [Unreleased]` heading above it for future work.
- **Do NOT push, tag, or amend.** This is the final commit on the branch; the user will handle merge + tag.
- **Commit message:**
  ```
  bump package version to 1.1.0 and finalise CHANGELOG

  Promotes the [Unreleased] section to [1.1.0]. Restored v1.0.0 weights
  compatibility plus a substantial set of bug fixes (see CHANGELOG)
  warrants a minor bump rather than a patch.
  ```

---

## Verification gate (must pass before declaring done)

After all commits land, run these in order. **Do not declare done until all 10 pass.** If any fails, surface the failure and stop — do not paper over.

1. `git status` — clean, no unstaged changes.
2. `python -m pytest tests -m "not slow and not gpu" --tb=short -q` — all green; new regression tests added in P0/P1 commits should also pass.
3. `python examples/example_ais.py` — script completes; `example_ais_projection.png` is created. Eyeball: SLE rises through 2100 with non-zero uncertainty bands.
4. `python examples/example_gris.py` — same, GrIS.
5. `python examples/example_absolute_forcings.py` — completes without exception.
6. `python -c "from ise.models import ISEFlow_AIS; m = ISEFlow_AIS(version='v1.0.0'); print('v1.0.0 OK')"` — backward compat smoke.
7. `python -c "from ise.models import ISEFlow_AIS; m = ISEFlow_AIS(version='v1.1.0'); print('v1.1.0 OK')"` — current-version smoke.
8. `ruff check .` — no new errors introduced (existing baseline noise is OK if it was already there before your work).
9. `python -m build && twine check dist/*` — sdist/wheel build cleanly. (Skip if `build` is not installed; note in your report.)
10. `git log --oneline feature/iseflow_v1-0-0_compat..HEAD` — review the final commit graph. Each commit should be atomic and have the prescribed message format. No fixup or amend commits.

If any step fails, **stop and report** — do not attempt a "quick fix" without checking back with the user.

---

## Critical files modified (summary)

- [ise/data/dataclasses.py](ise/data/dataclasses.py) — P0.1, P2.2
- [ise/data/feature_engineer.py](ise/data/feature_engineer.py) — P0.2, P0.3, P1.2, P2.3, P2.7
- [ise/data/inputs.py](ise/data/inputs.py) — P0.5, P0.6, P1.3, P2.9
- [ise/models/normalizing_flow.py](ise/models/normalizing_flow.py) — P0.4
- [ise/models/iseflow.py](ise/models/iseflow.py) — P0.7, P1.9, P1.10 (ASK), P2.1, P2.4, P3.6
- [ise/models/loss.py](ise/models/loss.py) — P1.4, P1.5
- [ise/models/lstm.py](ise/models/lstm.py) — P1.6, P1.7, P1.8, P3.2
- [ise/models/deep_ensemble.py](ise/models/deep_ensemble.py) — P2.5
- [ise/models/pretrained/__init__.py](ise/models/pretrained/__init__.py) — P2.11
- [ise/utils/functions.py](ise/utils/functions.py) — P1.1, P3.1
- [ise/evaluation/metrics.py](ise/evaluation/metrics.py) — P0.8, P3.4
- [ise/data/process.py](ise/data/process.py) — P2.10
- [pyproject.toml](pyproject.toml) + [CHANGELOG.md](CHANGELOG.md) — P2.12, plus per-commit CHANGELOG entries
- [tests/](tests/) — regression tests added with each fix

## Quick reference: what NOT to do

- Do not push, tag, force, amend, or rebase.
- Do not skip hooks (`--no-verify`).
- Do not coauthor commits with Claude (project policy).
- Do not batch unrelated fixes into a single commit.
- Do not skip CHANGELOG updates — every commit must add a bullet under `[Unreleased]`.
- Do not "improve" code outside the scope of the listed item, even if you spot a bug.
- Do not proceed past **STOP AND ASK** markers.
- Do not declare done until all 10 verification steps pass.

ISEFlow Model Versions
======================

ISEFlow pretrained weights are versioned independently from the ``ise-py`` package.
Weights are hosted on HuggingFace Hub at ``pvankatwyk/ISEFlow`` and downloaded
automatically on first use.  Local bundled fallback weights are used when HuggingFace
is unavailable (air-gapped HPC, offline development).

Use the ``version`` argument when loading a pretrained model:

.. code-block:: python

   from ise.models import ISEFlow_AIS, ISEFlow_GrIS

   model = ISEFlow_AIS(version="v1.1.0")   # latest
   model = ISEFlow_AIS(version="v1.0.0")   # legacy AIS-only weights

Current recommended version: **v1.1.0**

---

Version history
---------------

v1.1.0 — AIS + GrIS (current)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Released: 2026-05-08
:Ice sheets: AIS (18 sectors), GrIS (6 drainage basins)
:Classes: ``ISEFlow_AIS(version="v1.1.0")``, ``ISEFlow_GrIS(version="v1.1.0")``

**What changed from v1.0.0**

- **``mrro_anomaly`` removed from AIS inputs.** The melt-runoff anomaly variable was
  dropped from the AIS feature set after ablation analysis showed it contributed noise
  rather than signal.  Code using v1.0.0-style inputs that passes ``mrro_anomaly`` must
  remove it before calling ``predict()`` with v1.1.0 weights.
- Scalers saved alongside weights per ice sheet and version so that
  ``predict(..., output_scaler=True)`` works correctly for both AIS and GrIS.
- **NormalizingFlow architecture upgraded.** The context encoder was changed from a
  single ``nn.Linear(input_size, output_size * 2)`` layer to a 2-layer MLP
  (``Linear → ReLU → Linear``), and ``flow_hidden_features`` is now a tunable
  hyperparameter rather than being hard-coded to ``output_size * 2``.
- **``get_latent`` changed from deterministic to stochastic.** v1.0.0 pushed a fixed
  zero vector through the forward transform to produce latents; v1.1.0 draws samples
  from the conditional base distribution instead.  Both approaches are preserved —
  ``NormalizingFlow.load()`` auto-detects the version from saved metadata and sets
  ``legacy_v1_0_0=True`` for old weights so inference reproduces the original latents.

**AIS inputs (v1.1.0)**

.. code-block:: text

   year, sector,
   pr_anomaly, evspsbl_anomaly, smb_anomaly, ts_anomaly,
   ocean_thermal_forcing, ocean_salinity, ocean_temperature,
   ice_shelf_fracture, ocean_sensitivity, ocean_forcing_type, standard_melt_type,
   [ISM model characteristics]

**GrIS inputs (v1.1.0)**

.. code-block:: text

   year, region,
   smb_anomaly, st_anomaly, ocean_thermal_forcing, basin_runoff,
   [ISM model characteristics]

---

v1.0.0 — AIS + GrIS (legacy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Released: 2026-05-07
:Ice sheets: AIS (18 sectors), GrIS (6 drainage basins)
:Classes: ``ISEFlow_AIS(version="v1.0.0")``, ``ISEFlow_GrIS(version="v1.0.0")``

The original pretrained release.  Maintained for reproducibility of results from the
ISEFlow manuscript.

**AIS inputs (v1.0.0)**

Identical to v1.1.0 **plus** ``mrro_anomaly`` (melt-runoff anomaly):

.. code-block:: text

   year, sector,
   pr_anomaly, evspsbl_anomaly, smb_anomaly, ts_anomaly, mrro_anomaly,
   ocean_thermal_forcing, ocean_salinity, ocean_temperature,
   ice_shelf_fracture, ocean_sensitivity, ocean_forcing_type, standard_melt_type,
   [ISM model characteristics]

.. note::
   ``ISEFlow_AIS(version="v1.0.0")`` and ``ISEFlow_GrIS(version="v1.0.0")`` are still
   loadable as of package v1.1.0.  The ``NormalizingFlow`` loader auto-detects the older
   architecture from saved metadata.  ``ISEFlow_GrIS_DE/NF_v1_0_0`` emit a
   ``DeprecationWarning`` on instantiation.

---

Migrating between versions
---------------------------

v1.0.0 → v1.1.0 (AIS)
~~~~~~~~~~~~~~~~~~~~~~~

Remove ``mrro_anomaly`` from your ``ISEFlowAISInputs`` constructor call and switch the
``version`` argument:

.. code-block:: python

   # v1.0.0 style — no longer valid for v1.1.0
   inputs = ISEFlowAISInputs(
       ...,
       mrro_anomaly=mrro,   # remove this line
   )
   model = ISEFlow_AIS(version="v1.0.0")

   # v1.1.0 style
   inputs = ISEFlowAISInputs(
       year=year,
       sector=sector,
       pr_anomaly=pr,
       evspsbl_anomaly=evspsbl,
       smb_anomaly=smb,
       ts_anomaly=ts,
       ocean_thermal_forcing=otf,
       ocean_salinity=sal,
       ocean_temperature=temp,
       model_configs="AWI_PISM1",
       ice_shelf_fracture=False,
       ocean_sensitivity="medium",
       ocean_forcing_type="standard",
       standard_melt_type="local",
   )
   model = ISEFlow_AIS(version="v1.1.0")

Using absolute forcings instead of anomalies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have raw absolute climate values rather than pre-computed anomalies, use the
``from_absolute_forcings()`` classmethod.  Pass ``aogcm`` for a bundled CMIP model or
``custom_climatology`` for a new model not in the bundled set:

.. code-block:: python

   inputs = ISEFlowAISInputs.from_absolute_forcings(
       year=year,
       sector=sector,
       pr=pr_abs,
       evspsbl=evspsbl_abs,
       smb=smb_abs,
       ts=ts_abs,
       ocean_thermal_forcing=otf,
       ocean_salinity=sal,
       ocean_temperature=temp,
       aogcm="MIROC6",          # or custom_climatology={...}
       model_configs="AWI_PISM1",
       ice_shelf_fracture=False,
       ocean_sensitivity="medium",
       ocean_forcing_type="standard",
       standard_melt_type="local",
   )

.. note::
   ``from_raw_values()`` is a deprecated alias for ``from_absolute_forcings()`` and will
   be removed in a future release.

---

Weight resolution
-----------------

The path resolution order is:

1. HuggingFace Hub (``pvankatwyk/ISEFlow``) — downloaded on first use and cached locally.
2. Bundled local weights under ``ise/models/pretrained/ISEFlow/`` — used automatically
   when HuggingFace is unreachable.

Use ``get_model_dir(version, ice_sheet)`` (from ``ise.models.pretrained``) to resolve
the path programmatically.  Do not use the deprecated ``_path`` constants.

.. code-block:: python

   from ise.models.pretrained import get_model_dir

   path = get_model_dir("v1.1.0", "AIS")

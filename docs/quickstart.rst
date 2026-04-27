Quickstart
==========

This page walks through the most common use case: running a pretrained
**ISEFlow-AIS** emulator to produce sea level projections with uncertainty
quantification for a single Antarctic sector.

Installation
------------

.. code-block:: shell

   pip install -e .

or with ``uv``:

.. code-block:: shell

   uv venv && uv pip install -r requirements.txt

Running the pretrained AIS emulator
------------------------------------

.. code-block:: python

   import numpy as np
   from ise.models import ISEFlow_AIS
   from ise.data.inputs import ISEFlowAISInputs

   # 86 annual timesteps covering 2015–2100
   year   = np.arange(2015, 2101)
   sector = np.ones(86, dtype=int)   # AIS sector 1 of 18

   # Build the validated input dataclass.
   # Use model_configs="<ISM_NAME>" to load all ISM settings automatically,
   # or set each parameter individually (see ISEFlowAISInputs docs).
   inputs = ISEFlowAISInputs(
       year=year,
       sector=sector,
       pr_anomaly=np.zeros(86),
       evspsbl_anomaly=np.zeros(86),
       smb_anomaly=np.zeros(86),
       ts_anomaly=np.zeros(86),
       ocean_thermal_forcing=np.zeros(86),
       ocean_salinity=np.zeros(86),
       ocean_temperature=np.zeros(86),
       model_configs="AWI_PISM1",       # loads all ISM config fields automatically
       ice_shelf_fracture=False,
       ocean_sensitivity="medium",
       ocean_forcing_type="standard",
       standard_melt_type="local",
   )

   # Load pretrained v1.1.0 model (weights ship with the package)
   model = ISEFlow_AIS(version="v1.1.0")

   # Run inference — returns unscaled SLE projections (mm) + uncertainty dict
   predictions, uncertainties = model.predict(inputs, smoothing_window=5)

   print(predictions.shape)           # (86, 1)
   print(uncertainties["epistemic"])  # array of shape (86,)
   print(uncertainties["aleatoric"])  # array of shape (86,)
   print(uncertainties["total"])      # epistemic + aleatoric

Running the pretrained GrIS emulator
--------------------------------------

.. code-block:: python

   from ise.models import ISEFlow_GrIS
   from ise.data.inputs import ISEFlowGrISInputs

   inputs = ISEFlowGrISInputs(
       year=np.arange(2015, 2101),
       sector=np.ones(86, dtype=int),
       aST=np.zeros(86),
       aSMB=np.zeros(86),
       ocean_thermal_forcing=np.zeros(86),
       basin_runoff=np.zeros(86),
       model_configs="AWI_ISSM1",
       ice_shelf_fracture=False,
       ocean_sensitivity="medium",
       standard_ocean_forcing=True,
   )

   model = ISEFlow_GrIS(version="v1.1.0")
   predictions, uncertainties = model.predict(inputs)

Training ISEFlow from scratch
------------------------------

.. code-block:: python

   from ise.models.iseflow import ISEFlow
   from ise.models.deep_ensemble import DeepEnsemble
   from ise.models.normalizing_flow import NormalizingFlow

   nf = NormalizingFlow(input_size=93, output_size=1, num_flow_transforms=5)
   de = DeepEnsemble(input_size=93, num_ensemble_members=5, output_sequence_length=86)

   model = ISEFlow(deep_ensemble=de, normalizing_flow=nf)

   # X shape: (N, n_features), y shape: (N,)
   model.fit(
       X_train, y_train,
       nf_epochs=100,
       de_epochs=100,
       X_val=X_val,
       y_val=y_val,
       early_stopping=True,
       patience=15,
   )

   model.save("my_model/", input_features=list(X_train.columns))

See the ``ISEFlow``, ``DeepEnsemble``, and ``NormalizingFlow`` API docs for all
available arguments.

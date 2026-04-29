ise.models
==========

Ice sheet emulator models: ISEFlow, LSTM, DeepEnsemble, NormalizingFlow,
loss functions, training utilities, and pretrained weight paths.

The flagship model is **ISEFlow** (``ise.models.iseflow``), a hybrid emulator
that chains a ``NormalizingFlow`` (aleatoric uncertainty) with a ``DeepEnsemble``
of ``LSTM`` networks (epistemic uncertainty).  Pretrained weights for AIS v1.1.0
and GrIS v1.1.0 are accessed via ``ISEFlow_AIS`` and ``ISEFlow_GrIS``.

Additional experimental models (GP, PCA, ScenarioPredictor,
VariationalLSTMEmulator) live in ``ise.models._experimental`` and are not
part of the primary API.

Submodules
----------

ise.models.iseflow
------------------

.. automodule:: ise.models.iseflow
   :members:
   :undoc-members:
   :show-inheritance:

ise.models.deep\_ensemble
-------------------------

.. automodule:: ise.models.deep_ensemble
   :members:
   :undoc-members:
   :show-inheritance:

ise.models.lstm
---------------

.. automodule:: ise.models.lstm
   :members:
   :undoc-members:
   :show-inheritance:

ise.models.normalizing\_flow
-----------------------------

.. automodule:: ise.models.normalizing_flow
   :members:
   :undoc-members:
   :show-inheritance:

ise.models.loss
---------------

.. automodule:: ise.models.loss
   :members:
   :undoc-members:
   :show-inheritance:

ise.models.training
-------------------

.. automodule:: ise.models.training
   :members:
   :undoc-members:
   :show-inheritance:

ise.models.pretrained
---------------------

.. automodule:: ise.models.pretrained
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: ise.models
   :members:
   :undoc-members:
   :show-inheritance:

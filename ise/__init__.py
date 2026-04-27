"""ISE — Ice Sheet Emulator package.

ISE provides end-to-end tools for training and running ice sheet emulators,
with a focus on ISEFlow: a hybrid normalizing-flow + deep-ensemble model that
produces sea level projections with full uncertainty quantification (epistemic
and aleatoric) for the Antarctic Ice Sheet (AIS) and Greenland Ice Sheet (GrIS).

Quickstart
----------
Install the package::

    pip install -e .

Load the pretrained AIS emulator and make a prediction::

    from ise.models import ISEFlow_AIS
    from ise.data.inputs import ISEFlowAISInputs

    model = ISEFlow_AIS(version="v1.1.0")
    inputs = ISEFlowAISInputs(...)
    predictions, uncertainties = model.predict(inputs)

Package layout
--------------
- ``ise.data``       — forcing/grid loading, feature engineering, dataset classes
- ``ise.models``     — ISEFlow, DeepEnsemble, NormalizingFlow, LSTM, loss functions
- ``ise.evaluation`` — point, probabilistic, and distribution metrics
- ``ise.utils``      — data helpers, tensor utilities

For questions contact Peter Van Katwyk at pvankatwyk@gmail.com.
"""

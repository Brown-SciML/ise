import os

import pandas as pd
from paths import RESULTS_DATASET


# Test Results Data
def test_results_data_exists():
    assert os.path.exists(
        RESULTS_DATASET
    ), "Results dataset doesn't exist. Run the testing procedure to generate using ise.utils.data.combine_testing_results."


results = pd.read_csv(RESULTS_DATASET)


def test_results_nonempty():
    assert not results.empty, "Results dataset is empty."


def test_results_attributes():
    assert all(
        [col in results.columns for col in ["sectors", "aogcm", "modelname", "exp_id", "salinity"]]
    )

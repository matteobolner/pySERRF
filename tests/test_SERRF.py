import pandas as pd
import numpy as np
import pytest
from pyserrf.serrf import SERRF

data = pd.read_table("test_data/SERRF_example_dataset.tsv")

a = SERRF(
    sample_type_column="sampleType",
    batch_column="batch",
    sample_metadata_columns=["sampleType", "batch", "label", "time"],
    random_state=42,
    minus=False,
    n_correlated_metabolites=10,
)

a.fit(data)

a.transform(data, return_data_only=False)
a.normalized_dataset_
a.normalized_dataset_.to_csv("class_results_2.tsv", index=False, sep="\t")

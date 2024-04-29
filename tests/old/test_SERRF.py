import pandas as pd
import numpy as np
import pytest
from pyserrf import SERRF

from sklearn.preprocessing import StandardScaler

data = pd.read_table("test_data/SERRF_example_dataset.tsv")

#a = pd.read_table("class_results.tsv")
#b = pd.read_table("class_results_2.tsv")
#c = pd.read_table("class_results_3.tsv")
normdict={}

for i in [1,2]:
    a = SERRF(
        sample_type_column="sampleType",
        batch_column="batch",
        sample_metadata_columns=["sampleType", "batch", "label", "time"],
        random_state=i,
        minus=False,
        n_correlated_metabolites=10,
        )
    normalized=a.fit_transform(data, return_data_only=True)
    normdict[i]=normalized
    normalized.to_csv(f"test_data/random_seed_test/{i}.tsv", index=False, sep='\t')

a = SERRF(
        sample_type_column="sampleType",
        batch_column="batch",
        sample_metadata_columns=["sampleType", "batch", "label", "time"],
        random_state=1,
        minus=False,
        n_correlated_metabolites=10,
        )
normalized=a.fit_transform(data, return_data_only=True)
normdict['1_bis']=normalized
normalized.to_csv(f"test_data/random_seed_test/1_bis.tsv", index=False, sep='\t')

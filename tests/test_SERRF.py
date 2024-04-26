import pandas as pd
import numpy as np
import pytest
from pyserrf.read_data import read_serff_format_data_simple
from pyserrf.SERRF_class import SERRF

data=pd.read_table("test_data/SERRF_example_dataset.tsv")

a=SERRF(random_state=42, sample_metadata_columns=['batch','sampleType','time','label'])

normalized=pd.concat(normalized, axis=1)
normalized=a._fit(merged=data, minus=False)
normalized.to_csv("class_results.tsv", index=False, sep='\t')

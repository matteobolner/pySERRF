import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pyserrf.read_data import read_serff_format_data_simple
from pyserrf.utils import replace_zero_values, replace_nan_values, center_data, standard_scaler


merged = read_serff_format_data_simple("test_data/SERRF example dataset.xlsx")

sample_metadata_columns=['batch','sampleType','time','label']

training=merged[merged['sampleType']=='qc']
target=merged[merged['sampleType']!='qc']

sample_metadata=merged[sample_metadata_columns]
data=merged.drop(columns=sample_metadata_columns)
data=data.astype(float)

metabolites=list(data.columns)

def check_for_nan_or_zero(data):
    if ((0 not in data.values) & (~data.isna().any().any())):
        print("No missing/zero values")
        return False
    else:
        return True

def handle_zero_and_nan(merged):
    if check_for_nan_or_zero(merged[metabolites]):
        groups=[]
        for name,group in merged.groupby(by='batch'):
            if check_for_nan_or_zero(group[metabolites]):
                for metabolite in metabolites:
                    changed_values=False
                    values=group[metabolite]
                    if 0 in values:
                        values=replace_zero_values(values)
                        changed_values=True
                    if values.isna().any():
                        values=replace_nan_values(values)
                        changed_values=True
                    if changed_values:
                        group[metabolite]=values
                groups.append(group)
            else:
                groups.append(group)
        merged=pd.concat(groups)
        #CHECK ORDER
        return merged
    else:
        return merged

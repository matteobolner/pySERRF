from pyserrf.read_data import read_serff_format_data
import pandas as pd
import numpy as np

data=read_serff_format_data("test_data/SERRF example dataset.xlsx")
e=data['e']
p=data['p']
batches=p['batch']
e_matrix=data['e_matrix']

training = e_matrix[[i for i in e_matrix.columns if "QC" in i] ]
target=e_matrix[[i for i in e_matrix.columns if "QC" not in i]]


all=pd.concat([training, target], axis=1)
normalized=np.zeros(len(all.columns))

batch="A"


def replace_zero_values(row):
    zero_values=row[row==0]
    zero_replacements = np.random.normal(loc=np.min(row.dropna())+1,
                                      scale=0.1 * (np.min(row.dropna()) + 0.1),
                                      size=len(zero_values))
    row.loc[zero_values.index]=zero_replacements
    return(row)


def replace_nan_values(row):
    nan_values=row[row.isna()]
    nan_replacements = np.random.normal(loc=(0.5 * np.min(row.dropna()+1)),
                                      scale=0.1 * (np.min(row.dropna()) + 0.1),
                                      size=len(nan_values))
    row.loc[nan_values.index]=nan_replacements
    return(row)



for index,row in all.iterrows():
    for batch in batches.unique():
        all_current_batch=row[p[p['batch']==batch]['label']]
        ###TEMPORARY!!!!
        #to_zero=all_current_batch.sample(10, random_state=42)
        #all_current_batch.loc[to_zero.index]=0
        ###TEMPORARY!!!!
        row=replace_row_zero_values(row)
        row=replace_nan_values(row)
        all.loc[index]=row

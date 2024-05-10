import pandas as pd

from pyserrf import SERRF

df=pd.read_table("test_data/SERRF_example_dataset.tsv")

a=SERRF(other_columns=['label'], threads=4)
a.fit_transform(df)

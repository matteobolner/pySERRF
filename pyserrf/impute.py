from statsmodels.imputation.mice import *
import pandas as pd
import statsmodels

a=pd.read_table("../pigphenomics_metabolomics/data/raw_data/data_filtered.tsv")
a=a[a.columns[6::]]
imp = MICEData(a)
fml = 'y ~ x1 + x2 + x3 + x4'
mice = MICE(fml, statsmodels.regression.linear_model.OLS, imp)
results = mice.fit(1, 1)
x = mice.next_sample()

x

print(results.summary())

for i in mice.next_sample():
    print(i)

presults = mice.fit(10, 10)
print(results.summary())

data
data=pd.read_table("test_data/e.tsv")

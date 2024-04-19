import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pyserrf.read_data import read_serff_format_data_simple
from pyserrf.utils import (
    replace_zero_values,
    replace_nan_values,
    check_for_nan_or_zero,
    handle_zero_and_nan,
    center_data,
    standard_scaler,
    get_corrs_by_sample_type_and_batch,
    get_top_metabolites_in_both_correlations,
    detect_outliers,
)

##PARAMETERS
n_metabolites = 10
minus = False
##PARAMETERS

merged = read_serff_format_data_simple("test_data/SERRF example dataset.xlsx")

sample_metadata_columns = ["batch", "sampleType", "time", "label"]


sample_metadata = merged[sample_metadata_columns]
data = merged.drop(columns=sample_metadata_columns)
data = data.astype(float)

newcolumns = [f"MET_{i}" for i in range(1, len(data.columns) + 1)]
metabolite_dict = {i: j for i, j in zip(newcolumns, data.columns)}
data.columns = newcolumns
metabolites = list(data.columns)
merged = pd.concat([sample_metadata, data], axis=1)

merged = handle_zero_and_nan(merged, metabolites)


corrs_train, corrs_target = get_corrs_by_sample_type_and_batch(merged, metabolites)


# metabolite='MET_1'

pred = []

for metabolite in metabolites:
    print(metabolite)
    normalized = []
    for batch, group in merged.groupby(by="batch"):
        print(batch)
        training_group = group[group["sampleType"] == "qc"]
        test_group = group[group["sampleType"] != "qc"]
        corr_train = corrs_train[batch]
        corr_target = corrs_target[batch]
        corr_train_order = (
            corr_train.abs()
            .loc[metabolite]
            .sort_values(ascending=False)
            .drop(metabolite)
        )
        corr_target_order = (
            corr_target.abs()
            .loc[metabolite]
            .sort_values(ascending=False)
            .drop(metabolite)
        )
        top_correlated = get_top_metabolites_in_both_correlations(
            corr_train_order, corr_target_order, 10
        )
        factor = np.std(training_group[metabolite], ddof=1) / np.std(
            test_group[metabolite], ddof=1
        )
        if (factor == 0) | (factor == np.nan) | (factor < 1):
            training_data_y = center_data(training_group[metabolite])
        else:
            if len(training_group[metabolite]) * 2 > len(test_group[metabolite]):
                training_data_y = (
                    training_group[metabolite] - training_group[metabolite].mean()
                ) / factor
            else:
                training_data_y = center_data(training_group[metabolite])
        training_data_x = training_group[list(top_correlated)].apply(standard_scaler)
        if training_group[list(top_correlated)].empty:
            test_data_x = standard_scaler(test_group[list(top_correlated)])
        else:
            test_data_x = test_group[list(top_correlated)].apply(standard_scaler)
        training_NA_index = training_data_x.apply(lambda x: x.isna().any())
        training_data_x = training_data_x[training_NA_index[~training_NA_index].index]
        test_data_x = test_data_x[training_NA_index[~training_NA_index].index]

        if training_data_x.empty:
            norm = group[metabolite]
            normalized = pd.concat([normalized, norm])
        else:
            model = RandomForestRegressor(
                n_estimators=500, min_samples_leaf=5, random_state=1
            )
            model.fit(X=training_data_x, y=training_data_y, sample_weight=None)
            training_prediction = model.predict(training_data_x)
            test_prediction = model.predict(test_data_x)
            if minus:
                norm_training = training_group[metabolite] - (
                    (training_prediction + training_group[metabolite].mean())
                    - merged[merged["sampleType"] == "qc"][metabolite].mean()
                )
                norm_test = test_group[metabolite] - (
                    (test_prediction + test_group[metabolite].mean())
                    - merged[merged["sampleType"] != "qc"][metabolite].median()
                )
            else:
                norm_training = training_group[metabolite] / (
                    (training_prediction + training_group[metabolite].mean())
                    / merged[merged["sampleType"] == "qc"][metabolite].mean()
                )
                norm_test = test_group[metabolite] / (
                    (
                        test_prediction
                        + test_group[metabolite].mean()
                        - test_prediction.mean()
                    )
                    / merged[merged["sampleType"] != "qc"][metabolite].median()
                )
            norm_test[norm_test < 0] = test_group[metabolite].loc[
                norm_test[norm_test < 0].index
            ]
            norm_training = norm_training / (
                norm_training.median()
                / merged[merged["sampleType"] == "qc"][metabolite].median()
            )
            norm_test = norm_test / (
                norm_test.median()
                / merged[merged["sampleType"] != "qc"][metabolite].median()
            )
            norm = pd.concat([norm_training, norm_test]).sort_index()
            ##NOT SURE ABOUT LINE BELOW, WORKS AS INTENDED BUT VALUES GENERATED ARE VERY LOW (CENTERED ON ZERO); MIGHT BENEFIT FROM NORM MEAN AS CENTER
            norm[~np.isfinite(norm)] = np.random.normal(
                scale=np.std(norm[np.isfinite(norm)], ddof=1) * 0.01,
                size=len(norm[~np.isfinite(norm)]),
            )
            outliers = detect_outliers(data=norm, threshold=3)
            outliers = outliers[outliers]
            attempt = (
                test_group[metabolite]
                - (
                    (test_prediction + test_group[metabolite].mean())
                    - (merged[merged["sampleType"] != "qc"][metabolite].median())
                )
            ).loc[outliers.index]

            if len(outliers) > 0 & len(attempt) > 0:
                if outliers.mean() > norm.mean():
                    if attempt.mean() < outliers.mean():
                        norm_test.loc[out.index] = attempt
                else:
                    if attempt.mean() > outliers.mean():
                        norm_test.loc[out.index] = attempt

            if len(norm_test[norm_test < 0]) > 0:
                norm_test[norm_test < 0] = test_group.loc[
                    norm_test[norm_test < 0].index
                ][metabolite]
            norm = pd.concat([norm_training, norm_test]).sort_index()
            normalized.append(norm)
    normalized = pd.concat(normalized)
    c = (
        normalized.loc[merged[merged["sampleType"] != "qc"].index].median()
        + (
            merged[merged["sampleType"] == "qc"][metabolite].median()
            - merged[merged["sampleType"] != "qc"][metabolite].median()
        )
        / merged[merged["sampleType"] != "qc"][metabolite].std(ddof=1)
        * normalized.loc[merged[merged["sampleType"] != "qc"].index].std()
    ) / normalized.loc[merged[merged["sampleType"] == "qc"].index].median()
    if c > 0:
        normalized.loc[merged[merged["sampleType"] == "qc"].index] = (
            normalized.loc[merged[merged["sampleType"] == "qc"].index] * c
        )
    pred.append(normalized)

normed = pd.concat(pred, axis=1)
normed=pd.concat([sample_metadata, normed], axis=1)
normed_target=normed[normed['sampleType']!='qc']


for index,row in normed_target.iterrows():
    metadata=row.drop(metabolites)
    row=row[metabolites]

    changed_row=False
    nan_values=row[row.isna()]
    if len(nan_values)>0:
        nan_values_replaced=np.random.normal(
                loc=row.min(),
                scale=np.std(row, ddof=1) * 0.01,
                size=len(nan_values),
            )
        row.loc[nan_values.index]=nan_values_replaced
        changed_row=True
    negative_values=row[row<0]
    if len(negative_values) > 0:
        negative_values_replaced=np.random.uniform(0, 1)*min(row[row>0])
        row.loc[negative_values.index]=row.loc[negative_values.index].replace(negative_values.values, negative_values_replaced)
        changed_row=True
    if changed_row==True:
        row=pd.concat([metadata, row])
        normed_target.loc[index]=row

normed_train=normed[normed['sampleType']=='qc']

for index,row in normed_train.iterrows():
    metadata=row.drop(metabolites)
    row=row[metabolites]

    changed_row=False
    nan_values=row[row.isna()]
    if len(nan_values)>0:
        nan_values_replaced=np.random.normal(
                loc=row.min(),
                scale=np.std(row, ddof=1) * 0.01,
                size=len(nan_values),
            )
        row.loc[nan_values.index]=nan_values_replaced
        changed_row=True
    negative_values=row[row<0]
    if len(negative_values) > 0:
        negative_values_replaced=np.random.uniform(0, 1)*min(row[row>0])
        row.loc[negative_values.index]=row.loc[negative_values.index].replace(negative_values.values, negative_values_replaced)
        changed_row=True
    if changed_row==True:
        row=pd.concat([metadata, row])
        normed_train.loc[index]=row

normed=pd.concat([normed_train, normed_target])
normed.to_csv("normed.tsv", index=False, sep='\t')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from pyserrf.read_data import read_serff_format_data

data=read_serff_format_data("test_data/SERRF example dataset.xlsx")

qc=data['p'][data['p']['sampleType']=='qc']
samples=data['p'][data['p']['sampleType']=='sample']


data['e'].columns=[i.split("_")[0] for i in data['e'].columns]
train=data['e'][qc['label'].unique().tolist()]
target=data['e'][samples['label'].unique().tolist()]

batch=pd.concat([qc['batch'], samples['batch']])
sampleType=pd.concat([qc['sampleType'], samples['sampleType']])
batch=batch.reset_index(drop=True)
sampleType=sampleType.reset_index(drop=True)

def standard_scaler(data):
    mean=np.mean(data, axis=0)
    std=np.std(data, axis=0, ddof=1)
    centered=data-mean
    scaled=centered/std
    return scaled

    mean=np.mean(train, axis=1)
    std=np.std(train, axis=1, ddof=1)

    centered=train-mean
mean

mean
train
    scaled=centered/std


standard_scaler(train)


train.apply(standard_scaler)
train



train[]

def pyserrf(train, target, num=10, batch=None, time=None, sampleType=None, minus=False):

    #TODO: MISSING DATA/ 0 data imputation

    all_data = pd.concat([train, target], axis=1)
    normalized = np.zeros(all_data.shape[1])

    corrs_train = {}
    corrs_target = {}

    for current_batch in batch.unique():
        current_batch_qc=qc[qc['batch']==current_batch]
        current_batch_samples=samples[samples['batch']==current_batch]
        train_scale = train[current_batch_qc['label']].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )
        if target[current_batch_samples['label']].shape[1] == 0:
            target_scale = (
                target[current_batch_samples['label']]
                - target[current_batch_samples['label']].mean()
            ) / target[current_batch_samples['label']].std()
        else:
            target_scale = target[current_batch_samples['label']].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        corrs_train[current_batch] = train_scale.corr(method='spearman')
        corrs_target[current_batch] = target_scale.corr(method='spearman')





    pred = np.zeros((all_data.shape[0], len(sampleType)))
    for j in range(all_data.shape[0]):
        normalized = np.zeros(all_data.shape[1])
        qc_train_value = []
        qc_predict_value = []
        sample_value = []
        sample_predict_value = []

        for current_batch in batch.unique():
            e_current_batch = all_data.loc[:, batch == current_batch]
            corr_train = corrs_train[current_batch]
            corr_target = corrs_target[current_batch]

            corr_train_order = np.argsort(np.abs(corr_train[j, :]))[::-1]
            corr_target_order = np.argsort(np.abs(corr_target[j, :]))[::-1]

            sel_var = []
            l = num
            while len(sel_var) < num:
                sel_var = np.intersect1d(corr_train_order[:l], corr_target_order[:l])
                sel_var = sel_var[sel_var != j]
                l += 1

            train_index_current_batch = sampleType[batch == current_batch]
            factor = (
                e_current_batch.iloc[j, train_index_current_batch == "qc"].std()
                / e_current_batch.iloc[j, train_index_current_batch != "qc"].std()
            )
            if np.isnan(factor) or factor == 0 or factor < 1:
                train_data_y = e_current_batch.iloc[
                    j, train_index_current_batch == "qc"
                ]
            else:
                if (train_index_current_batch == "qc").sum() * 2 >= (
                    train_index_current_batch != "qc"
                ).sum():
                    train_data_y = (
                        e_current_batch.iloc[j, train_index_current_batch == "qc"]
                        - e_current_batch.iloc[
                            j, train_index_current_batch == "qc"
                        ].mean()
                    ) / factor
                else:
                    train_data_y = (
                        e_current_batch.iloc[j, train_index_current_batch == "qc"]
                        - e_current_batch.iloc[
                            j, train_index_current_batch == "qc"
                        ].mean()
                    ) / e_current_batch.iloc[j, train_index_current_batch == "qc"].std()

            train_data_x = e_current_batch.iloc[
                sel_var, train_index_current_batch == "qc"
            ].apply(lambda x: (x - x.mean()) / x.std(), axis=1)
            if (
                e_current_batch.iloc[sel_var, train_index_current_batch != "qc"].shape[
                    1
                ]
                == 0
            ):
                test_data_x = (
                    e_current_batch.iloc[sel_var, train_index_current_batch != "qc"]
                    - e_current_batch.iloc[
                        sel_var, train_index_current_batch != "qc"
                    ].mean()
                ) / e_current_batch.iloc[
                    sel_var, train_index_current_batch != "qc"
                ].std()
            else:
                test_data_x = e_current_batch.iloc[
                    sel_var, train_index_current_batch != "qc"
                ].apply(lambda x: (x - x.mean()) / x.std(), axis=1)

            train_NA_index = train_data_x.apply(lambda x: x.isna().sum() > 0)
            train_data_x = train_data_x.loc[:, ~train_NA_index]
            test_data_x = test_data_x.loc[:, ~train_NA_index]

            good_column = train_data_x.apply(
                lambda x: x.isna().sum() == 0
            ) & test_data_x.apply(lambda x: x.isna().sum() == 0)
            train_data_x = train_data_x.loc[:, good_column]
            test_data_x = test_data_x.loc[:, good_column]

            train_data = pd.concat([train_data_y, train_data_x], axis=1)
            if train_data.shape[1] == 1:
                norm = e_current_batch.iloc[j, :]
                normalized[batch == current_batch] = norm
            else:
                train_data.columns = ["y"] + [
                    "V" + str(i) for i in range(train_data.shape[1] - 1)
                ]
                model = RandomForestRegressor(random_state=1)
                model.fit(train_data.drop(columns=["y"]), train_data["y"])
                test_data = pd.DataFrame(test_data_x)
                test_data.columns = train_data.columns[1:]
                norm = e_current_batch.iloc[j, :]

                if minus:
                    norm[train_index_current_batch == "qc"] = e_current_batch.iloc[
                        j, train_index_current_batch == "qc"
                    ] - (
                        (
                            model.predict(train_data.drop(columns=["y"]))
                            + e_current_batch.iloc[
                                j, train_index_current_batch == "qc"
                            ].mean()
                        )
                        - all_data.iloc[j, sampleType == "qc"].mean()
                    )
                    norm[train_index_current_batch != "qc"] = e_current_batch.iloc[
                        j, train_index_current_batch != "qc"
                    ] - (
                        (
                            model.predict(test_data)
                            + e_current_batch.iloc[
                                j, train_index_current_batch != "qc"
                            ].mean()
                        )
                        - all_data.iloc[j, sampleType != "qc"].median()
                    )
                else:
                    norm[train_index_current_batch == "qc"] = e_current_batch.iloc[
                        j, train_index_current_batch == "qc"
                    ] / (
                        (
                            model.predict(train_data.drop(columns=["y"]))
                            + e_current_batch.iloc[
                                j, train_index_current_batch == "qc"
                            ].mean()
                        )
                        / all_data.iloc[j, sampleType == "qc"].mean()
                    )
                    norm[train_index_current_batch != "qc"] = e_current_batch.iloc[
                        j, train_index_current_batch != "qc"
                    ] / (
                        (
                            model.predict(test_data)
                            + e_current_batch.iloc[
                                j, train_index_current_batch != "qc"
                            ].mean()
                        )
                        / all_data.iloc[j, sampleType != "qc"].median()
                    )

                norm[train_index_current_batch != "qc"][
                    norm[train_index_current_batch != "qc"] < 0
                ] = e_current_batch.iloc[j, train_index_current_batch != "qc"][
                    norm[train_index_current_batch != "qc"] < 0
                ]
                normalized[batch == current_batch] = norm

            pred[j, :] = normalized

    normed = pred
    normed_target = normed[:, sampleType != "qc"]
    normed_target[np.isnan(normed_target)] = np.random.normal(
        np.min(normed_target[~np.isnan(normed_target)]),
        np.std(normed_target[~np.isnan(normed_target)]) * 0.1,
        np.sum(np.isnan(normed_target)),
    )
    normed_target[normed_target < 0] = np.random.uniform(0, 1) * np.min(
        normed_target[normed_target > 0]
    )

    normed_train = normed[:, sampleType == "qc"]
    normed_train[np.isnan(normed_train)] = np.random.normal(
        np.min(normed_train[~np.isnan(normed_train)]),
        np.std(normed_train[~np.isnan(normed_train)]) * 0.1,
        np.sum(np.isnan(normed_train)),
    )
    normed_train[normed_train < 0] = np.random.uniform(0, 1) * np.min(
        normed_train[normed_train > 0]
    )

    return {"normed_train": normed_train, "normed_target": normed_target}

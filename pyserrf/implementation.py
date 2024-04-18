import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pyserrf.read_data import read_serff_format_data
from pyserrf.utils import replace_zero_values, replace_nan_values, center_data, standard_scaler

data = read_serff_format_data("test_data/SERRF example dataset.xlsx")

e = data["e"]
p = data["p"]
batches = p["batch"]
e_matrix = data["e_matrix"]


training = e_matrix[[i for i in e_matrix.columns if "QC" in i]]
target = e_matrix[[i for i in e_matrix.columns if "QC" not in i]]

num=10

###START FUNCTION HERE


all = pd.concat([training, target], axis=1)

#all.columns=[i.split("_")[0] for i in all.columns]
#normalized = np.zeros(len(all.columns))


batch='A'

LA QUESTIONE RENAME LABEL IDENTICHE VA CAPITA A MONTE



for index, row in all.iterrows():
    for batch in batches.unique():
        all_current_batch = row[p[p["batch"] == batch]["label"]]
        changed_row=False
        if 0 in row:
            row = replace_zero_values(row)
            changed_row=True
        if row.isna().any():
            row = replace_nan_values(row)
            changed_row=True
        if changed_row:
            all.loc[index] = row


corrs_train={}
corrs_target={}


for batch in batches.unique():
    train_current_batch = training[p[(p["batch"] == batch)&(p["sampleType"] == 'qc')]["label"]]
    train_current_batch_scaled=train_current_batch.apply(standard_scaler, axis=1)#.transpose()
    target_current_batch = target[p[(p["batch"] == batch)&(p["sampleType"] == 'sample')]["label"]]
    if len(target_current_batch.columns)==0:
        corrs_train[batch]=np.nan
    else:
        target_current_batch_scaled=target_current_batch.apply(standard_scaler, axis=1)#.transpose()

    corrs_train[batch]=train_current_batch_scaled.transpose().corr(method='spearman')
    corrs_target[batch]=target_current_batch_scaled.transpose().corr(method='spearman')



pred = pd.DataFrame(index=range(all.shape[0]), columns=range(all.shape[1]))
batch='A'

for index,row in all.iterrows():
    break
    #normalized = np.zeros(len(all.columns))
    normalized=pd.Series()
    qc_train_value = []
    qc_predict_value= []
    sample_value = []
    sample_predict_value = []
    for batch in batches.unique():
        current_time=p[p['batch']==batch]['time']
        e_current_batch = all[p[p["batch"] == batch]["label"]]
        corr_train=corrs_train[batch]
        corr_target=corrs_target[batch]
        corr_train_order=corr_train.abs().loc[index].sort_values(ascending=False)
        corr_target_order=corr_target.abs().loc[index].sort_values(ascending=False)

        sel_var=set()
        l=num
        while len(sel_var)<num:
            sel_var=sel_var.union(set(corr_train_order[1:l].index).intersection(set(corr_target_order[1:l].index)))
            sel_var.discard(index)
            l+=1

        current_batch_qc_labels=p[(p['batch']==batch)&(p['sampleType']=='qc')]['label'].tolist()
        current_batch_sample_labels=p[(p['batch']==batch)&(p['sampleType']=='sample')]['label'].tolist()
        current_batch_qc_data=row[current_batch_qc_labels]
        current_batch_target_data=row[current_batch_sample_labels]

        factor=np.std(current_batch_qc_data, ddof=1)/np.std(current_batch_target_data, ddof=1)

        if ((factor == 0) | (factor==np.nan) | (factor <1)):
            train_data_y = center_data(current_batch_qc_data)
        else:
            if len(current_batch_qc_data)*2 > len(current_batch_target_data):
                train_data_y = (current_batch_qc_data-current_batch_qc_data.mean())/factor
            else:
                train_data_y = center_data(current_batch_qc_data)
        train_data_x=e_current_batch.loc[list(sel_var),current_batch_qc_labels].apply(standard_scaler, axis=1).transpose()

        ####DA CAPIRE. CONTROLLA CODICE R
        if e_current_batch.loc[list(sel_var),current_batch_qc_labels].empty:
            test_data_x = standard_scaler(e_current_batch.loc[list(sel_var),current_batch_sample_labels])
        ####DA CAPIRE. CONTROLLA CODICE R
        else:
            test_data_x=e_current_batch.loc[list(sel_var),current_batch_sample_labels].apply(standard_scaler, axis=1).transpose()
        train_NA_index=train_data_x.apply(lambda x:x.isna().any())

        train_data_x=train_data_x[train_NA_index[~train_NA_index].index]
        test_data_x=test_data_x[train_NA_index[~train_NA_index].index]
        ### TODO: DA CAPIRE, PER ORA NON INCLUSA -> PROBABILMENTE se test_data_x e una serie la transponi
        #if(!"matrix" %in% class(test_data_x)){ # !!!
        #  test_data_x = t(test_data_x)
        #}
        ### DA CAPIRE, PER ORA NON INCLUSA -> PROBABILMENTE se test_data_x e una serie la transponi
        good_column_train=train_data_x.apply(lambda x:~x.isna().any())
        good_column_test=test_data_x.apply(lambda x:~x.isna().any())
        good_column=good_column_train[good_column_test]

        train_data_x = train_data_x[good_column.index]
        test_data_x = test_data_x[good_column.index]
        ### TODO: DA CAPIRE, PER ORA NON INCLUSA -> PROBABILMENTE se test_data_x e una serie la transponi
        #if(!"matrix" %in% class(test_data_x)){ # !!!
        #  test_data_x = t(test_data_x)
        #}
        ### DA CAPIRE, PER ORA NON INCLUSA -> PROBABILMENTE se test_data_x e una serie la transponi
        train_data_y.name='y'
        train_data=pd.concat([train_data_y, train_data_x], axis=1)
        if len(train_data.columns)==1:
            norm = e_current_batch.loc[index]
            normalized=pd.concat([normalized, norm])
        else:

            train_data.columns=['y']+[f"V{i}" for i in range(1,len(train_data.columns))]
            #random_seed=1
            model=RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=1)
            model.fit(X=train_data[train_data.columns[1::]], y=train_data['y'], sample_weight=None)
            test_data = test_data_x
            test_data.columns = train_data.columns[1::]
            norm = e_current_batch.loc[index]

            minus=True
            if minus:
                norm[current_batch_qc_labels]=e_current_batch.loc[index, current_batch_qc_labels]-((model.predict(train_data[train_data.columns[1::]])+e_current_batch.loc[index, current_batch_qc_labels].mean())-all.loc[index,p[p["sampleType"] == 'qc']["label"]].mean())
                norm[current_batch_sample_labels]=e_current_batch.loc[index, current_batch_sample_labels]-((model.predict(test_data)+e_current_batch.loc[index, current_batch_sample_labels].mean())-all.loc[index,p[p["sampleType"] == 'sample']["label"]].median())
            else:
                norm[current_batch_qc_labels]=e_current_batch.loc[index, current_batch_qc_labels]/((model.predict(train_data[train_data.columns[1::]])+e_current_batch.loc[index, current_batch_qc_labels].mean())/all.loc[index,p[p["sampleType"] == 'qc']["label"]].mean())
                norm[current_batch_sample_labels]=e_current_batch.loc[index, current_batch_sample_labels]/((model.predict(test_data)+e_current_batch.loc[index, current_batch_sample_labels].mean() - model.predict(test_data).mean())/all.loc[index,p[p["sampleType"] == 'sample']["label"]].median())

zero_norm_values=norm[current_batch_sample_labels][norm[current_batch_sample_labels]<186026.346163]
zero_norm_values_e_value=e_current_batch.loc[index,current_batch_sample_labels].loc[zero_norm_values.index]
original_order=norm[current_batch_sample_labels].index
norm[current_batch_sample_labels]=norm[current_batch_sample_labels].drop(zero_norm_values.index)
norm[current_batch_sample_labels].drop(zero_norm_values.index)
for i in zero_norm_values_e_value.index:
    norm[current_batch_sample_labels].loc[]

norm[current_batch_sample_labels].loc['sample01']=0

norm.loc[i, ]
zero_norm_values
zero_norm_values_e_value

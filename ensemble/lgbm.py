# Parameters
XGB_WEIGHT = 0.6095
BASELINE_WEIGHT = 0.0048
OLS_WEIGHT = 0.0812
NN_WEIGHT = 0.0200

XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

# BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


import numpy as np
import pandas as pd
# import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
from datetime import datetime

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout, BatchNormalization
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Imputer



##### READ IN RAW DATA
#
# # print( "\nReading data from disk ...")
prop = pd.read_pickle('../Data/properties')
train = pd.read_pickle('../Data/train')
# sample = pd.read_csv('../Data/sample_submission.csv')
submission = pd.read_csv("../Data/sample_submission.csv")

#
# #################
# # DOWNSAMPLE # comment out if running full code and not testing
# #################
# # print( "\nReading downsampled, locally saved pickles from disk ...")
# # sampled_prop = prop.sample(100000)
# # sampled_prop.to_pickle('sampled_prop')
# # sampled_train = train.sample(10000)
# # sampled_train.to_pickle('sampled_train')
# # prop = pd.read_pickle('sampled_prop')
# # train = pd.read_pickle('sampled_train')
# #################
#
# # Rankings according to Nikunj XGB importance f_score
prop['N-LivingAreaProp'] = prop['calculatedfinishedsquarefeet']/prop['lotsizesquarefeet'] #1
prop['N-ValueRatio'] = prop['taxvaluedollarcnt']/prop['taxamount'] #2
prop['N-ValueProp'] = prop['structuretaxvaluedollarcnt']/prop['landtaxvaluedollarcnt'] #3
prop["N-location"] = prop["latitude"] + prop["longitude"] #4
prop["N-location-2"] = prop["latitude"]*prop["longitude"] #5
prop['N-ExtraSpace'] = prop['lotsizesquarefeet'] - prop['calculatedfinishedsquarefeet']#6

zip_count = prop['regionidzip'].value_counts().to_dict()
prop['N-zip_count'] = prop['regionidzip'].map(zip_count) #7
prop['N-TaxScore'] = prop['taxvaluedollarcnt']*prop['taxamount'] #8

group = prop.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
prop['N-Avg-structuretaxvaluedollarcnt'] = prop['regionidcity'].map(group)#9
prop["N-structuretaxvaluedollarcnt-2"] = prop["structuretaxvaluedollarcnt"] ** 2 #10



################
################
##  LightGBM  ##
################
################
#
# This section is (I think) originally derived from SIDHARTH's script:
#   https://www.kaggle.com/sidharthkumar/trying-lightgbm
# which was forked and tuned by Yuqing Xue:
#   https://www.kaggle.com/yuqingxue/lightgbm-85-97
# and updated by me (Andy Harless):
#   https://www.kaggle.com/aharless/lightgbm-with-outliers-remaining
# and a lot of additional changes have happened since then,
#   the most recent of which are documented in my comments above


### PROCESS DATA FOR LIGHTGBM

print( "\nProcessing data for LightGBM ..." )
for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')
df_train.fillna(df_train.median(),inplace = True)
# df_train.fillna(-1,inplace = True)


x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                         'propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train['logerror'].values
BASELINE_PRED = np.mean(y_train)
print(x_train.shape, y_train.shape)

print('info:', x_train.info())

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)



del df_train; gc.collect()

x_train = x_train.values.astype(np.float32, copy=False)
d_train = lgb.Dataset(x_train, label=y_train)


##### RUN LIGHTGBM

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3

np.random.seed(0)
random.seed(0)

print("\nFitting LightGBM model ...")
clf = lgb.train(params, d_train, 430)

del d_train; gc.collect()
# del x_train; gc.collect() # keep x train for ensemble

print("\nPrepare for LightGBM prediction ...")
print("   Read sample file ...")
print("   ...")
submission['parcelid'] = submission['ParcelId']
print("   Merge with property data ...")
df_test = submission.merge(prop, on='parcelid', how='left')
print("   ...")
del submission, prop; gc.collect()
print("   ...")
#df_test['Ratio_1'] = df_test['taxvaluedollarcnt']/df_test['taxamount']
x_test = df_test[train_columns]
print("   ...")
del df_test; gc.collect()
print("   Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
print("   ...")
x_test = x_test.values.astype(np.float32, copy=False)

print("\nStart LightGBM prediction ...")
p_test = clf.predict(x_test)
p_train = clf.predict(x_train) # for making ensemble method
pd.DataFrame(p_train).to_pickle('lgm_train_preds')
pd.DataFrame(p_test).to_pickle('lgm_test_preds')

#
del x_test; gc.collect()
del x_train; gc.collect()





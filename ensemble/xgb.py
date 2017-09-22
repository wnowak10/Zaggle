# Parameters
XGB_WEIGHT = 0.6095
BASELINE_WEIGHT = 0.0048
OLS_WEIGHT = 0.0812
NN_WEIGHT = 0.0200

XGB1_WEIGHT = 0.8083  # Weight of first in combination of two XGB models

# BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg


import numpy as np
import pandas as pd
import xgboost as xgb
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
##  XGBoost   ##
################
################

# This section is (I think) originally derived from Infinite Wing's script:
#   https://www.kaggle.com/infinitewing/xgboost-without-outliers-lb-0-06463
# inspired by this thread:
#   https://www.kaggle.com/c/zillow-prize-1/discussion/33710
# but the code has gone through a lot of changes since then


#### RE-READ prop FILE
#### (I tried keeping a copy, but the program crashed.)
# prop = pd.read_pickle('../Data/properties')

# print( "\nRe-reading prop file ...")
#
#
#
#
# ##### PROCESS DATA FOR XGBOOST
#
print( "\nProcessing data for XGBoost ...")
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))

train_df = train.merge(prop, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = prop.drop(['parcelid'], axis=1)
# shape
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df_minus_outliers=train_df[ train_df.logerror > -0.4 ]
train_df_minus_outliers=train_df[ train_df.logerror < 0.419 ]
x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)

x_train_minus_outliers=train_df_minus_outliers.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df_minus_outliers["logerror"].values.astype(np.float32)
y_mean = np.mean(train_df_minus_outliers.logerror)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train_minus_outliers.shape, x_test.shape))




##### RUN XGBOOST

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
	# 'booster' : 'gblinear',
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train_minus_outliers, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 250
print("num_boost_rounds="+str(num_boost_rounds))

# train model
print( "\nTraining XGBoost ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost ...")
# xgb_pred1 = model.predict(dtest)

xgb1_train_preds = model.predict(xgb.DMatrix(x_train))
pd.DataFrame(xgb1_train_preds).to_pickle('xgb1_pred_train')

print( "\nFirst XGBoost predictions:" )
xgb_pred1_test = model.predict(dtest)
print( pd.DataFrame(xgb_pred1_test).head() )
pd.DataFrame(xgb_pred1_test).to_pickle('xgb1_pred_test')


# model.save_model('xgb1 model')
# xgb_pred1 = pd.read_pickle('xgb_pred1')

#### RUN XGBOOST AGAIN

print("\nSetting up data for XGBoost ...")
# xgboost params
xgb_params = {
	# 'booster' : 'gblinear',
    'eta': 0.033,
    'max_depth': 7, #was 6
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

num_boost_rounds = 150
print("num_boost_rounds="+str(num_boost_rounds))

print( "\nTraining XGBoost again ...")
model2 = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

print( "\nPredicting with XGBoost again ...")
xgb_pred2_test = model2.predict(dtest)
print( pd.DataFrame(xgb_pred2_test).head() )
pd.DataFrame(xgb_pred2_test).to_pickle('xgb2_pred_test')


xgb2_train_preds = model2.predict(xgb.DMatrix(x_train))
pd.DataFrame(xgb2_train_preds).to_pickle('xgb2_pred_train')


# xgb_pred2 = pd.read_pickle('xgb_pred2')




#### COMBINE XGBOOST RESULTS
# xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2
# xgb_pred = xgb_pred1

# print( "\nCombined XGBoost predictions:" )
# print( pd.DataFrame(xgb_pred).head() )

del train_df
del x_train
del x_test
# del prop
del dtest
del dtrain
### del xgb_pred1
### del xgb_pred2
gc.collect()


# to do
# we are removing outliers and therefore the train predictions are too small



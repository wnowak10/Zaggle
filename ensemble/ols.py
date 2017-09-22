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


##### READ IN RAW DATA
#
# # print( "\nReading data from disk ...")
# prop = pd.read_pickle('../Data/properties')
# train = pd.read_pickle('../Data/train')
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


################
################
##    OLS     ##
################
################

# This section is derived from the1owl's notebook:
#    https://www.kaggle.com/the1owl/primer-for-the-zillow-pred-approach
# which I (Andy Harless) updated and made into a script:
#    https://www.kaggle.com/aharless/updated-script-version-of-the1owl-s-basic-ols

np.random.seed(17)
random.seed(17)

print( "\n\nProcessing data for OLS ...")

# train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
# prop = pd.read_csv("../input/prop_2016.csv")
prop = pd.read_pickle('../Data/properties')
train = pd.read_csv('../Data/train_2016_v2.csv', parse_dates=["transactiondate"])


# Rankings according to Nikunj XGB importance f_score
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


print(len(train),len(prop),len(submission))

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    return df

def MAE(y, ypred):
    #logerror=log(Zestimate)−log(SalePrice)
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)

train = pd.merge(train, prop, how='left', on='parcelid')
y = train['logerror'].values
test = pd.merge(submission, prop, how='left', left_on='ParcelId', right_on='parcelid')
prop = [] #memory

exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']
col = [c for c in train.columns if c not in exc]

train = get_features(train[col])
test['transactiondate'] = '2016-01-01' #should use the most common training date
test = get_features(test[col])


print("\nFitting OLS...")
reg = LinearRegression(n_jobs=-1)
reg.fit(train, y); print('fit...')
print(MAE(y, reg.predict(train)))

ols_test_preds = reg.predict(test)
ols_train_preds = reg.predict(train)

print('test preds shape:', ols_test_preds.shape)
print('train preds shape:', ols_train_preds.shape)


pd.DataFrame(ols_test_preds).to_pickle('ols_test_preds')
pd.DataFrame(ols_train_preds).to_pickle('ols_train_preds')

# train = [];  y = [] #memory




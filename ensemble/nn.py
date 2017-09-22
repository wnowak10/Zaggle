
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
from sklearn.linear_model import LinearRegression
import random
import datetime as dt
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer




#####################
#####################
#  Neural Network  ##
#####################
#####################

# Neural network copied from this script:
#   https://www.kaggle.com/aharless/keras-neural-network-lb-06492 (version 20)
# which was built on the skeleton in this notebook:
#   https://www.kaggle.com/prasunmishra/ann-using-keras


# Read in data for neural network
print( "\n\nProcessing data for Neural Network ...")
print('\nLoading train, prop and sample data...')
train = pd.read_csv("../Data/train_2016_v2.csv", parse_dates=["transactiondate"])
prop = pd.read_pickle('../Data/properties')
sample = pd.read_csv('../Data/sample_submission.csv')


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


print('Fitting Label Encoder on prop...')
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)
    if prop[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(prop[c].values))
        prop[c] = lbl.transform(list(prop[c].values))

print('Creating training set...')
df_train = train.merge(prop, how='left', on='parcelid')

df_train["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_train["transactiondate_year"] = df_train["transactiondate"].dt.year
df_train["transactiondate_month"] = df_train["transactiondate"].dt.month
df_train['transactiondate_quarter'] = df_train['transactiondate'].dt.quarter
df_train["transactiondate"] = df_train["transactiondate"].dt.day

print('Filling NA/NaN values...' )
df_train.fillna(-1.0)

print('Creating x_train and y_train from df_train...' )
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode','fireplacecnt', 'fireplaceflag'], axis=1)
y_train = df_train["logerror"]



y_mean = np.mean(y_train)
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

print('Creating df_test...')
sample['parcelid'] = sample['ParcelId']

print("Merging Sample with property data...")
df_test = sample.merge(prop, on='parcelid', how='left')

df_test["transactiondate"] = pd.to_datetime(df_train["transactiondate"])
df_test["transactiondate_year"] = df_test["transactiondate"].dt.year
df_test["transactiondate_month"] = df_test["transactiondate"].dt.month
df_test['transactiondate_quarter'] = df_test['transactiondate'].dt.quarter
df_test["transactiondate"] = df_test["transactiondate"].dt.day
x_test = df_test[train_columns]

print('Shape of x_test:', x_test.shape)
print("Preparing x_test...")
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)


## Preprocessing
print("\nPreprocessing neural network data...")
imputer= Imputer()
imputer.fit(x_train.iloc[:, :])
x_train = imputer.transform(x_train.iloc[:, :])
imputer.fit(x_test.iloc[:, :])
x_test = imputer.transform(x_test.iloc[:, :])

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train_for_predict  = x_train.copy()

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2) # trying to add validation

len_x=int(x_train.shape[1])
print("len_x is:",len_x)


# Neural Network
print("\nSetting up neural network model...")
nn = Sequential()
nn.add(Dense(units = 360 , kernel_initializer = 'normal', activation = 'tanh', input_dim = len_x))
nn.add(Dropout(.17))
nn.add(Dense(units = 150 , kernel_initializer = 'normal', activation = 'relu'))
nn.add(BatchNormalization())
nn.add(Dropout(.4))
nn.add(Dense(units = 60 , kernel_initializer = 'normal', activation = 'relu'))
nn.add(BatchNormalization())
nn.add(Dropout(.32))
nn.add(Dense(units = 25, kernel_initializer = 'normal', activation = 'relu'))
nn.add(BatchNormalization())
nn.add(Dropout(.22))
nn.add(Dense(1, kernel_initializer='normal'))
nn.compile(loss='mae', optimizer='adam')

print("\nFitting neural network model...")
nn.fit(np.array(x_train), np.array(y_train), 
				validation_data=(x_valid,y_valid.values), # trying to add validation data
				batch_size = 32, 
				epochs = 60, 
				verbose=2)

print("\nPredicting with neural network model...")
#print("x_test.shape:",x_test.shape)
y_pred_test = nn.predict(x_test, verbose = 1)
y_pred_train = nn.predict(x_train_for_predict, verbose = 1)


print( "\nPreparing results for write..." )
nn_pred_test = y_pred_test.flatten()
nn_pred_train = y_pred_train.flatten()

print( "Type of nn_pred is ", type(nn_pred_test) )
print( "Shape of nn_pred is ", nn_pred_test.shape )

print( "\nNeural Network predictions:" )
print( pd.DataFrame(nn_pred_test).shape )
print( pd.DataFrame(nn_pred_test).head() )


pd.DataFrame(nn_pred_test).to_pickle('nn_pred_test')
pd.DataFrame(nn_pred_train).to_pickle('nn_pred_train')


# Cleanup
del train
del prop
del sample
del x_train
del x_test
del df_train
del df_test
# del y_pred_ann
gc.collect()


# version 1 forked from andy harless kaggle
# v2 added validation data to watch during training. implement early stopping?




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

train = pd.read_pickle('../Data/train')
y_train = train.logerror
#
# # ########################
# # ########################
# # ##  Combine and Save  ##
# # ########################
# # ########################

lgm_test_preds = pd.read_pickle('lgm_test_preds')
lgm_train_preds = pd.read_pickle('lgm_train_preds')
xgb1_pred_test = pd.read_pickle('xgb1_pred_test')
xgb1_pred_train = pd.read_pickle('xgb1_pred_train')
xgb2_pred_test = pd.read_pickle('xgb2_pred_test')
xgb2_pred_train = pd.read_pickle('xgb2_pred_train')
ols_test_preds = pd.read_pickle('ols_test_preds')
ols_train_preds = pd.read_pickle('ols_train_preds')
nn_pred_test = pd.read_pickle('nn_pred_test')
nn_pred_train = pd.read_pickle('nn_pred_train')

# my NN is too small. why?

# so take all the preds on train
# and fit a linear model to find coefficients to combine all these
# predictions together
# train this on y_train

# once i get optimal weights on each individual prediction
# use this model and apply to a matrix that is all of the test_preds files above

# how to get all the train dfs agove into one big matrix?

# pd.merge on index of all of them. should be same
pd.concat([lgm_train_preds, xgb1_pred_train, xgb2_pred_train, ols_train_preds, nn_pred_train], axis = 1)


from sklearn import linear_model
reg = linear_model.LinearRegression()

reg.fit(x_train.values, y_train.values)


##### COMBINE PREDICTIONS
# fit model for linear weights based on training

# lgm_train_preds

# print( "\nCombining XGBoost, LightGBM, NN, and baseline predictions ..." )
# lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT - NN_WEIGHT) / (1 - OLS_WEIGHT)
# xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
# baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)
# nn_weight0 = NN_WEIGHT / (1 - OLS_WEIGHT)
# pred0 = 0
# pred0 += xgb_weight0*xgb_pred
# pred0 += baseline_weight0*BASELINE_PRED
# pred0 += lgb_weight*p_test
# pred0 += nn_weight0*nn_pred
#
# print( "\nCombined XGB/LGB/NN/baseline predictions:" )
# print( pd.DataFrame(pred0).head() )
#
test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']
#


# print( "\nPredicting with OLS and combining with XGB/LGB/NN/baseline predicitons: ..." )
for i in range(len(test_dates)):
#     test['transactiondate'] = test_dates[i]
#     pred = OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0
    pred = (lightgbm_preds+ xgb1_pred + xgb2_pred) / 3
    # submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]
    submission[test_columns[i]] = pred
    submission = submission.round(4)

#     print('predict...', i)
#
# print( "\nCombined XGB/LGB/NN/baseline/OLS predictions:" )
print( submission.head() )



# # ##### WRITE THE RESULTS
#
#
# # print( "\nWriting results to disk ..." )
submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
#
# # print( "\nFinished ...")
#
#
#
#
# ########################
# ########################
# ##  Version Comments  ##
# ########################
# ########################
#
# # version 1: forked from "XGBoost, LightGBM, and OLS"
# # version 2: deleted old comments and added keras imports
# # version 3: added NN code (not using result yet)
# # version 5: use results of NN with 0.1 weight (arbitrary first guess)
# # version 7: reduce neural network weight to 0.05 and adjust others proportionally
# # version 8: reduce NN weight to 0.03
# # version 9: reduce NN weight to 0.01
# # version 10: increasee NN weight to 0.02
# # version 11: added Nikunj features

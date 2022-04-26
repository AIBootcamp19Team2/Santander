import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import os
import glob

def train_predict_logistic_regression(X_train, y_train, X_test):
    clf = LogisticRegression(C = 2.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    return y_pred

def train_predict_xgboost(X_train, y_train, X_test):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.02
    param['max_depth'] = 5
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['nthread'] = 6
    param['gamma'] = 1.0
    param['min_child_weight'] = 5
    param['subsample'] = 0.8
    param['colsample_bytree'] = 1.0
    param['colsample_bylevel'] = 0.7
    num_round = 500
    param['seed'] = 123089
    plst = list(param.items())
    xgmat_train = xgb.DMatrix(X_train, label=y_train, missing = -999.0)
    xgmat_test = xgb.DMatrix(X_test, missing = -999.0)
    bst = xgb.train(plst, xgmat_train, num_round)
    y_pred = bst.predict( xgmat_test )
    return y_pred



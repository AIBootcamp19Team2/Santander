import numpy as np
import xgboost as xgb
import pandas as pd
from scipy.stats import pearsonr


from sklearn.preprocessing import normalize,StandardScaler
import os
from sklearn.datasets import dump_svmlight_file

from sklearn.cluster import KMeans
from preprocess import *

np.random.seed(981004)

INPUT_PATH = '/Users/jyp/Desktop/ml_basics/team_project/data/'
OUTPUT_PATH = '/Users/jyp/Desktop/ml_basics/team_project/data/features/'

train = pd.read_csv(INPUT_PATH + 'Santander Customer Satisfaction_train.csv')
test = pd.read_csv(INPUT_PATH + 'Santander Customer Satisfaction_test.csv')

train, test = process_base(train, test)
train, test = drop_sparse(train, test)
train, test = drop_duplicated(train, test)
train, test = add_features(train, test, ['SumZeros'])
train, test = normalize_features(train, test)

flist = [x for x in train.columns if not x in ['ID','TARGET']]

flist_kmeans = []
for ncl in range(2,11):
    cls = KMeans(n_clusters=ncl)
    cls.fit_predict(train[flist].values)
    train['kmeans_cluster'+str(ncl)] = cls.predict(train[flist].values)
    test['kmeans_cluster'+str(ncl)] = cls.predict(test[flist].values)
    flist_kmeans.append('kmeans_cluster'+str(ncl))

train[['ID']+flist_kmeans].append(test[['ID']+flist_kmeans], ignore_index=True).to_csv(OUTPUT_PATH + 'kmeans_feats.csv', index=False)

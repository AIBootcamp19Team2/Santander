import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import normalize,StandardScaler



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

flist = [x for x in train.columns if not x in ['ID','TARGET']]

pca = PCA(n_components=2)
x_train_projected = pca.fit_transform(normalize(train[flist], axis=0))
x_test_projected = pca.transform(normalize(test[flist], axis=0))
train.insert(1, 'PCAOne', x_train_projected[:, 0])
train.insert(1, 'PCATwo', x_train_projected[:, 1])
test.insert(1, 'PCAOne', x_test_projected[:, 0])
test.insert(1, 'PCATwo', x_test_projected[:, 1])
pca_feats = train[['ID', 'PCAOne', 'PCATwo']].append(test[['ID', 'PCAOne', 'PCATwo']], ignore_index=True)
pca_feats.to_csv(OUTPUT_PATH + 'pca_feats.csv')
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


INPUT_PATH = '/Users/jyp/Desktop/ml_basics/team_project/data/'
OUTPUT_PATH = '/Users/jyp/Desktop/ml_basics/team_project/data/'

def process_base(train, test):
    train.loc[(train['var38']>117310.979) & (train['var38']<117310.98), 'var38'] = -999.0
    test.loc[(test['var38']>117310.979) & (test['var38']<117310.98), 'var38'] = -999.0

    train.loc[train['var3']==-999999, 'var3'] = -999.0
    test.loc[test['var3']==-999999, 'var3'] = -999.0

    for f in ['imp_op_var40_comer_ult1', 'imp_op_var40_efect_ult3', 'imp_op_var41_comer_ult3', 'imp_sal_var16_ult1']:
        train.loc[train[f]==0.0, f] = -999.0
        test.loc[test[f]==0.0, f] = -999.0

    return train, test

def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','TARGET']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test

def drop_duplicated(train, test):
    # drop var6 variable (it is similar to var29)
    flist = [x for x in train.columns if not x in ['ID','TARGET']]            
    train.drop([x for x in flist if 'var6' in x], axis=1, inplace=True)
    test.drop([x for x in flist if 'var6' in x], axis=1, inplace=True)

    # remove repeated columns with _0 in the name
    flist = [x for x in train.columns if not x in ['ID','TARGET']]        
    flist_remove = []
    for i in range(len(flist)-1):
        v = train[flist[i]].values
        for j in range(i+1, len(flist)):
            if np.array_equal(v, train[flist[j]].values):
                if '_0' in flist[j]:
                    flist_remove.append(flist[j])
                elif  '_0' in flist[i]:
                    flist_remove.append(flist[i])
    train.drop(flist_remove, axis=1, inplace=True)
    test.drop(flist_remove, axis=1, inplace=True)

    flist_remove = ['saldo_medio_var13_medio_ult1', 'delta_imp_reemb_var13_1y3', 'delta_imp_reemb_var17_1y3', 
                   'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var17_out_1y3',
                   'delta_imp_trasp_var33_in_1y3', 'delta_imp_trasp_var33_out_1y3']
    train.drop(flist_remove, axis=1, inplace=True)
    test.drop(flist_remove, axis=1, inplace=True)

    return train, test


def add_features(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','TARGET']]
    if 'SumZeros' in features:
        train.insert(1, 'SumZeros', (train[flist] == 0).astype(int).sum(axis=1))
        test.insert(1, 'SumZeros', (test[flist] == 0).astype(int).sum(axis=1))

    flist = [x for x in train.columns if not x in ['ID','TARGET']]

    if 'tsne' in features:
        tsne_feats = pd.read_csv(OUTPUT_PATH + 'features/tsne_feats.csv')
        train = pd.merge(train, tsne_feats, on='ID', how='left')
        test = pd.merge(test, tsne_feats, on='ID', how='left')

    if 'pca' in features:
        pca_feats = pd.read_csv(OUTPUT_PATH + 'features/pca_feats.csv')
        train = pd.merge(train, pca_feats, on='ID', how='left')
        test = pd.merge(test, pca_feats, on='ID', how='left')
    return train, test

def normalize_features(train, test):
    flist = [x for x in train.columns if not x in ['ID','TARGET']]
    for f in flist:
        if train[f].max() == 9999999999.0:
            fmax = train.loc[train[f]<9999999999.0, f].max()
            train.loc[train[f]==9999999999.0, f] = fmax + 1

        if len(train.loc[train[f]<0, f].value_counts()) == 1:
            train.loc[train[f]<0, f] = -1.0
            test.loc[test[f]<0, f] = -1.0
            fmax = max(np.max(train[f]), np.max(test[f]))
            if fmax > 0:
                train.loc[train[f]>0, f] = 1.0*train.loc[train[f]>0, f]/fmax
                test.loc[test[f]>0, f] = 1.0*test.loc[test[f]>0, f]/fmax

        if len(train.loc[train[f]<0, f]) == 0:
            fmax = max(np.max(train[f]), np.max(test[f]))
            if fmax > 0:
                train.loc[train[f]>0, f] = 1.0*train.loc[train[f]>0, f]/fmax
                test.loc[test[f]>0, f] = 1.0*test.loc[test[f]>0, f]/fmax

        if len(train.loc[train[f]<0, f].value_counts()) > 1:
            fmax = max(np.max(train[f]), np.max(test[f]))
            if fmax > 0:
                train[f] = 1.0*train[f]/fmax
                test[f] = 1.0*test[f]/fmax

    return train, test
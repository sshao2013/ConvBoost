import os

import numpy as np
import scipy.io
import pandas as pd

from Utils.utils import get_project_root
from data.process_gotov import read_data_gotov


def loadingDB(args, fileDir, DB=79):
    if DB == 79:
        matfile = fileDir + 'Opp' + str(DB) + '.mat'
        print(matfile)
        data = scipy.io.loadmat(matfile)

        X_train = np.transpose(data['trainingData'])
        X_valid = np.transpose(data['valData'])
        X_test = np.transpose(data['testingData'])
        print('normalising... zero mean, unit variance')
        mn_trn = np.mean(X_train, axis=0)
        std_trn = np.std(X_train, axis=0)
        X_train = (X_train - mn_trn) / std_trn
        X_valid = (X_valid - mn_trn) / std_trn
        X_test = (X_test - mn_trn) / std_trn
        print('normalising...X_train, X_valid, X_test... done')

        y_train = data['trainingLabels'].reshape(-1) - 1
        y_valid = data['valLabels'].reshape(-1) - 1
        y_test = data['testingLabels'].reshape(-1) - 1

        y_train = pd.get_dummies(y_train, prefix='labels')
        y_valid = pd.get_dummies(y_valid, prefix='labels')
        y_test = pd.get_dummies(y_test, prefix='labels')

        y_valid.insert(17, 'labels_17', 0, allow_duplicates=False)  # as validation only has 17 lables
        print('loading the 79-dim matData successfully . . .')

    if DB == 9:
        gotov_data = read_data_gotov(fileDir).item()
        X_train = gotov_data["X_train"].astype(np.float32)
        y_train = gotov_data["y_train"].reshape(-1).astype(np.int64)
        X_valid = gotov_data["X_val"].astype(np.float32)
        y_valid = gotov_data["y_val"].reshape(-1).astype(np.int64)
        X_test = gotov_data["X_test"].astype(np.float32)
        y_test = gotov_data["y_test"].reshape(-1).astype(np.int64)

        # remove null class
        Xtrain_null_cls_index = np.squeeze(np.array(np.argwhere(y_train == 0)))
        Ytrain_null_cls_index = np.squeeze(np.array(np.argwhere(y_train == 0)))
        Xval_null_cls_index = np.squeeze(np.array(np.argwhere(y_valid == 0)))
        Yval_null_cls_index = np.squeeze(np.array(np.argwhere(y_valid == 0)))
        Xtest_null_cls_index = np.squeeze(np.array(np.argwhere(y_test == 0)))
        Ytest_null_cls_index = np.squeeze(np.array(np.argwhere(y_test == 0)))

        X_train = np.delete(X_train, Xtrain_null_cls_index, 0)
        y_train = np.delete(y_train, Ytrain_null_cls_index, 0)
        X_valid = np.delete(X_valid, Xval_null_cls_index, 0)
        y_valid = np.delete(y_valid, Yval_null_cls_index, 0)
        X_test = np.delete(X_test, Xtest_null_cls_index, 0)
        y_test = np.delete(y_test, Ytest_null_cls_index, 0)
        # end of remove null class

        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0)
        X_train = (X_train - mean_train) / std_train
        X_valid = (X_valid - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train

        y_train = pd.get_dummies(y_train, prefix='labels')
        y_valid = pd.get_dummies(y_valid, prefix='labels')
        y_test = pd.get_dummies(y_test, prefix='labels')

        print('the GOTOV dataset was normalized to zero-mean, unit variance')

    if DB == 52:
        matfile = fileDir + 'PAMAP2.mat'
        data = scipy.io.loadmat(matfile)
        X_train = data['X_train']
        X_valid = data['X_valid']
        X_test = data['X_test']
        y_train = data['y_train'].reshape(-1)
        y_valid = data['y_valid'].reshape(-1)
        y_test = data['y_test'].reshape(-1)

        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0)
        X_train = (X_train - mean_train) / std_train
        X_valid = (X_valid - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train

        y_train = pd.get_dummies(y_train, prefix='labels')
        y_valid = pd.get_dummies(y_valid, prefix='labels')
        y_test = pd.get_dummies(y_test, prefix='labels')

        print('the PAMAP2 dataset was normalized to zero-mean, unit variance')
        print('loading the 33HZ PAMAP2 52d matData successfully . . .')

    X_train = X_train.astype(np.float32)
    X_valid = X_valid.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = y_train.astype(np.uint8)
    y_valid = y_valid.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def data_processing(pids, fileDir):
    data_x = []
    data_y = []
    for pid in pids:
        data_pid_x = scipy.io.loadmat(os.path.join(fileDir, fr"{pid}.mat"))['data_x']
        data_pid_y = scipy.io.loadmat(os.path.join(fileDir, fr"{pid}.mat"))['data_y']
        data_x.append(data_pid_x)
        data_y.append(np.squeeze(data_pid_y))
    x = np.vstack(data_x)
    y = np.concatenate(data_y)
    return x, y

import numpy as np
from tqdm import tqdm

def data_prepro(X_train, X_val, X_test, cfg_prepro):
    print("**************************************************")
    print("preprocessing")
    print("norm level:{}, method:{}".format(cfg_prepro.name, cfg_prepro.method))
    if cfg_prepro.name == "dataset":
        if cfg_prepro.method == "min-max":
            X_train_std = min_max(X_train, X_train, cfg_prepro.axis)
            X_val_std = min_max(X_val, X_train, cfg_prepro.axis)
            X_test_std = min_max(X_test, X_train, cfg_prepro.axis)
            if cfg_prepro.axis:
                print("axis=0:True")
            else:
                print("axis=0:False")
        elif cfg_prepro.method == "std":
            X_train_std = std(X_train, X_train, cfg_prepro.axis)
            X_val_std = std(X_val, X_train, cfg_prepro.axis)
            X_test_std = std(X_test, X_train, cfg_prepro.axis)
            if cfg_prepro.axis:
                print("axis=0:True")
            else:
                print("axis=0:False")
        elif cfg_prepro.method == "0-1":
            X_train_std = X_train / 4095
            X_val_std = X_val / 4095
            X_test_std = X_test / 4095
        else:
            X_train_std = X_train
            X_val_std = X_val
            X_test_std = X_test
    elif cfg_prepro.name == "pixwise":
        if cfg_prepro.method == "min-max":
            X_train_std = pix_wise_min_max(X_train)
            X_val_std = pix_wise_min_max(X_val)
            X_test_std = pix_wise_min_max(X_test)
        elif cfg_prepro.method == "std":
            X_train_std = pix_wise_std(X_train)
            X_val_std = pix_wise_std(X_val)
            X_test_std = pix_wise_std(X_test)
    else:
        print("error")
    print("X_train.shape = {}, X_val.shape = {}, X_test.shape = {}".format(X_train_std.shape, X_val_std.shape, X_test_std.shape))
    return X_train_std, X_val_std, X_test_std

def min_max(X, X_std, axis):
    if axis:
        X = (X - np.min(X_std, axis=0)) / (np.max(X_std, axis=0) - np.min(X_std, axis=0))
    else:
        X = (X - np.min(X_std)) / (np.max(X_std) - np.min(X_std))
    return X

def std(X, X_std, axis):
    if axis:
        X = (X - np.mean(X_std, axis=0)) / np.std(X_std, axis=0)
    else:
        X = (X - np.mean(X_std)) / np.std(X_std)
    return X


def pix_wise_min_max(X):
    min_vals = X.min(axis=1, keepdims=True)
    max_vals = X.max(axis=1, keepdims=True)
    X_norm = (X - min_vals) / (max_vals - min_vals)
    return X_norm

def pix_wise_std(X):
    mean_vals = X.mean(axis=1, keepdims=True)
    std_vals = X.std(axis=1, keepdims=True)
    X_norm = (X - mean_vals) / std_vals
    return X_norm
from twaml.data import dataset
from twaml.data import scale_weight_sum
from twaml.pytorch import TworchDataset

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import time
import os
import sys


rseed = 414


def prepare_raw_data(branchinfo_file="vars.yaml", region="2j2b"):
    with open(branchinfo_file, "r") as f:
        branches = yaml.load(f, Loader=yaml.FullLoader)
    branches = branches[region]

    ttbar = dataset.from_pytables(
        f"/home/ddavis/ATLAS/data/h5s/ttbar_r{region}.h5", label=0, auxlabel=1
    )
    tW_DR = dataset.from_pytables(
        f"/home/ddavis/ATLAS/data/h5s/tW_DR_r{region}.h5", label=1, auxlabel=1
    )
    tW_DS = dataset.from_pytables(
        f"/home/ddavis/ATLAS/data/h5s/tW_DS_r{region}.h5", label=1, auxlabel=0
    )

    tW_DR.keep_columns(branches)
    ttbar.keep_columns(branches)
    tW_DS.keep_columns(branches)
    scale_weight_sum(tW_DS, ttbar)
    scale_weight_sum(tW_DR, ttbar)
    tW_DR.weights *= 50
    tW_DS.weights *= 50
    ttbar.weights *= 100

    X = pd.concat([ttbar.df, tW_DR.df, tW_DS.df]).to_numpy()
    w = np.concatenate([ttbar.weights, tW_DR.weights, tW_DS.weights])
    y = np.concatenate([ttbar.label_asarray, tW_DR.label_asarray, tW_DS.label_asarray])
    z = np.concatenate(
        [ttbar.auxlabel_asarray, tW_DR.auxlabel_asarray, tW_DS.auxlabel_asarray]
    )

    return (X, y, w, z)


def prepare_torch_data():
    X_raw, y_raw, w_raw, z_raw = prepare_raw_data()
    folder = KFold(n_splits=2, shuffle=True, random_state=rseed)

    for i, (test_idx, train_idx) in enumerate(folder.split(X_raw)):
        X_train, y_train, w_train = X_raw[train_idx], y_raw[train_idx], w_raw[train_idx]
        X_test, y_test, w_test = X_raw[test_idx], y_raw[test_idx], w_raw[test_idx]

        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_train)

        trainDataset = TworchDataset(X_train, y_train, w_train)
        testDataset = TworchDataset(X_test, y_test, w_test)

        trainLoader = DataLoader(trainDataset, batch_size=512, shuffle=True)
        testLoader = DataLoader(testDataset, batch_size=512, shuffle=True)

        for batch_idx, (X, y, w) in enumerate(trainLoader):
            if y[0] == 1:
                print(batch_idx, X[0], y[0], w[0])

        break


def main():
    prepare_torch_data()
    return 0


if __name__ == "__main__":
    main()

from twaml.data import dataset
from twaml.data import scale_weight_sum
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, model_from_json
from keras.utils.vis_utils import plot_model
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


def simple_model(
    inshape, nlayers=5, nnode=64, lr=0.03, momentum=0.7, opt=None, plot=None
):
    input_layer = Input(shape=inshape, name="Input")
    dense_layer = Dense(nnode, activation="elu", name="Dense1")(input_layer)
    for i in range(nlayers - 1):
        dense_layer = Dense(nnode, activation="elu", name=f"Dense{i+2}")(dense_layer)
    output_layer = Dense(1, activation="sigmoid", name="Output")(dense_layer)
    model = Model(inputs=[input_layer], outputs=[output_layer], name="Model")
    if opt is None:
        opt = SGD(lr=lr, momentum=momentum)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    if plot is not None:
        model.summary()
        plot_model(model, to_file=f"{plot}")
    return model


def train_model(X_train, y_train, w_train, X_test, y_test, w_test, output_dir=None):
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if output_dir is None:
        timestamp = time.time()
        output_dir = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    origdir = os.getcwd()
    os.mkdir(output_dir)
    os.chdir(output_dir)
    simpmodel = simple_model((X_train.shape[1],), plot="model.png")
    fitres = simpmodel.fit(
        X_train,
        y_train,
        epochs=25,
        sample_weight=w_train,
        batch_size=512,
        validation_data=(X_test, y_test, w_test),
    )

    acc = fitres.history["acc"]
    loss = fitres.history["loss"]
    val_acc = fitres.history["val_acc"]
    val_loss = fitres.history["val_loss"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(acc, label="train acc")
    ax[0].plot(val_acc, label="valid acc")
    ax[0].set_title("accuracy")
    ax[0].set_xlabel("epoch")
    ax[1].plot(loss, label="train loss")
    ax[1].plot(val_loss, label="valid loss")
    ax[1].set_title("loss")
    ax[1].set_xlabel("epoch")
    fig.savefig("metrics.pdf")

    with open("model.json", "w") as f:
        f.write(simpmodel.to_json())
    simpmodel.save_weights("model.h5")
    joblib.dump(scaler, "scaler.save")

    os.chdir(origdir)


def load_model(from_dir):
    with open(f"{from_dir}/model.json", "r") as f:
        simpmodel = model_from_json(f.read())
    simpmodel.load_weights(f"{from_dir}/model.h5")
    scaler = joblib.load(f"{from_dir}/scaler.save")
    return simpmodel, scaler


def prepare_data(branchinfo_file="vars.yaml", region="2j2b"):
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


def main():
    if len(sys.argv) < 2:
        return 0

    X, y, w, z = prepare_data()
    folder = KFold(n_splits=3, shuffle=True, random_state=rseed)

    if sys.argv[1] == "train":
        for i, (train_idx, test_idx) in enumerate(folder.split(X)):
            X_train, y_train, w_train = X[train_idx], y[train_idx], w[train_idx]
            X_test, y_test, w_test = X[test_idx], y[test_idx], w[test_idx]
            train_model(
                X_train,
                y_train,
                w_train,
                X_test,
                y_test,
                w_test,
                output_dir=f"proofOfConcept_fold{i}",
            )
        return 0

    if sys.argv[1] == "apply":
        for i, (train_idx, test_idx) in enumerate(folder.split(X)):
            X_train, y_train, w_train = X[train_idx], y[train_idx], w[train_idx]
            X_test, y_test, w_test = X[test_idx], y[test_idx], w[test_idx]

            simpmodel, scaler = load_model(f"proofOfConcept_fold{i}")
            X_test = scaler.transform(X_test)
            X_train = scaler.transform(X_train)
            y_pred = simpmodel.predict(X_test)
            y_pred_train = simpmodel.predict(X_train)
            false_pos, true_pos, thresh = roc_curve(
                y_test, y_pred, sample_weight=w_test
            )
            plt.plot(false_pos, true_pos)
            plt.savefig(f"proofOfConcept_fold{i}/roc.pdf")
            plt.cla()

            sig = y_pred[y_test == 1]
            bkg = y_pred[y_test == 0]
            sig_train = y_pred_train[y_train == 1]
            bkg_train = y_pred_train[y_train == 0]

            plt.hist(
                [sig, bkg],
                bins=np.linspace(bkg.min(), sig.max(), 41),
                density=True,
                label=["sig", "bkg"],
                histtype="step",
                weights=[w_test[y_test == 1], w_test[y_test == 0]],
            )
            plt.hist(
                [sig_train, bkg_train],
                bins=np.linspace(bkg.min(), sig.max(), 41),
                density=True,
                label=["sig train", "bkg train"],
                histtype="step",
                weights=[w_train[y_train == 1], w_train[y_train == 0]],
                linestyle="--",
            )
            plt.legend()
            plt.savefig(f"proofOfConcept_fold{i}/hist.pdf")
            plt.cla()
        return 0


if __name__ == "__main__":
    main()
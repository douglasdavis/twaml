import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from twaml.data import dataset
from twaml.data import scale_weight_sum
import matplotlib.pyplot as plt


def gen_combined():
    branches = [
        "nloosejets",
        "pT_jetL1",
        "pTsys_lep1lep2jet1jet2",
        "mass_lep1lep2",
        "mass_lep2jet1",
        "mass_lep1jet2",
    ]
    ttbar1 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tt_a.h5", "ttbar", branches, label=0
    )
    ttbar2 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tt_d.h5", "ttbar", branches, label=0
    )
    tW_DR1 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tW_DR_48_a.h5", "tW_DR", branches, label=0
    )
    tW_DR2 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tW_DR_48_d.h5", "tW_DR", branches, label=0
    )
    tW_DR3 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tW_DR_49_a.h5", "tW_DR", branches, label=0
    )
    tW_DR4 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tW_DR_49_d.h5", "tW_DR", branches, label=0
    )
    tW_DS1 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tW_DS_56_a.h5", "tW_DS", branches, label=0
    )
    tW_DS2 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tW_DS_56_d.h5", "tW_DS", branches, label=0
    )
    tW_DS3 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tW_DS_57_a.h5", "tW_DS", branches, label=0
    )
    tW_DS4 = dataset.from_h5(
        "/Users/ddavis/Desktop/h5files/tW_DS_57_d.h5", "tW_DS", branches, label=0
    )
    ttbar = ttbar1 + ttbar2
    tW_DR = tW_DR1 + tW_DR2 + tW_DR3 + tW_DR4
    tW_DS = tW_DS1 + tW_DS2 + tW_DS3 + tW_DS4
    ttbar.to_pytables("ttbar.h5")
    tW_DR.to_pytables("tW_DR.h5")
    tW_DS.to_pytables("tW_DS.h5")


def get_combined():
    ttbar = dataset.from_pytables("ttbar.h5", "ttbar", label=0)
    tW_DR = dataset.from_pytables("tW_DR.h5", "tW_DR", label=0)
    tW_DS = dataset.from_pytables("tW_DS.h5", "tW_DS", label=1)
    scale_weight_sum(tW_DR, ttbar)
    scale_weight_sum(tW_DS, ttbar)
    tW_DR.weights *= 0.5
    tW_DS.weights *= 0.5
    print(ttbar.weights.sum(), tW_DR.weights.sum(), tW_DS.weights.sum())
    return ttbar, tW_DR, tW_DS


def main():
    ttbar, tW_DR, tW_DS = get_combined()
    sow = ttbar.weights.sum() + tW_DR.weights.sum() + tW_DS.weights.sum()
    mwfl = sow * 0.01
    y = np.concatenate(
        [
            np.ones_like(tW_DR.weights),
            np.ones_like(tW_DS.weights),
            np.zeros_like(ttbar.weights),
        ]
    )
    X = np.concatenate([tW_DR.df.values, tW_DS.df.values, ttbar.df.values])
    w = np.concatenate([tW_DR.weights, tW_DS.weights, ttbar.weights])

    folder = KFold(n_splits=3, shuffle=True, random_state=414)

    ttbar_dist = []
    ttbar_w_dist = []
    tW_dist = []
    tW_w_dist = []
    roc_aucs = []
    print("preproc done")
    print("starting folded training")
    for train_idx, test_idx in folder.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train, w_test = w[train_idx], w[test_idx]

        param = {"max_depth": 4, "n_estimators": 150, "min_child_weight": mwfl}
        param["nthread"] = 4
        param["eval_metric"] = "auc"
        model = xgb.XGBClassifier(**param)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric="logloss",
            verbose=False,
        )
        # print(model.evals_result())
        predictions = model.predict(X_test)
        predict_pro = model.predict_proba(X_test)
        roc_aucs.append(roc_auc_score(y_test, predictions))
        tW_w_dist.append(w_test[y_test == 1])
        ttbar_w_dist.append(w_test[y_test == 0])
        ttbar_dist.append(predict_pro[y_test == 0])
        tW_dist.append(predict_pro[y_test == 1])
        print("just finished a fold")
    ttbar_dist_all = np.concatenate(ttbar_dist)
    ttbar_w_dist_all = np.concatenate(ttbar_w_dist)
    tW_dist_all = np.concatenate(tW_dist)
    tW_w_dist_all = np.concatenate(tW_w_dist)
    _ = plt.hist(
        tW_dist_all.T[1],
        bins=100,
        weights=tW_w_dist_all,
        density=True,
        histtype="step",
        label="tW",
    )
    _ = plt.hist(
        ttbar_dist_all.T[1],
        bins=_[1],
        density=True,
        weights=ttbar_w_dist_all,
        histtype="step",
        label="ttbar",
    )
    _ = plt.legend()
    plt.savefig("out.pdf")
    del _


if __name__ == "__main__":
    main()

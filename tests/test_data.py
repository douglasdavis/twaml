import sys
import pandas as pd
import numpy as np
try:
    import twaml.data
except ImportError:
    sys.path.append('.')
    import twaml.data

branches = ['pT_lep1', 'pT_lep2', 'eta_lep1', 'eta_lep2']
ds = twaml.data.root_dataset(['tests/data/test_file.root'], name='myds',
                             branches=branches)
ds.construct()


def test_name():
    assert ds.name == 'myds'


def test_content():
    ts = ds.uproot_trees
    raws = [t.array('pT_lep1') for t in ts]
    raw = np.concatenate([raws])
    bins = np.linspace(0, 800, 21)
    n1, bins1 = np.histogram(raw, bins=bins)
    n2, bins2 = np.histogram(ds.df.pT_lep1.values, bins=bins)
    np.testing.assert_array_equal(n1, n2)


def test_weight():
    ts = ds.uproot_trees
    raws = [t.array('weight_nominal') for t in ts]
    raw = np.concatenate(raws)
    raw = raw * 150.0
    ds.weights = ds.weights * 150.0
    np.testing.assert_array_almost_equal(raw, ds.weights, 6)


def test_add():
    ds2 = twaml.data.root_dataset(['tests/data/test_file.root'], name='ds2',
                                  branches=branches, force_construct=True)
    ds2.weights = ds2.weights * 22
    combined = ds + ds2
    comb_w = np.concatenate([ds.weights, ds2.weights])
    comb_df = pd.concat([ds.df, ds2.df])
    np.testing.assert_array_almost_equal(comb_w, combined.weights, 5)
    np.testing.assert_array_almost_equal(comb_df.get_values(),
                                         combined.df.get_values(), 5)


def test_append():
    branches = ['pT_lep1', 'pT_lep2', 'eta_lep1', 'eta_lep2']
    ds1 = twaml.data.root_dataset(['tests/data/test_file.root'], name='myds',
                                  branches=branches, force_construct=True)
    ds2 = twaml.data.root_dataset(['tests/data/test_file.root'], name='ds2',
                                  branches=branches, force_construct=True)
    ds2.weights = ds2.weights * 5
    # raw
    comb_w = np.concatenate([ds1.weights, ds2.weights])
    comb_df = pd.concat([ds1.df, ds2.df])
    # appended
    ds1.append(ds2)
    # now test
    np.testing.assert_array_almost_equal(comb_w, ds1.weights, 5)
    np.testing.assert_array_almost_equal(comb_df.get_values(),
                                         ds1.df.get_values(), 5)


def test_label():
    ds2 = twaml.data.root_dataset(['tests/data/test_file.root'], name='ds2',
                                  branches=branches, force_construct=True)
    assert ds2.label is None
    assert ds2.label_array is None
    ds2.label = 6
    la = ds2.label_array
    la_raw = np.ones_like(ds2.weights, dtype=np.int64) * 6
    np.testing.assert_array_equal(la, la_raw)


def test_save_and_read():
    ds.to_h5('outfile.h5')
    new_ds = twaml.data.h5_dataset('outfile.h5', name=ds.name,
                                   force_construct=True)
    X1 = ds.df.values
    X2 = new_ds.df.values
    w1 = ds.weights
    w2 = new_ds.weights
    np.testing.assert_array_almost_equal(X1, X2, 6)
    np.testing.assert_array_almost_equal(w1, w2, 6)


def test_scale_weight_sum():
    ds1 = twaml.data.root_dataset(['tests/data/test_file.root'], name='myds',
                                  branches=branches, force_construct=True)
    ds2 = twaml.data.root_dataset(['tests/data/test_file.root'], name='ds2',
                                  branches=branches, force_construct=True)
    ds2.weights = np.random.randn(len(ds1)) * 10
    twaml.data.scale_weight_sum(ds1, ds2)
    testval = abs(1.0 - ds2.weights.sum()/ds1.weights.sum())
    assert testval < 1.0e-4

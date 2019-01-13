import os
import twaml.viz
import numpy as np
import matplotlib as mpl
mpl.use('agg')

def test_compare_dist():
    bins = np.linspace(-2, 2, 11)
    d1 = np.random.randn(10000)
    d2 = np.random.randn(10000)
    nh1 = np.histogram(d1, bins=bins)
    nh2 = np.histogram(d2, bins=bins)
    fig, ax, h1, h2 = twaml.viz.compare_distributions(d1, d2, bins=bins,
                                                      colors=['red', 'green'])
    np.testing.assert_array_equal(nh1[0], h1[0])
    np.testing.assert_array_equal(nh2[0], h2[0])
    fig.savefig('test.pdf')

    d1 = np.random.randn(1000)
    d2 = np.random.randn(1000)
    nh1 = np.histogram(d1)
    nh2 = np.histogram(d2, bins=nh1[1])
    fig, ax, h1, h2 = twaml.viz.compare_distributions(d1, d2, ratio=False)
    np.testing.assert_array_equal(nh1[0], h1[0])
    np.testing.assert_array_equal(nh2[0], h2[0])


def test_cleanup():
    os.remove('test.pdf')
    assert True

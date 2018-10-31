import sys
try:
    import twanet.data
except ImportError:
    sys.path.append('.')
    import twanet.data
import numpy as np


def test_name():
    assert 'name' == 'name'

# -*- coding: utf-8 -*-

"""twaml.utils

A module providing some utlities

Attributes
----------
SD_1J1B : Dict
  selection dictionary for 1j1b region
SD_2J1B : Dict
  selection dictionary for 2j1b region
SD_2J2B : Dict
  selection dictionary for 2j2b region
TEXIT : Dict
  Maps simple strings to common TeX strings
"""

import numpy as np


def get_device():
    """helper function for getting pytorch device"""
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SD_1J1B = {
    "OS": (np.equal, True),
    "elmu": (np.equal, True),
    "reg1j1b": (np.equal, True),
}

SD_2J1B = {
    "OS": (np.equal, True),
    "elmu": (np.equal, True),
    "reg2j1b": (np.equal, True),
}

SD_2J2B = {
    "OS": (np.equal, True),
    "elmu": (np.equal, True),
    "reg2j2b": (np.equal, True),
}

TEXIT = {"ttbar": r"$t\bar{t}$", "tW": r"$tW$", "elmu": r"$e\mu$"}

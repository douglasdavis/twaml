# -*- coding: utf-8 -*-

"""twaml.utils

A module providing some utlities

Attributes
----------
SELECTION_1j1b: str
  selection tW for 1j1b region
SELECTION_2j1b: str
  selection tW for 2j1b region
SELECTION_2j2b: str
  selection tW for 2j2b region
TEXIT: dict
  Maps simple strings to common TeX strings
"""


def get_device():
    """helper function for getting pytorch device"""
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SELECTION_1j1b = "(OS == True) & (elmu == True) & (reg1j1b == True)"
SELECTION_2j1b = "(OS == True) & (elmu == True) & (reg2j1b == True)"
SELECTION_2j2b = "(OS == True) & (elmu == True) & (reg2j2b == True)"

TEXIT = {
    "ttbar": r"$t\bar{t}$",
    "tW": r"$tW$",
    "elmu": r"$e\mu$",
    "tW_DR": r"$tW$",
    "tW_DS": r"$tW$ (DS)",
}

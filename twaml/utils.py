import numpy as np


def get_device():
    """helper function for getting pytorch device"""
    import torch
    import torch.cuda
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


SD_1j1b = {'OS': (np.equal, True),
           'elmu': (np.equal, True),
           'reg1j1b': (np.equal, True)}


SD_2j1b = {'OS': (np.equal, True),
           'elmu': (np.equal, True),
           'reg2j1b': (np.equal, True)}

SD_2j2b = {'OS': (np.equal, True),
           'elmu': (np.equal, True),
           'reg2j2b': (np.equal, True)}


BRANCHES_1j1b = [
    'cent_lep1lep2',
    'deltaR_lep1_jet1',
    'deltapT_lep1_jet1',
    'deltapT_lep1_lep2',
    'mT_jet1met',
    'mass_lep1jet1',
    'mass_lep2jet1',
    'pTsys_lep1lep2jet1met'
]

BRANCHES_2j1b = [
    'deltaR_lep1_jet1',
    'deltaR_lep1lep2_jet1jet2',
    'deltaR_lep1lep2_jet1jet2met',
    'deltaR_lep2_jet2',
    'deltapT_lep1_jet1',
    'deltapT_lep1_lep2',
    'deltapT_lep1lep2jet1_met',
    'deltapT_lep2_jet2',
    'mT_jet1met',
    'mass_jet1jet2',
    'mass_lep1jet1',
    'mass_lep1jet2',
    'mass_lep2jet1',
    'mv2c10_jetF',
    'pT_jetF',
    'pTsys_lep1lep2jet1jet2met',
    'pTsys_lep1lep2jet1met'
]

BRANCHES_2j2b = [
    'mass_lep1jet1',
    'mass_lep1jet2',
    'mass_lep2jet1',
    'mass_lep2jet2',
    'deltapT_lep1lep2jet1_met',
    'deltapT_lep2_jet2',
    'pT_jetF',
    'sigpTsys_lep1lep2jet1met',
    'deltaR_lep1lep2_jet1jet2',
    'deltaR_lep2_jet1',
    'deltaR_lep1_jet1',
    'deltaR_lep2_jet2',
    'mT_jet1met',
    'deltapT_jet1_met'
]


class _texdict:
    def __init__(self):
        self.data = {
            'ttbar': r'$t\bar{t}$',
            'tW': r'$tW$',
            'elmu': r'$e\mu$'
        }

    def __call__(self, val):
        if val not in self.data:
            return val
        return self.data[val]


TeXdict = _texdict()

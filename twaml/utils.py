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

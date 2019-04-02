import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from typing import Optional


def get_device(option: str = "auto"):
    """helper function for getting pytorch device

    Paramaters
    ----------
    option:
      - if "auto" it tries to get the GPU and returns CPU if unavailable
      - if "cpu" just use CPU
    """
    if option == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif option == "cpu":
        return torch.device("cpu")


device = get_device("auto")


class TworchDataset(Dataset):
    """Training dataset container"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        z: Optional[np.ndarray] = None,
    ):
        """initialize a TwamlDataset

        Parameters
        ----------
        X:
          sample feature matrix
        y:
          sample target label
        w:
          sample weights
        z:
          sample auxiliary label (for Adversarial)
        """
        self.X = torch.from_numpy(X).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.w = torch.from_numpy(weights).to(device)
        if z is not None:
            self.z = torch.from_numpy(z).to(device)
        else:
            self.z = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.z is not None:
            return (self.X[idx], self.y[idx], self.w[idx], self.z[idx])
        else:
            return (self.X[idx], self.y[idx], self.w[idx])

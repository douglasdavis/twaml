import uproot
import pandas as pd
import numpy as np
from pathlib import PosixPath
from typing import List, Dict


class dataset:
    """A class to define a dataset with a pandas.DataFrame as the payload
    of the class. The payload is lazily constructed, so a construct()
    function must be defined.

    Attributes
    ----------
    files: List[str]
      List of files delivering the dataset
    name: str
      Name for the dataset
    weights: numpy.ndarray
      The array of event weights
    df: pandas.DataFrame
      The payload of the class, a dataframe

    """

    def __init__(self, files: List[str], name: str = '',
                 weight_name: str = 'weight_nominal') -> None:
        """
        Default dataset creation

        Parameters
        ----------
        files: List[str]
          List of ROOT files to use
        name: str
          Name of the dataset (if none use first file name)
        weight_name: str
          Name of the weight branch

        """
        self._weights = None
        self._df = None
        self.files = [PosixPath(f) for f in files]
        for f in self.files:
            assert f.exists()
        if not name:
            self.name = files[0]
        else:
            self.name = name
        self.weight_name = weight_name

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, new: pd.DataFrame) -> None:
        assert len(new) == len(self._weights)
        self._df = new

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, new) -> None:
        assert len(new) == len(self._df)
        self._weights = new

    def construct(self):
        """Not implemented for base class"""
        raise NotImplementedError

    def __add__(self, other):
        """Add to datasets together"""
        raise NotImplementedError

    def append(self, other):
        """Append a dataset to an exiting one"""
        raise NotImplementedError


class root_dataset(dataset):
    """
    A ROOT specific dataset

    Attributes
    ----------
    tree_name: str
      The ROOT tree name delivering the dataset
    branches: List[str]
      The list of branches in the trees to use
    """

    def __init__(self, files: List[str], name: str = '',
                 tree_name: str = 'WtLoop_nominal',
                 weight_name: str = 'weight_nominal',
                 branches: List[str] = None,
                 select: Dict = None,
                 construct: bool = False) -> None:
        """Create a ROOT dataset

        Parameters
        ----------
        files: List[str]
          List of ROOT files to use
        name: str
          Name of the dataset (if none use first file name)
        tree_name: str
          Name of the tree in the file to use
        weight_name: str
          Name of the weight branch
        branches: List[str]
          List of branches to store in the dataset
        select: Dict
          A dictionary of selections to apply of the form:
          ``{branch_name: (numpy.ufunc, test_value)}``. the
          selections are combined using ``np.logical_and``
        construct: bool
          Force construction (normally lazily constructed)

        Examples
        --------
        Example with a single file and two branches:

        >>> ds = root_dataset(['file.root'], name='myds', branches=['pT_lep1', 'pT_lep2'])

        Example with multiple files and a selection (uses all
        branches). The selection requires the branch ``nbjets == 1``
        and ``njets >= 1``.

        >>> flist = ['file1.root', 'file2.root', 'file3.root']
        >>> ds = root_dataset(flist, select={'nbjets': (np.equal, 1), 'njets': (np.greater, 1)}

        """

        super().__init__(files, name=name, weight_name=weight_name)

        self.tree_name = tree_name
        self.weight_name = weight_name
        self.branches = branches
        self.uproot_trees = [uproot.open(file_name)[tree_name]
                             for file_name in files]
        self._selection = select
        self._weights = None
        self._df = None
        if construct:
            construct(self)
        self.constructed = construct

    def construct(self) -> None:
        """Construct the payload from the ROOT files"""
        weight_list, frame_list = [], []
        for t in self.uproot_trees:
            raw_w = t.array(self.weight_name)
            raw_f = t.pandas.df(branches=self.branches, namedecode='utf-8')
            isel = np.ones((raw_w.shape[0]), dtype=bool)
            if self._selection is not None:
                selections = {k: v[0](t.array(k), v[1]) for k, v
                              in self._selection.items()}
                for k, v in selections.items():
                    isel = np.logical_and(isel, v)
            weight_list.append(raw_w[isel])
            frame_list.append(raw_f[isel])
        self._weights = np.concatenate(weight_list)
        self._df = pd.concat(frame_list)


class h5_dataset(dataset):
    """
    h5 dataset

    Attributes
    ----------
    """

    def __init__(self, files: List[str], name: str = '',
                 weight_name: str = 'weight_nominal',
                 construct: bool = False) -> None:
        """
        Create an h5 dataset

        Parameters
        ----------
        files: List[str]
          List of h5 files to use
        name: str
          Name of the dataset (if none use first file name)
        weight_name: str
          Name of the weight branch
        construct: bool
          Force construction (normally lazily constructed)
        """
        super().__init__(self, files, name=name, weight_name=weight_name)

        if construct:
            construct(self)
        self.constructed = construct

    def construct(self) -> None:
        """Construct the payload from the h5 files"""
        raise NotImplementedError

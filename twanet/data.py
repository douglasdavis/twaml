import uproot
import pandas as pd
import numpy as np
from pathlib import PosixPath
from typing import List, Dict, Tuple, Optional


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
    is_constructed: bool
      Flag to know if the dataset is constructed
    label: Optional[int]
      Optional dataset label (as an int)

    """

    def __init__(self, files: List[str], name: str = '',
                 weight_name: str = 'weight_nominal',
                 label: Optional[int] = None) -> None:
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
        self.constructed = False
        self._label = label

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, new: pd.DataFrame) -> None:
        assert len(new) == len(self._weights), 'df length != weight length'
        self._df = new

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, new) -> None:
        assert len(new) == len(self._df), 'weight length != frame length'
        self._weights = new

    @property
    def label(self) -> Optional[int]:
        return self._label

    @label.setter
    def label(self, new: int) -> None:
        self._label = new

    @property
    def label_array(self) -> Optional[np.ndarray]:
        if self._label is not None:
            return np.ones((len(self._weights)), dtype=np.int64) * self._label
        else:
            return None

    @property
    def is_constructed(self) -> bool:
        return self.constructed

    @property
    def shape(self) -> Tuple:
        """Get shape of dataset (shortcut to pd.DataFrame.shape)"""
        return self.df.shape

    @shape.setter
    def shape(self, new) -> None:
        raise NotImplementedError('Cannot set shape manually')

    def _set_df_and_weights(self, df: pd.DataFrame, w: np.ndarray) -> None:
        assert len(df) == len(w), 'unequal length df and weights'
        self._df = df
        self._weights = w
        self.constructed = True

    def construct(self):
        """Not implemented for base class"""
        raise NotImplementedError

    def __add__(self, other: 'dataset') -> 'dataset':
        """Add to datasets together

        We perform contatenations of the dataframes and weights to
        generate a new dataset with a new payload.

        """
        assert self.is_constructed, \
            'Unconstructed df (self)'
        assert other.is_constructed, \
            'Unconstructed df (other)'
        assert self.weight_name == other.weight_name, \
            'different weight names'
        assert self.shape[1] == other.shape[1], \
            'different df columns'
        assert self.weights.shape == other.weights.shape, \
            'weight shapes are different'
        new_weights = np.concatenate([self.weights, other.weights])
        new_df = pd.concat([self.df, other.df])
        new_files = self.files + other.files
        new_ds = dataset(new_files, self.name, self.weight_name)
        new_ds._set_df_and_weights(new_df, new_weights)
        return new_ds

    def append(self, other: 'dataset') -> None:
        """Append a dataset to an exiting one

        We perform concatenations of the dataframes and weights to
        update the existing datasets payload.

        Parameters
        ----------
        other : twanaet.data.dataset
          The dataset to append

        """
        assert self.is_constructed, \
            'Unconstructed df (self)'
        assert other.is_constructed, \
            'Unconstructed df (other)'
        assert self.weight_name == other.weight_name, \
            'different weight names'
        assert self.shape[1] == other.shape[1], \
            'different df columns'
        assert self.weights.shape == other.weights.shape, \
            'weight shapes are different'
        self._df = pd.concat([self._df, other.df])
        self._weights = np.concatenate([self._weights, other.weights])
        self.files = self.files + other.files
        self.constructed = True

    def to_h5(self, file_name) -> None:
        """Write payload to disk as an h5 file with strict options

        The key in the file is the name of the dataset. The weights
        array is stored as a separate frame with the key being the
        weight_name attribute.

        Parameters
        ----------
        file_name : str
          output file name,

        """
        weights_frame = pd.DataFrame(dict(weights=self._weights))
        self._df.to_hdf(file_name, self.name, mode='w')
        weights_frame.to_hdf(file_name, self.weight_name, mode='a')


class root_dataset(dataset):
    """A ROOT specific dataset

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
                 label: Optional[int] = None,
                 force_construct: bool = False) -> None:
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

        super().__init__(files, name=name,
                         weight_name=weight_name, label=label)

        self.tree_name = tree_name
        self.weight_name = weight_name
        self.branches = branches
        self.uproot_trees = [uproot.open(file_name)[tree_name]
                             for file_name in files]
        self._selection = select
        self._weights = None
        self._df = None
        if force_construct:
            self.construct()

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
        self.constructed = True


class h5_dataset(dataset):
    """
    h5 dataset

    Attributes
    ----------
    """

    def __init__(self, file_name: str, name: str = '',
                 weight_name: str = 'weight_nominal',
                 label: Optional[int] = None,
                 force_construct: bool = False) -> None:
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
        super().__init__([file_name], name=name, label=label,
                         weight_name=weight_name)

        if force_construct:
            self.construct()

    def construct(self) -> None:
        """Construct the payload from the h5 files"""
        main_frame = pd.read_hdf(self.files[0], self.name)
        main_weight_frame = pd.read_hdf(self.files[0], self.weight_name)
        w_array = main_weight_frame.weights.values
        self._set_df_and_weights(main_frame, w_array)

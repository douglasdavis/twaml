# -*- coding: utf-8 -*-

"""twaml.data module

This module contains classes to abstract datasets using
pandas.DataFrames as the payload for feeding to machine learning
frameworks.

"""

import uproot
import pandas as pd
import h5py
import numpy as np
from pathlib import PosixPath
from typing import List, Dict, Tuple, Optional


class dataset:
    """A class to define a dataset with a pandas.DataFrame as the payload
    of the class. The twaml.data module provides a set of functions to
    construct a dataset. The class constructor should be used only in
    very special cases.

    Attributes
    ----------
    files: List[PosixPath]
      List of files delivering the dataset
    name: str
      Name for the dataset
    tree_name: str
      All of our datasets had to come from a ROOT tree at some point
    weights: numpy.ndarray
      The array of event weights
    df: pandas.DataFrame
      The payload of the class, a dataframe
    label: Optional[int]
      Optional dataset label (as an int)
    has_payload: bool
      Flag to know that the dataset actually wraps data

    """

    def __init__(self, input_files: List[str], name: Optional[str] = None,
                 tree_name: str = 'WtLoop_nominal', weight_name: str = 'weight_nominal',
                 label: Optional[int] = None) -> None:
        """Default dataset creation

        Parameters
        ----------
        input_files: List[str]
          List of input files
        name: Optional[str]
          Name of the dataset (if none use first file name)
        tree_name: str
          Name of tree which this dataset originated from
        weight_name: str
          Name of the weight branch
        label: Optional[int]
          Give dataset an integer based label
        """
        self._weights = pd.DataFrame({})
        self._df = np.array([])
        self.files = [PosixPath(f) for f in input_files]
        for f in self.files:
            assert f.exists(), '{} does not exist'.format(f)
        if name is None:
            self.name = str(self.files[0].parts[-1])
        else:
            self.name = name
        self.weight_name = weight_name
        self.tree_name = tree_name
        self._label = label

    @property
    def has_payload(self):
        has_df = not self._df.empty
        has_weights = self._weights.shape[0] > 0
        return has_df and has_weights

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
    def weights(self, new: np.ndarray) -> None:
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
        if self.label is None:
            return None
        return np.ones_like(self.weights, dtype=np.int64) * self.label

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

    def __add__(self, other: 'dataset') -> 'dataset':
        """Add to datasets together

        We perform concatenations of the dataframes and weights to
        generate a new dataset with a new payload.

        """
        assert self.has_payload, 'Unconstructed df (self)'
        assert other.has_payload, 'Unconstructed df (other)'
        assert self.weight_name == other.weight_name, 'different weight names'
        assert self.shape[1] == other.shape[1], 'different df columns'
        assert self.weights.shape == other.weights.shape, 'weight shapes are different'
        new_weights = np.concatenate([self.weights, other.weights])
        new_df = pd.concat([self.df, other.df])
        new_files = [str(f) for f in (self.files + other.files)]
        new_ds = dataset(new_files, self.name, weight_name=self.weight_name,
                         tree_name=self.tree_name, label=self.label)
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
        assert self.has_payload, 'Unconstructed df (self)'
        assert other.has_payload, 'Unconstructed df (other)'
        assert self.weight_name == other.weight_name, 'different weight names'
        assert self.shape[1] == other.shape[1], 'different df columns'
        assert self.weights.shape == other.weights.shape, 'weight shapes are different'
        self._df = pd.concat([self._df, other.df])
        self._weights = np.concatenate([self._weights, other.weights])
        self.files = self.files + other.files
        self.constructed = True

    def to_pytables(self, file_name: str) -> None:
        """Write payload to disk as an pytables h5 file with strict options

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

    def __len__(self) -> int:
        return len(self.weights)


def root_dataset(input_files: List[str], name: Optional[str] = None,
                 tree_name: str = 'WtLoop_nominal',
                 weight_name: str = 'weight_nominal',
                 branches: List[str] = None,
                 selection: Dict = None,
                 label: Optional[int] = None) -> dataset:
    """Create a ROOT dataset

    Parameters
    ----------
    input_files: List[str]
        List of ROOT input_files to use
    name: str
        Name of the dataset (if none use first file name)
    tree_name: str
        Name of the tree in the file to use
    weight_name: str
        Name of the weight branch
    branches: List[str]
        List of branches to store in the dataset
    selection: Dict
        A dictionary of selections to apply of the form:
        ``{branch_name: (numpy.ufunc, test_value)}``. the
        selections are combined using ``np.logical_and``
    label: Optional[int]
        Give the dataset an integer label

    Examples
    --------
    Example with a single file and two branches:

    >>> ds1 = root_dataset(['file.root'], name='myds',
    ...                    branches=['pT_lep1', 'pT_lep2'], label=1)

    Example with multiple input_files and a selection (uses all
    branches). The selection requires the branch ``nbjets == 1``
    and ``njets >= 1``.

    >>> flist = ['file1.root', 'file2.root', 'file3.root']
    >>> ds = root_dataset(flist, select={'nbjets': (np.equal, 1),
    ...                                  'njets': (np.greater, 1)}
    >>> ds.construct() ## construct
    >>> ds.label = 2 ## add label after the fact

    """

    ds = dataset(input_files, name, tree_name=tree_name,
                 weight_name=weight_name, label=label)

    uproot_trees = [uproot.open(file_name)[tree_name]
                    for file_name in input_files]

    weight_list, frame_list = [], []
    for t in uproot_trees:
        raw_w = t.array(weight_name)
        raw_f = t.pandas.df(branches=branches, namedecode='utf-8')
        isel = np.ones((raw_w.shape[0]), dtype=bool)
        if selection is not None:
            selections = {k: v[0](t.array(k), v[1]) for k, v
                          in selection.items()}
            for k, v in selections.items():
                isel = np.logical_and(isel, v)
        weight_list.append(raw_w[isel])
        frame_list.append(raw_f[isel])
    weights_array = np.concatenate(weight_list)
    df = pd.concat(frame_list)
    ds._set_df_and_weights(df, weights_array)
    return ds


def pytables_dataset(file_name: str, name: str,
                     tree_name: str = 'WtLoop_nominal',
                     weight_name: str = 'weight_nominal',
                     label: Optional[int] = None) -> dataset:
    """Create an h5 dataset from pytables output generated from
    dataset.to_pytables

    Parameters
    ----------
    file_name: str
        Name of h5 file containing the payload
    name: str
        Name of the dataset inside the h5 file
    tree_name: str
        Name of tree where dataset originated
    weight_name: str
        Name of the weight array inside the h5 file
    label: Optional[int]
        Give the dataset an integer label
    force_construct: bool
        Force construction (normally lazily constructed)

    Examples
    --------

    >>> ds1 = h5_dataset('ttbar.h5', name='ttbar', force_construct=True)
    >>> ds1.label = 1 ## add label to constructed dataset
    >>> ds2 = h5_dataset('tW_DR.h5', name='tW_DR', label=2)
    >>> ds2.construct() ## construct already labeled dataset

    """
    main_frame = pd.read_hdf(file_name, name)
    main_weight_frame = pd.read_hdf(file_name, weight_name)
    w_array = main_weight_frame.weights.values

    ds = dataset([file_name], name, weight_name=weight_name,
                 tree_name=tree_name, label=label)
    ds._set_df_and_weights(main_frame, w_array)
    return ds


def h5_dataset(file_name: str, name: str, columns: List[str],
               tree_name: str = 'WtLoop_nominal',
               weight_name: str = 'weight_nominal',
               label: Optional[int] = None) -> dataset:
    """Create a dataset from generic h5 input (loosely expected to be from
    the ATLAS Analysis Release utility ``ttree2hdf5``

    Parameters
    ----------
    file_name: str
        Name of h5 file containing the payload
    name: str
        Name of the dataset inside the h5 file
    columns: List[str]
        Names of columns (branches) to include in payload
    tree_name: str
        Name of tree dataset originates from
    weight_name: str
        Name of the weight array inside the h5 file
    label: Optional[int]
        Give the dataset an integer label

    """
    ds = dataset([file_name], name=name, weight_name=weight_name,
                 tree_name=tree_name, label=label)

    """Construct the payload from the h5 files"""
    f = h5py.File(file_name, mode='r')
    full_ds = f[name]
    w_array = f[name][weight_name]
    coldict = {}
    for col in columns:
        coldict[col] = full_ds[col]
    frame = pd.DataFrame(coldict)
    ds._set_df_and_weights(frame, w_array)
    return ds


def scale_weight_sum(to_update: 'dataset', reference: 'dataset') -> None:
    """
    Scale the weights of the `to_update` dataset such that the sum of
    weights are equal to the sum of weights of the `reference` dataset.

    Parameters
    ----------
    to_update : twanet.data.dataset
        dataset with weights to be scaled
    reference : twanet.data.dataset
        dataset to scale to

    """
    assert to_update.constructed, 'to_update is not constructed'
    assert reference.constructed, 'reference is not constructed'
    sum_to_update = to_update.weights.sum()
    sum_reference = reference.weights.sum()
    to_update.weights *= (sum_reference/sum_to_update)

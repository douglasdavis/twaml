# -*- coding: utf-8 -*-

"""twaml.data module

This module contains a classe to abstract datasets using
pandas.DataFrames as the payload for feeding to machine learning
frameworks and other general data investigating

"""

import uproot
import pandas as pd
import h5py
import numpy as np
import re
import yaml
from pathlib import PosixPath
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import logging

log = logging.getLogger(__name__)

__all__ = ["dataset", "scale_weight_sum"]


class dataset:
    """A class to define a dataset with a pandas.DataFrame as the payload
    of the class. The class provides a set of static functions to
    construct a dataset. The class constructor should be used only in
    very special cases.

    ``datasets`` should `always` be constructed from a staticmethod,
    currently there are 3 available:

      - :meth:`dataset.from_root`
      - :meth:`dataset.from_pytables`
      - :meth:`dataset.from_h5`

    Attributes
    ----------
    files: List[PosixPath]
      List of files delivering the dataset
    name: str
      Name for the dataset
    tree_name: str
      All of our datasets had to come from a ROOT tree at some
      point. This is the name
    weights: numpy.ndarray
      The array of event weights
    df: :class:`pandas.DataFrame`
      The payload of the class, a dataframe
    auxweights: Optional[pandas.DataFrame]
      Extra weights to have access too
    label: Optional[int]
      Optional dataset label (as an int)
    auxlabel: Optional[int]
      Optional auxiliary label (as an int) - sometimes we need two labels
    label_asarray: Optional[numpy.ndarray]
      Optional dataset label (as an array of ints)
    auxlabel_asarray: Optional[numpy.ndarray]
      Optional dataset auxiliary label (as an array of ins)
    has_payload: bool
      Flag to know that the dataset actually wraps data
    cols: List[str]
      Column names as a list of strings
    shape: Tuple
      Shape of the main payload dataframe
    wtloop_metas: Optional[Dict[str, Dict[str]]]
      A dictionary of files to meta dictionaries

    """

    _weights = None
    _df = None
    _auxweights = None
    files = None
    name = None
    weight_name = None
    tree_name = None
    _label = None
    _auxlabel = None
    wtloop_metas = None

    def _init(
        self,
        input_files: List[str],
        name: Optional[str] = None,
        tree_name: str = "WtLoop_nominal",
        weight_name: str = "weight_nominal",
        label: Optional[int] = None,
        auxlabel: Optional[int] = None,
    ) -> None:
        """Default initialization - should only be called by internal
        staticmethods ``from_root``, ``from_pytables``, ``from_h5``

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
        auxlabel: Optional[int]
          Give dataset an integer based auxiliary label
        """
        self._weights = np.array([])
        self._df = pd.DataFrame({})
        self._auxweights = None
        self.files = [PosixPath(f) for f in input_files]
        for f in self.files:
            assert f.exists(), f"{f} does not exist"
        if name is None:
            self.name = str(self.files[0].parts[-1])
        else:
            self.name = name
        self.weight_name = weight_name
        self.tree_name = tree_name
        self._label = label
        self._auxlabel = auxlabel

    @staticmethod
    def _combine_wtloop_metas(meta1, meta2) -> Optional[dict]:
        if meta1 is not None and meta2 is not None:
            return {**meta1, **meta2}
        elif meta1 is None and meta2 is not None:
            return {**meta2}
        elif meta1 is not None and meta2 is None:
            return {**meta1}
        else:
            return None

    @property
    def has_payload(self) -> bool:
        has_df = not self._df.empty
        has_weights = self._weights.shape[0] > 0
        return has_df and has_weights

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, new: pd.DataFrame) -> None:
        assert len(new) == len(self._weights), "df length != weight length"
        self._df = new

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, new: np.ndarray) -> None:
        assert len(new) == len(self._df), "weight length != frame length"
        self._weights = new

    @property
    def auxweights(self) -> pd.DataFrame:
        return self._auxweights

    @auxweights.setter
    def auxweights(self, new: pd.DataFrame) -> None:
        if new is not None:
            assert len(new) == len(self._df), "auxweights length != frame length"
        self._auxweights = new

    @property
    def label(self) -> Optional[int]:
        return self._label

    @label.setter
    def label(self, new: int) -> None:
        self._label = new

    @property
    def label_asarray(self) -> Optional[np.ndarray]:
        if self._label is None:
            return None
        return np.ones_like(self.weights, dtype=np.int64) * self._label

    @property
    def auxlabel(self) -> Optional[int]:
        return self._auxlabel

    @auxlabel.setter
    def auxlabel(self, new: int) -> None:
        self._auxlabel = new

    @property
    def auxlabel_asarray(self) -> Optional[np.ndarray]:
        if self._auxlabel is None:
            return None
        return np.ones_like(self.weights, dtype=np.int64) * self._auxlabel

    @property
    def cols(self) -> List[str]:
        return list(self.df.columns)

    @property
    def shape(self) -> Tuple:
        return self.df.shape

    @shape.setter
    def shape(self, new) -> None:
        raise NotImplementedError("Cannot set shape manually")

    def _set_df_and_weights(
        self, df: pd.DataFrame, w: np.ndarray, extra: Optional[pd.DataFrame] = None
    ) -> None:
        assert len(df) == len(w), "unequal length df and weights"
        self._df = df
        self._weights = w
        if extra is not None:
            assert len(df) == len(extra), "unequal length df and extra weights"
            self._auxweights = extra

    def keep_columns(self, cols: List[str]) -> None:
        """Drop all columns not included in ``cols``

        Parameters
        ----------
        cols: List[str]
          Columns to keep
        """
        self._df = self._df[cols]

    def keep_weights(self, weights: List[str]) -> None:
        """Drop all columns from the extra weights frame that are not in
        ``weights``

        Parameters
        ----------
        weights: List[str]
          Weights to keep in the extra weights frame
        """
        self._auxweights = self._auxweights[weights]

    def rm_weight_columns(self) -> None:
        """Remove all payload df columns which begin with ``weight_``

        If you are reading a dataset that was created retaining
        weights in the main payload, this is a useful function to
        remove them. The design of ``twaml.data.dataset`` expects
        weights to be separated from the payload's main dataframe.

        Internally this is done by calling
        :meth:`pandas.DataFrame.drop` with ``inplace`` on the payload

        """
        import re

        pat = re.compile("^weight_")
        rmthese = [c for c in self._df.columns if re.match(pat, c)]
        self._df.drop(columns=rmthese, inplace=True)

    def rmcolumns_re(self, pattern: str) -> None:
        """Remove some columns from the payload based on regex paterns

        Internally this is done by calling
        :meth:`pandas.DataFrame.drop` with ``inplace`` on the payload

        Parameters
        ----------
        pattern : str
          Regex used to remove columns
        """
        pat = re.compile(pattern)
        rmthese = [c for c in self._df.columns if re.search(pat, c)]
        self._df.drop(columns=rmthese, inplace=True)

    def rmcolumns(self, cols: List[str]) -> None:
        """Remove columns from the dataset

        Internally this is done by calling
        :meth:`pandas.DataFrame.drop` with ``inplace`` on the payload

        Parameters
        ----------
        cols: List[str]
          List of column names to remove

        """
        self._df.drop(columns=cols, inplace=True)

    def change_weights(self, wname: str) -> None:
        """Change the main weight of the dataset

        this function will swap the current main weight array of the
        dataset with one in the ``auxweights`` frame (based on its
        name in the ``auxweights`` frame).

        Parameters
        ----------
        wname:
          name of weight in ``auxweight`` DataFrame to turn into the main weight.

        """
        assert self._auxweights is not None, "extra weights do not exist"

        old_name = self.weight_name
        old_weights = self.weights
        self._auxweights[old_name] = old_weights

        self.weights = self._auxweights[wname].to_numpy()
        self.weight_name = wname

        self._auxweights.drop(columns=[wname], inplace=True)

    def append(self, other: "dataset") -> None:

        """Append a dataset to an exiting one

        We perform concatenations of the dataframes and weights to
        update the existing dataset's payload.

        if one dataset has extra weights and the other doesn't,
        the extra weights are dropped.

        Parameters
        ----------
        other : twanaet.data.dataset
          The dataset to append

        """
        assert self.has_payload, "Unconstructed df (self)"
        assert other.has_payload, "Unconstructed df (other)"
        assert self.weight_name == other.weight_name, "different weight names"
        assert self.shape[1] == other.shape[1], "different df columns"

        if self._auxweights is not None and other.auxweights is not None:
            assert (
                self._auxweights.shape[1] == other.auxweights.shape[1]
            ), "extra weights are different lengths"

        self._df = pd.concat([self._df, other.df])
        self._weights = np.concatenate([self._weights, other.weights])
        self.files = self.files + other.files
        self.wtloop_metas = self._combine_wtloop_metas(
            self.wtloop_metas, other.wtloop_metas
        )

        if self._auxweights is not None and other.auxweights is not None:
            self._auxweights = pd.concat([self._auxweights, other.auxweights])
        else:
            self._auxweights = None

    def to_pytables(self, file_name: str) -> None:
        """Write dataset to disk as a pytables h5 file (with a strict
        twaml-compatible naming scheme)

        An existing dataset label **is not stored**. The properties of
        the class that are serialized to disk:

        - ``df`` as ``{name}_payload``
        - ``weights`` as ``{name}_{weight_name}``
        - ``auxweights`` as ``{name}_auxweights``
        - ``wtloop_metas`` as ``{name}_wtloop_metas``

        These properties are wrapped in a pandas DataFrame (if they
        are not already) to be stored in a .h5 file. The
        :meth:`from_pytables` is designed to read in this output; so
        the standard use case is to call this function to store a
        dataset that was intialized via :meth:`from_root`.

        Internally this function uses :meth:`pandas.DataFrame.to_hdf`
        on a number of structures.

        Parameters
        ----------
        file_name:
          output file name,

        Examples
        --------

        >>> ds = twaml.dataset.from_root("file.root", name="myds",
        ...                              detect_weights=True, wtloop_metas=True)
        >>> ds.to_pytables("output.h5")
        >>> ds_again = twaml.dataset.from_pytables("output.h5")
        >>> ds_again.name
        'myds'

        """
        if PosixPath(file_name).exists():
            log.warning(f"{file_name} exists, overwriting")
        weights_frame = pd.DataFrame(dict(weights=self._weights))
        self._df.to_hdf(file_name, f"{self.name}_payload", mode="w")
        weights_frame.to_hdf(file_name, f"{self.name}_{self.weight_name}", mode="a")
        if self._auxweights is not None:
            self._auxweights.to_hdf(file_name, f"{self.name}_auxweights", mode="a")
        if self.wtloop_metas is not None:
            tempdict = {k: np.array([str(v)]) for k, v in self.wtloop_metas.items()}
            wtmetadf = pd.DataFrame.from_dict(tempdict)
            wtmetadf.to_hdf(file_name, f"{self.name}_wtloop_metas", mode="a")

    def __add__(self, other: "dataset") -> "dataset":
        """Add two datasets together

        We perform concatenations of the dataframes and weights to
        generate a new dataset with the combined a new payload.

        if one dataset has extra weights and the other doesn't,
        the extra weights are dropped.

        """
        assert self.has_payload, "Unconstructed df (self)"
        assert other.has_payload, "Unconstructed df (other)"
        assert self.weight_name == other.weight_name, "different weight names"
        assert self.shape[1] == other.shape[1], "different df columns"

        if self._auxweights is not None and other.auxweights is not None:
            assert (
                self._auxweights.shape[1] == other.auxweights.shape[1]
            ), "extra weights are different lengths"

        new_weights = np.concatenate([self.weights, other.weights])
        new_df = pd.concat([self.df, other.df])
        new_files = [str(f) for f in (self.files + other.files)]
        new_ds = dataset()
        new_ds._init(
            new_files,
            self.name,
            weight_name=self.weight_name,
            tree_name=self.tree_name,
            label=self._label,
            auxlabel=self._auxlabel,
        )
        new_ds.wtloop_metas = self._combine_wtloop_metas(
            self.wtloop_metas, other.wtloop_metas
        )

        if self._auxweights is not None and other.auxweights is not None:
            new_aw = pd.concat([self._auxweights, other.auxweights])
        else:
            new_aw = None

        new_ds._set_df_and_weights(new_df, new_weights, extra=new_aw)
        return new_ds

    def __len__(self) -> int:
        """length of the dataset"""
        return len(self.weights)

    def __repr__(self) -> str:
        """standard repr"""
        return f"<twaml.data.dataset(name={self.name}, shape={self.shape})>"

    def __str__(self) -> str:
        """standard str"""
        return f"dataset(name={self.name})"

    @staticmethod
    def from_root(
        input_files: Union[str, List[str]],
        name: Optional[str] = None,
        tree_name: str = "WtLoop_nominal",
        weight_name: str = "weight_nominal",
        branches: List[str] = None,
        selection: Optional[str] = None,
        label: Optional[int] = None,
        auxlabel: Optional[int] = None,
        allow_weights_in_df: bool = False,
        auxweights: Optional[List[str]] = None,
        detect_weights: bool = False,
        nthreads: Optional[int] = None,
        wtloop_meta: bool = False,
    ) -> "dataset":
        """Initialize a dataset from ROOT files

        Parameters
        ----------
        input_files:
          Single or list of ROOT input file(s) to use
        name:
          Name of the dataset (if none use first file name)
        tree_name:
          Name of the tree in the file to use
        weight_name:
          Name of the weight branch
        branches:
          List of branches to store in the dataset, if None use all
        selection:
          A string passed to pandas.DataFrame.eval to apply a selection
          based on branch/column values. e.g. ``(reg1j1b == True) & (OS == True)``
          requires the ``reg1j1b`` and ``OS`` branches to be ``True``.
        label:
          Give the dataset an integer label
        auxlabel:
          Give the dataset an integer auxiliary label
        allow_weights_in_df:
          Allow "^weight_" branches in the payload dataframe
        auxweights:
          Extra weights to store in a second dataframe.
        detect_weights:
          If True, fill the auxweights df with all "^weight_"
          branches If ``auxweights`` is not None, this option is
          ignored.
        nthreads:
          Number of threads to use reading the ROOT tree
          (see uproot.TTreeMethods_pandas.df)
        wtloop_meta:
          grab and store the `WtLoop_meta` YAML entries. stored as a dictionary
          of the form ``{ str(filename) : dict(yaml) }`` in the class variable
          ``wtloop_metas``.

        Examples
        --------
        Example with a single file and two branches:

        >>> ds1 = dataset.from_root(["file.root"], name="myds",
        ...                         branches=["pT_lep1", "pT_lep2"], label=1)

        Example with multiple input_files and a selection (uses all
        branches). The selection requires the branch ``nbjets == 1``
        and ``njets >= 1``, then label it 5.

        >>> flist = ["file1.root", "file2.root", "file3.root"]
        >>> ds = dataset.from_root(flist, selection='(nbjets == 1) & (njets >= 1)')
        >>> ds.label = 5

        Example using extra weights

        >>> ds = dataset.from_root(flist, name="myds", weight_name="weight_nominal",
        ...                        auxweights=["weight_sys_radLo", " weight_sys_radHi"])

        Example where we detect extra weights automatically

        >>> ds = dataset.from_root(flist, name="myds", weight_name="weight_nominal",
        ...                        detect_weights=True)

        Example using a ThreadPoolExecutor (16 threads):

        >>> ds = dataset.from_root(flist, name="myds", nthreads=16)

        """

        if isinstance(input_files, (str, bytes)):
            input_files = [input_files]
        else:
            try:
                iter(input_files)
            except TypeError:
                input_files = [input_files]
            else:
                input_files = list(input_files)

        executor = None
        if nthreads is not None:
            executor = ThreadPoolExecutor(nthreads)

        ds = dataset()
        ds._init(
            input_files,
            name,
            tree_name=tree_name,
            weight_name=weight_name,
            label=label,
            auxlabel=auxlabel,
        )

        if wtloop_meta:
            meta_trees = {
                file_name: uproot.open(file_name)["WtLoop_meta"]
                for file_name in input_files
            }
            ds.wtloop_metas = {
                fn: yaml.full_load(mt.array("meta_yaml")[0])
                for fn, mt in meta_trees.items()
            }

        uproot_trees = [uproot.open(file_name)[tree_name] for file_name in input_files]

        wpat = re.compile("^weight_")
        if auxweights is not None:
            w_branches = auxweights
        elif detect_weights:
            urtkeys = [k.decode("utf-8") for k in uproot_trees[0].keys()]
            w_branches = [k for k in urtkeys if re.match(wpat, k)]
            if weight_name in w_branches:
                w_branches.remove(weight_name)
        else:
            w_branches = None

        frame_list, weight_list, extra_frame_list = [], [], []
        for t in uproot_trees:
            raw_w = t.array(weight_name)
            raw_f = t.pandas.df(
                branches=branches, namedecode="utf-8", executor=executor
            )
            if not allow_weights_in_df:
                rmthese = [c for c in raw_f.columns if re.match(wpat, c)]
                raw_f.drop(columns=rmthese, inplace=True)

            if w_branches is not None:
                raw_aw = t.pandas.df(branches=w_branches, namedecode="utf-8")

            if selection is not None:
                iselec = raw_f.eval(selection)
                raw_w = raw_w[iselec]
                raw_f = raw_f[iselec]
                if w_branches is not None:
                    raw_aw = raw_aw[iselec]

            assert len(raw_w) == len(raw_f), "frame length and weight length different"
            weight_list.append(raw_w)
            frame_list.append(raw_f)
            if w_branches is not None:
                extra_frame_list.append(raw_aw)
                assert len(raw_w) == len(
                    raw_aw
                ), "aux weight length and weight length different"

        weights_array = np.concatenate(weight_list)
        df = pd.concat(frame_list)
        if w_branches is not None:
            aw_df = pd.concat(extra_frame_list)
        else:
            aw_df = None

        ds._set_df_and_weights(df, weights_array, extra=aw_df)

        return ds

    @staticmethod
    def from_pytables(
        file_name: str,
        name: str = "auto",
        tree_name: str = "none",
        weight_name: str = "auto",
        label: Optional[int] = None,
        auxlabel: Optional[int] = None,
    ) -> "dataset":
        """Initialize a dataset from pytables output generated from
        dataset.to_pytables

        The payload is extracted from the .h5 pytables files using the
        name of the dataset and the weight name. If the name of the
        dataset doesn't exist in the file you'll crash. Extra weights
        are retrieved if available.

        Parameters
        ----------
        file_name:
          Name of h5 file containing the payload
        name:
          Name of the dataset inside the h5 file. If ``"auto"`` (default),
          we attempt to determine the name automatically from the h5 file.
        tree_name:
          Name of tree where dataset originated (only for reference)
        weight_name:
          Name of the weight array inside the h5 file. If ``"auto"`` (default),
          we attempt to determine the name automatically from the h5 file.
        label:
          Give the dataset an integer label
        auxlabel:
          Give the dataset an integer auxiliary label

        Examples
        --------

        >>> ds1 = dataset.from_pytables("ttbar.h5", "ttbar")
        >>> ds1.label = 1 ## add label dataset after the fact

        """
        with h5py.File(file_name, "r") as f:
            keys = list(f.keys())
            if name == "auto":
                for k in keys:
                    if "_payload" in k:
                        name = k.split("_payload")[0]
                        break
            if weight_name == "auto":
                for k in keys:
                    if "_weight" in k:
                        weight_name = k.split(f"{name}_")[-1]
                        break

        main_frame = pd.read_hdf(file_name, f"{name}_payload")
        main_weight_frame = pd.read_hdf(file_name, f"{name}_{weight_name}")
        with h5py.File(file_name, "r") as f:
            if f"{name}_auxweights" in f:
                extra_frame = pd.read_hdf(file_name, f"{name}_auxweights")
            else:
                extra_frame = None
        w_array = main_weight_frame.weights.to_numpy()
        ds = dataset()
        ds._init(
            [file_name],
            name,
            weight_name=weight_name,
            tree_name=tree_name,
            label=label,
            auxlabel=auxlabel,
        )

        with h5py.File(file_name, "r") as f:
            if "wtloop_metas" in f:
                wtloop_metas = pd.read_hdf(file_name, f"{name}_wtloop_metas")
                ds.wtloop_metas = {
                    fn: yaml.full_load(wtloop_metas[fn].to_numpy()[0])
                    for fn in wtloop_metas.columns
                }

        ds._set_df_and_weights(main_frame, w_array, extra=extra_frame)
        return ds

    @staticmethod
    def from_h5(
        file_name: str,
        name: str,
        columns: List[str],
        tree_name: str = "WtLoop_nominal",
        weight_name: str = "weight_nominal",
        label: Optional[int] = None,
        auxlabel: Optional[int] = None,
    ) -> "dataset":
        """Initialize a dataset from generic h5 input (loosely expected to be
        from the ATLAS Analysis Release utility ``ttree2hdf5``

        The name of the HDF5 dataset inside the file is assumed to be
        ``tree_name``. The ``name`` argument is something *you
        choose*.

        Parameters
        ----------
        file_name:
          Name of h5 file containing the payload
        name:
          Name of the dataset you would like to define
        columns:
          Names of columns (branches) to include in payload
        tree_name:
          Name of tree dataset originates from (HDF5 dataset name)
        weight_name: str
          Name of the weight array inside the h5 file
        label:
          Give the dataset an integer label
        auxlabel:
          Give the dataset an integer auxiliary label

        Examples
        --------

        >>> ds = dataset.from_h5('file.h5', 'dsname', tree_name='WtLoop_EG_RESOLUTION_ALL__1up')

        """
        ds = dataset()
        ds._init(
            [file_name],
            name=name,
            weight_name=weight_name,
            tree_name=tree_name,
            label=label,
            auxlabel=auxlabel,
        )

        f = h5py.File(file_name, mode="r")
        full_ds = f[tree_name]
        w_array = f[tree_name][weight_name]
        coldict = {}
        for col in columns:
            coldict[col] = full_ds[col]
        frame = pd.DataFrame(coldict)
        ds._set_df_and_weights(frame, w_array)
        return ds


def scale_weight_sum(to_update: "dataset", reference: "dataset") -> None:
    """
    Scale the weights of the `to_update` dataset such that the sum of
    weights are equal to the sum of weights of the `reference` dataset.

    Parameters
    ----------
    to_update:
        dataset with weights to be scaled
    reference
        dataset to scale to

    """
    assert to_update.has_payload, f"{to_update} is without payload"
    assert reference.has_payload, f"{reference} is without payload"
    sum_to_update = to_update.weights.sum()
    sum_reference = reference.weights.sum()
    to_update.weights *= sum_reference / sum_to_update

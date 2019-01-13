"""
twaml command line applications
"""

import argparse
import numpy as np
from twaml.data import root_dataset


def root_to_pytables():
    """command line application which converts a set of ROOT files into a
    pytables hdf5 file via the ``twaml.data.root_dataset`` function
    and the ``to_pytables`` member function ofthe
    ``twaml.data.dataset``

    """
    parser = argparse.ArgumentParser(
        description=('Convert ROOT files to a pytables hdf5 dataset '
                     'via twaml.data.root_dataset and '
                     'twaml.data.dataset.to_pytables')

    )
    parser.add_argument('-i', '--input-files', type=str, nargs='+', required=True,
                        help='input ROOT files')
    parser.add_argument('-o', '--out', type=str, required=True,
                        help='output h5 file')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='dataset name')
    parser.add_argument('-b', '--branches', type=str, nargs='+', required=False,
                        help='branches to save (default to all')
    parser.add_argument('--tree-name', type=str, required=False,
                        default='WtLoop_nominal',
                        help='tree name')
    parser.add_argument('--weight-name', type=str, required=False,
                        default='weight_nominal',
                        help='weight branch name')
    parser.add_argument('--true-branches', type=str, nargs='+', required=False,
                        help='branches that must be true to pass selection')
    args = parser.parse_args()
    sel_dict = None
    if args.true_branches is not None:
        sel_dict = { bn : (np.equal, True) for bn in args.true_branches }

    ds = root_dataset(args.input_files, name=args.name,
                      tree_name=args.tree_name,
                      weight_name=args.weight_name,
                      selection=sel_dict,
                      branches=args.branches)
    ds.to_pytables(args.out)
    return 0

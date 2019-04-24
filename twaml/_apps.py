"""
twaml command line applications
"""

import argparse
from twaml.data import from_root
import twaml.utils
import yaml


def root2pytables():
    """command line application which converts a set of ROOT files into a
    pytables HDF5 file via the :meth:`twaml.data.from_root` function and
    the :meth:`twaml.data.dataset.to_pytables` member function of the :class:`twaml.data.dataset`
    class.

    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert ROOT files to a pytables hdf5 dataset "
            "via twaml.data.root_dataset and "
            "twaml.data.dataset.to_pytables"
        )
    )
    parser.add_argument(
        "-i",
        "--input-files",
        type=str,
        nargs="+",
        required=True,
        help="input ROOT files",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="dataset name (required when reading back into twaml.data.dataset)",
    )
    parser.add_argument(
        "-o",
        "--out-file",
        type=str,
        required=True,
        help="Output h5 file (existing file will be overwritten)",
    )
    parser.add_argument(
        "-b",
        "--branches",
        type=str,
        nargs="+",
        required=False,
        help="branches to save (defaults to all)",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        required=False,
        default="WtLoop_nominal",
        help="tree name",
    )
    parser.add_argument(
        "--weight-name",
        type=str,
        required=False,
        default="weight_nominal",
        help="weight branch name",
    )
    parser.add_argument(
        "--auxweights",
        type=str,
        nargs="+",
        required=False,
        help="extra auxiliary weights to save",
    )
    parser.add_argument(
        "--selection",
        type=str,
        required=False,
        help=(
            "A selection string or YAML file containing a map of selections "
            "(see `selection` argument docs in `twaml.data.from_root`)"
        ),
    )
    parser.add_argument(
        "--detect-weights",
        action="store_true",
        help="detect weights in the dataset, --auxweights overrides this",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        required=False,
        help="number of threads to use via ThreadPoolExecutor",
    )

    args = parser.parse_args()

    if not args.out_file.endswith(".h5"):
        raise ValueError("--out-file argument must end in .h5")

    ## if selection is not none and is a file ending in .yml or .yaml
    ## we do the yaml based selections. also a shortcut is implemented
    ## as a special case
    if args.selection is not None:
        if args.selection == "freq_shortcut":
            selection_yaml = {
                "r1j1b": twaml.utils.SELECTION_1j1b,
                "r2j1b": twaml.utils.SELECTION_2j1b,
                "r2j2b": twaml.utils.SELECTION_2j2b,
                "r3j1b": twaml.utils.SELECTION_3j1b,
                "r3jHb": twaml.utils.SELECTION_3jHb,
            }

        elif args.selection.endswith(".yml") or args.selection.endswith(".yaml"):
            with open(args.selection) as f:
                selection_yaml = yaml.full_load(f)

        full_ds = from_root(
            args.input_files,
            name=args.name,
            tree_name=args.tree_name,
            weight_name=args.weight_name,
            branches=args.branches,
            auxweights=args.auxweights,
            detect_weights=args.detect_weights,
            nthreads=args.nthreads if args.nthreads > 1 else None,
            wtloop_meta=True,
        )

        selected_dses = full_ds.apply_selections(selection_yaml)
        anchor = args.out_file.split(".h5")[0]
        for sdk, sdv in selected_dses.items():
            sdv.to_pytables(f"{anchor}_{sdk}.h5")
        return 0

    ## otherwise just take the string or None
    ds = from_root(
        args.input_files,
        name=args.name,
        tree_name=args.tree_name,
        selection=args.selection,
        weight_name=args.weight_name,
        branches=args.branches,
        auxweights=args.auxweights,
        detect_weights=args.detect_weights,
        nthreads=args.nthreads if args.nthreads > 1 else None,
        wtloop_meta=True,
    )
    ds.to_pytables(args.out_file)

    return 0

from twanet.data import root_dataset, h5_dataset
from twanet.utils import BRANCHES_2j2b, SD_2j2b
import numpy as np

def create_root_datasets() -> None:
    ttbar_files = [
        'ttbar_410472_FS_MC16a_nominal.root',
        'ttbar_410472_FS_MC16d_nominal.root'
    ]
    tW_DR_files = [
        'tW_DR_410648_FS_MC16a_nominal.root',
        'tW_DR_410648_FS_MC16d_nominal.root',
        'tW_DR_410649_FS_MC16a_nominal.root',
        'tW_DR_410649_FS_MC16d_nominal.root'
    ]

    tW_DS_files = [
        'tW_DS_410656_FS_MC16a_nominal.root',
        'tW_DS_410656_FS_MC16d_nominal.root',
        'tW_DS_410657_FS_MC16a_nominal.root',
        'tW_DS_410657_FS_MC16d_nominal.root'
    ]

    def prepend_base(flist):
        base_dir = '/var/phy/project/hep/atlas/users/drd25/top/analysis/run/all/wtnup/nominal'
        fl = ['{}/{}'.format(base_dir, f) for f in flist]
        return fl

    ttbar_files = prepend_base(ttbar_files)
    tW_DR_files = prepend_base(tW_DR_files)
    tW_DS_files = prepend_base(tW_DS_files)

    ttbar_rds = root_dataset(ttbar_files, name='ttbar', select=SD_2j2b,
                             branches=BRANCHES_2j2b, label=0,
                             force_construct=True)
    tW_DR_rds = root_dataset(tW_DR_files, name='tW_DR', select=SD_2j2b,
                             branches=BRANCHES_2j2b, label=0,
                             force_construct=True)
    tW_DS_rds = root_dataset(tW_DS_files, name='tW_DS', select=SD_2j2b,
                             branches=BRANCHES_2j2b, label=1,
                             force_construct=True)

    ttbar_rds.to_h5('ttbar.h5')
    tW_DR_rds.to_h5('tW_DR.h5')
    tW_DS_rds.to_h5('tW_DS.h5')

    return ttbar_rds, tW_DR_rds, tW_DS_rds

def read_h5_datasets():
    ttbar_ds = h5_dataset('ttbar.h5', name='ttbar',
                          label=0, force_construct=True)
    tW_DR_ds = h5_dataset('tW_DR.h5', name='tW_DR',
                          label=0, force_construct=True)
    tW_DS_ds = h5_dataset('tW_DS.h5', name='tW_DS',
                          label=0, force_construct=True)

    return ttbar_ds, tW_DR_ds, tW_DS_ds


def main():
    ttbar_rds, tW_DR_rds, tW_DS_rds = create_root_datasets()
    ttbar_hds, tW_DR_hds, tW_DS_hds = read_h5_datasets()

    print(tW_DR_rds.weights)
    print(tW_DR_hds.weights)

    np.testing.assert_array_almost_equal(ttbar_rds.df.values,
                                         ttbar_hds.df.values,
                                         5)


if __name__ == '__main__':
    main()

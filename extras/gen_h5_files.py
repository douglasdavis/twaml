from twaml.data import root_dataset, h5_dataset
from twaml.utils import BRANCHES_2j2b, SD_2j2b
from twaml.utils import BRANCHES_2j1b, SD_2j1b
from twaml.utils import BRANCHES_1j1b, SD_1j1b


def create_h5_from_root(regions, base_dir):
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
        fl = ['{}/{}'.format(base_dir, f) for f in flist]
        return fl

    ttbar_files = prepend_base(ttbar_files)
    tW_DR_files = prepend_base(tW_DR_files)
    tW_DS_files = prepend_base(tW_DS_files)

    for region in regions:
        if region == '2j2b':
            select = SD_2j2b
            branches = BRANCHES_2j2b
        elif region == '2j1b':
            select = SD_2j1b
            branches = BRANCHES_2j1b
        elif region == '1j1b':
            select = SD_1j1b
            branches = BRANCHES_1j1b
        else:
            raise ValueError('invalid region: {}', region)

        ttbar_rds = root_dataset(ttbar_files, name='ttbar_{}'.format(region),
                                 select=select, branches=branches, label=0,
                                 force_construct=True)
        tW_DR_rds = root_dataset(tW_DR_files, name='tW_DR_{}'.format(region),
                                 select=select, branches=branches, label=0,
                                 force_construct=True)
        tW_DS_rds = root_dataset(tW_DS_files, name='tW_DS_{}'.format(region),
                                 select=select, branches=branches, label=1,
                                 force_construct=True)

        ttbar_rds.to_h5('ttbar_{}.h5'.format(region))
        tW_DR_rds.to_h5('tW_DR_{}.h5'.format(region))
        tW_DS_rds.to_h5('tW_DS_{}.h5'.format(region))


def read_h5_datasets():
    ttbar_ds = h5_dataset('ttbar.h5', name='ttbar',
                          label=0, force_construct=True)
    tW_DR_ds = h5_dataset('tW_DR.h5', name='tW_DR',
                          label=0, force_construct=True)
    tW_DS_ds = h5_dataset('tW_DS.h5', name='tW_DS',
                          label=1, force_construct=True)

    return ttbar_ds, tW_DR_ds, tW_DS_ds


def main():
    base_dir = '/var/phy/project/hep/atlas/users/drd25/top/analysis/run/all/wtnup/nominal'
    create_h5_from_root(['1j1b', '2j1b', '2j2b'], base_dir)
    return 0


if __name__ == '__main__':
    main()

from .utils import report_stats


def remove_salts(nist_knovel_all, test_mols):
    """
    Remove any datapoint containing a SMILES string with multiple molecules present.
    """

    nist_knovel_all = nist_knovel_all[
        nist_knovel_all["MOL_1"].apply(lambda x: "." not in x)
    ]
    nist_knovel_all = nist_knovel_all[
        nist_knovel_all["MOL_2"].apply(lambda x: "." not in x)
    ]

    report_stats(nist_knovel_all, test_mols)

    return nist_knovel_all

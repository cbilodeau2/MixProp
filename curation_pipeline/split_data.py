import numpy as np
import pandas as pd
from .utils import report_stats


def split_nist_dippr(nist_knovel_all, args):
    """
    Split the combined NIST/DIPPR dataset sequentially, first splitting the non-pure data and second splitting
    the pure data.
    """

    ## Split non-pure data:
    nist_knovel_all["pure"] = (nist_knovel_all["MolFrac_1"] == 0.0) | (
        nist_knovel_all["MolFrac_1"] == 1.0
    )
    test_split = args["test_split"]
    subset = nist_knovel_all[~nist_knovel_all["pure"]]
    test_count = int(test_split * len(subset))

    subset_mols = pd.DataFrame(
        list(set(list(subset["MOL_1"].values) + list(subset["MOL_2"].values)))
    )

    test_size = 0
    i = 175
    while test_size < test_count:
        # print(test_size,test_count)
        test_mols = subset_mols.sample(n=i)[0].values
        if ("O" not in test_mols) & (
            "CCO" not in test_mols
        ):  # &('CCCCO' not in test_mols)&('CCCO' not in test_mols)&('c1ccccc1' not in test_mols)&('CO' not in test_mols):
            test_size = np.sum(
                nist_knovel_all["MOL_1"].apply(lambda smi: smi in test_mols)
                | nist_knovel_all["MOL_2"].apply(lambda smi: smi in test_mols)
            )
        i += 1

        nonpure_test_mols = test_mols

    ## Split pure data:
    subset = nist_knovel_all[nist_knovel_all["pure"]]
    test_count = int(test_split * len(subset))

    subset_mols = pd.DataFrame(
        list(set(list(subset["MOL_1"].values) + list(subset["MOL_2"].values)))
    )

    test_size = 0
    i = 1
    while test_size < test_count:
        test_mols = list(subset_mols.sample(n=i)[0].values)

        test_mols = list(set(test_mols + list(nonpure_test_mols)))

        if ("O" not in test_mols) & (
            "CCO" not in test_mols
        ):  # &('CCCCO' not in test_mols)&('CCCO' not in test_mols)&('c1ccccc1' not in test_mols)&('CO' not in test_mols):
            test_size = np.sum(
                nist_knovel_all["MOL_1"].apply(lambda smi: smi in test_mols)
                | nist_knovel_all["MOL_2"].apply(lambda smi: smi in test_mols)
            )
        i += 1

    report_stats(nist_knovel_all, test_mols)

    return test_mols

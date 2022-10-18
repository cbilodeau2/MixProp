import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .utils import report_stats


def flag_data(subset):
    """
    Create a list of indices corresponding to possibly erroneous data, based on
    spurious viscosity-temperature correlations.
    """
    index_drop_list = []

    pr = pearsonr(subset["T"], subset["logV"])[0]
    std = np.std(subset["logV"])

    def criteria(pr, std):
        return (pr > -0.75) & (std > 0.15)

    i = 0
    iters = 0
    max_iters = 50
    while (pr > -0.75) & (std > 0.15):
        if i >= len(subset.index):
            i = 0
            iters += 1
        if iters > max_iters:
            break
        index = subset.index[i]
        subset_dropped = subset.drop(index=index)
        pr_dropped = pearsonr(subset_dropped["T"], subset_dropped["logV"])[0]
        std_dropped = np.std(subset_dropped["logV"])

        # If dropping the data brings the series within our criteria, then drop it:
        if not criteria(pr_dropped, std_dropped):
            subset = subset_dropped
            index_drop_list.append(index)

        if pr - pr_dropped > 0.1:
            subset = subset_dropped
            index_drop_list.append(index)

        # Update pr and std and i
        pr = pearsonr(subset["T"], subset["logV"])[0]
        std = np.std(subset["logV"])
        i += 1

    index_drop_list = list(set(index_drop_list))

    return index_drop_list


def drop_flagged_data(data, test_mols):
    """
    Remove potentially spurious data based on unusual viscosity-temperature correlations.
    """

    pure1 = data[data["MolFrac_1"] == 1]
    pure1["TargetMol"] = pure1["MOL_1"]
    pure2 = data[data["MolFrac_1"] == 0]
    pure2["TargetMol"] = pure2["MOL_2"]

    pure_all = pd.concat((pure1, pure2))
    unique_mols = pure_all["TargetMol"].drop_duplicates().values

    correlations = []
    std = []
    for smi in unique_mols:
        subset = pure_all[pure_all["TargetMol"] == smi]
        if len(subset["T"].drop_duplicates()) > 1:
            correlations.append(pearsonr(subset["T"], subset["logV"])[0])
            std.append(np.std(subset["logV"]))
        else:
            correlations.append(None)
            std.append(None)

    pure_Tdep = pd.DataFrame(
        {"SMILES": unique_mols, "PR": correlations, "std": std}
    ).dropna()

    suspect_list = pure_Tdep[(pure_Tdep["PR"] > -0.75) & (pure_Tdep["std"] > 0.15)][
        "SMILES"
    ].values

    index_drop_list = []
    for smi in suspect_list:
        subset = pure_all[pure_all["TargetMol"] == smi]
        index_drop_list.extend(flag_data(subset))

    for index in index_drop_list:
        smi1, smi2, T = pure_all.loc[index][["MOL_1", "MOL_2", "T"]]

        index2drop = data[
            (data["MOL_1"] == smi1) & (data["MOL_2"] == smi2) & (data["T"] == T)
        ].index
        try:
            data = data.drop(index=index2drop)
        except:
            print("Data has already been dropped")

    report_stats(data, test_mols)

    return data

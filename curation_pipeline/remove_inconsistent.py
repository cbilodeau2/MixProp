import numpy as np
import pandas as pd
from .utils import report_stats


def remove_inconsistent(nist_knovel_all, test_mols, args):
    """
    Remove inconsistent data, defined as data where experimental values of viscosity
    vary widely.
    """
    ## Remove based on inconsistencies in pure data:--------------------

    # Create pure dataset (for further cleaning):
    df_1 = pd.DataFrame()
    df_1 = nist_knovel_all[nist_knovel_all["MolFrac_1"] == 1][
        ["MOL_1", "logV", "T"]
    ].rename(columns={"MOL_1": "SMILES"})

    df_2 = pd.DataFrame()
    df_2 = nist_knovel_all[nist_knovel_all["MolFrac_1"] == 0][
        ["MOL_2", "logV", "T"]
    ].rename(columns={"MOL_2": "SMILES"})

    nist_pure = pd.concat((df_1, df_2))
    nist_pure = nist_pure.drop_duplicates()

    nist_pure_mols = pd.DataFrame()
    nist_pure_mols["SMILES"] = [
        x[0] for x in list(nist_pure.groupby(["SMILES", "T"]).mean().index)
    ]
    nist_pure_mols["T"] = [
        x[1] for x in list(nist_pure.groupby(["SMILES", "T"]).mean().index)
    ]
    nist_pure_mols["logV_avg"] = (
        nist_pure.groupby(["SMILES", "T"]).mean()["logV"].values
    )
    nist_pure_mols["logV_std"] = nist_pure.groupby(["SMILES", "T"]).std()["logV"].values

    # Remove molecules that have a pure component STD of greater than 1:
    smi2remove = nist_pure_mols[nist_pure_mols["logV_std"] > 1.0]["SMILES"].values

    nist_knovel_all = nist_knovel_all[
        nist_knovel_all["MOL_1"].apply(lambda smi: smi not in smi2remove)
    ]
    nist_knovel_all = nist_knovel_all[
        nist_knovel_all["MOL_2"].apply(lambda smi: smi not in smi2remove)
    ]

    ## Remove based on inconsistencies in nearly-pure data:---------------

    thresh_pure = args["thresh_pure"]  # Distance from pure considered "pure-ish"
    thresh_logV = args["thresh_logV"]
    nist_near_pure = nist_knovel_all[
        (nist_knovel_all["MolFrac_1"] < thresh_pure)
        | (nist_knovel_all["MolFrac_1"] > (1 - thresh_pure))
    ]

    target_mol_list = []
    target_frac_list = []
    target_id_list = []
    for i in range(len(nist_near_pure)):
        if nist_near_pure.iloc[i]["MolFrac_1"] < 0.5:
            target_mol = nist_near_pure.iloc[i]["MOL_2"]
            target_frac = 1 - nist_near_pure.iloc[i]["MolFrac_1"]
            target_id = nist_near_pure.iloc[i]["ID_2"]
        elif nist_near_pure.iloc[i]["MolFrac_1"] > 0.5:
            target_mol = nist_near_pure.iloc[i]["MOL_1"]
            target_frac = nist_near_pure.iloc[i]["MolFrac_1"]
            target_id = nist_near_pure.iloc[i]["ID_1"]
        target_mol_list.append(target_mol)
        target_frac_list.append(target_frac)
        target_id_list.append(target_id)

    nist_near_pure["TargetMol"] = target_mol_list
    nist_near_pure["TargetFrac"] = target_frac_list
    nist_near_pure["TargetID"] = target_id_list

    nist_near_pure = nist_near_pure[
        [
            "TargetMol",
            "TargetFrac",
            "TargetID",
            "Visc",
            "Visc_Unc",
            "logV",
            "T",
            "P",
            "Ref_ID",
        ]
    ]
    nist_near_pure_medians = pd.DataFrame()

    nist_near_pure_medians["TargetMol"] = [
        x[0] for x in nist_near_pure.groupby(["TargetMol", "T"]).median()["logV"].index
    ]
    nist_near_pure_medians["T"] = [
        x[1] for x in nist_near_pure.groupby(["TargetMol", "T"]).median()["logV"].index
    ]
    nist_near_pure_medians["logV_median"] = (
        nist_near_pure.groupby(["TargetMol", "T"]).median()["logV"].values
    )

    suspicious_index_list = []
    for target_mol, target_median, T in nist_near_pure_medians[
        ["TargetMol", "logV_median", "T"]
    ].values:
        suspicious_index_list.extend(
            nist_near_pure[
                (nist_near_pure["TargetMol"] == target_mol)
                & (nist_near_pure["T"] == T)
                & (np.abs(nist_near_pure["logV"] - target_median) > thresh_logV)
            ].index
        )

    suspicious_data = nist_near_pure.loc[suspicious_index_list]

    nist_knovel_all["suspicious"] = nist_knovel_all["Ref_ID"].apply(
        lambda x: x in suspicious_data["Ref_ID"].values
    )

    nist_knovel_all = nist_knovel_all[~nist_knovel_all["suspicious"]]

    report_stats(nist_knovel_all, test_mols)

    return nist_knovel_all

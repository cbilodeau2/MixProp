import numpy as np
import pandas as pd
from .utils import soft_assign_phase, report_stats


def remove_not_liquid(nist_knovel_all, test_mols, args):
    """
    Remove compounds that are not liquid at the temperature at which they are reported.
    """

    nist_knovel_all_mols = list(
        set(
            list(nist_knovel_all["MOL_1"].values)
            + list(nist_knovel_all["MOL_2"].values)
        )
    )
    nist_knovel_all_mols_df = pd.DataFrame()
    nist_knovel_all_mols_df["SMILES"] = nist_knovel_all_mols

    # After running chemprop models:
    bp_df = pd.read_csv(args["bp_pred_path"])
    mp_df = pd.read_csv(args["mp_pred_path"])
    bp_df = bp_df[bp_df["BP"] != "Invalid SMILES"]
    mp_df = mp_df[mp_df["MP"] != "Invalid SMILES"]
    nist_knovel_all_mols_df = mp_df
    nist_knovel_all_mols_df["BP"] = bp_df["BP"].apply(lambda x: float(x) + 273)
    nist_knovel_all_mols_df["MP"] = nist_knovel_all_mols_df["MP"].apply(
        lambda x: float(x)
    )

    nist_knovel_all["Phase"] = nist_knovel_all.apply(
        lambda x: soft_assign_phase(
            x["T"], x["MOL_1"], x["MOL_2"], nist_knovel_all_mols_df
        ),
        axis=1,
    )

    nist_knovel_all = nist_knovel_all[nist_knovel_all["Phase"] == ("liquid", "liquid")]

    report_stats(nist_knovel_all, test_mols)

    return nist_knovel_all

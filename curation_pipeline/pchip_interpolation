import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .utils import series_std, report_stats


def pchip_interpolation(nist_knovel_all, test_mols):
    """
    Interpolate between data for a given mixture/temperature combination and average over data.
    """

    pairs = []
    one_off_data = []
    for smi1, smi2, id1, id2, T in (
        nist_knovel_all[["MOL_1", "MOL_2", "ID_1", "ID_2", "T"]]
        .drop_duplicates()
        .values
    ):
        subset = nist_knovel_all[
            (nist_knovel_all["MOL_1"] == smi1)
            & (nist_knovel_all["MOL_2"] == smi2)
            & (nist_knovel_all["T"] == T)
        ]

        pair_entry = {}
        pair_entry["MOL_1"], pair_entry["MOL_2"] = smi1, smi2
        pair_entry["ID_1"], pair_entry["ID_2"] = id1, id2

        frac_min = min(subset["MolFrac_1"])
        frac_max = max(subset["MolFrac_1"])
        xrange = np.arange(frac_min, frac_max + 0.1, 0.1)
        ydict = {round(x, 2): [] for x in xrange}

        pair_entry["ref_ids"] = subset.Ref_ID.drop_duplicates().values
        pair_entry["T"] = T

        # For each experimental setup:
        for smi1, smi2, ref_id in (
            subset[["MOL_1", "MOL_2", "Ref_ID"]].drop_duplicates().values
        ):
            sub_subset = subset[
                (subset["MOL_1"] == smi1)
                & (subset["MOL_2"] == smi2)
                & (subset["Ref_ID"] == ref_id)
            ].sort_values("MolFrac_1")
            sub_subset.groupby("MolFrac_1").mean()["logV"]

            # We will deal with them by averaging:
            fractions = sub_subset["MolFrac_1"].drop_duplicates()
            values = sub_subset.groupby("MolFrac_1").mean()["logV"]
            if len(fractions) > 1:

                pchip = PchipInterpolator(fractions, values)

                # Find min and max fractions for this experiment:
                sub_frac_min = min(sub_subset["MolFrac_1"])
                sub_frac_max = max(sub_subset["MolFrac_1"])

                for xval in ydict.keys():
                    if (xval >= sub_frac_min) & (xval <= sub_frac_max):
                        ydict[xval].append(pchip(xval).mean())
            else:
                one_off_data.append(
                    (
                        smi1,
                        smi2,
                        ref_id,
                        fractions.values,
                        values.values,
                        T,
                        sub_subset["Visc_Unc"].values[0],
                    )
                )

        pair_entry["Data"] = ydict
        pair_entry["Data_Avg"] = [
            list(ydict.keys()),
            [np.mean(ydict[x]) for x in ydict.keys()],
        ]
        pair_entry["Data_Std"] = [
            list(ydict.keys()),
            [np.std(ydict[x]) for x in ydict.keys()],
        ]
        pair_entry["Average Visc_Unc (mPas)"] = np.mean(sub_subset["Visc_Unc"] * 1000)
        pairs.append(pair_entry)

    pairs_df = pd.DataFrame(pairs)
    pairs_df["Series_Uncertainty"] = pairs_df["Data"].apply(series_std)
    pairs_df_clean = pairs_df

    pd_input = []
    mol_list = []
    for i in range(len(pairs_df_clean)):
        reported_unc = pairs_df_clean.iloc[i]["Average Visc_Unc (mPas)"]
        series_unc = pairs_df_clean.iloc[i]["Series_Uncertainty"]
        for frac, val in np.transpose(pairs_df_clean.iloc[i]["Data_Avg"]):
            pd_input_dict = {}
            pd_input_dict["MOL_1"], pd_input_dict["MOL_2"] = (
                pairs_df_clean.iloc[i]["MOL_1"],
                pairs_df_clean.iloc[i]["MOL_2"],
            )
            pd_input_dict["Ref_ID"] = pairs_df_clean.iloc[i]["ref_ids"]
            pd_input_dict["ID_1"], pd_input_dict["ID_2"] = (
                pairs_df_clean.iloc[i]["ID_1"],
                pairs_df_clean.iloc[i]["ID_2"],
            )
            pd_input_dict["MolFrac_1"] = frac
            pd_input_dict["logV"] = val
            pd_input_dict["T"] = pairs_df_clean.iloc[i]["T"]
            pd_input_dict["Avg_Reported_Unc"] = reported_unc
            pd_input_dict["Avg_Series_Unc"] = series_unc
            pd_input.append(pd_input_dict)
            mol_list.append(pd_input_dict["MOL_1"])
            mol_list.append(pd_input_dict["MOL_2"])
    ip_data = pd.DataFrame(pd_input)

    input_data = ip_data
    all_dicts = []
    for single_entry in one_off_data:
        single_dict = {
            "MOL_1": single_entry[0],
            "MOL_2": single_entry[1],
            "MolFrac_1": single_entry[3],
            "logV": single_entry[4],
            "T": single_entry[5],
            "Avg_Series_Unc": single_entry[6],
        }
        single_df = pd.DataFrame(single_dict)
        input_data = pd.concat((input_data, single_df))
        all_dicts.append(single_dict)

    input_data = input_data[
        [
            "MOL_1",
            "MOL_2",
            "MolFrac_1",
            "logV",
            "T",
            "Avg_Reported_Unc",
            "Avg_Series_Unc",
        ]
    ].drop_duplicates()

    report_stats(input_data, test_mols)

    return input_data

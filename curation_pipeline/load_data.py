import numpy as np
import pandas as pd


def process_nist(nist):
    """
    Apply standard processing for NIST dataset.
    Note that these settings are specific to the dataset provided.
    """

    nist["logV"] = np.log10(nist["Visc"] * 1000)

    nist = nist.drop_duplicates()
    nist = nist[nist["MOL_1"] != "MISSING_ID"]
    nist = nist[nist["MOL_2"] != "MISSING_ID"]

    return nist


def process_dippr(knovel, nist, args):
    """
    Apply standard processing for DIPPR dataset.
    Note that these settings are specific to the dataset provided.
    """

    nist_all_mols = list(set(list(nist["MOL_1"].values) + list(nist["MOL_2"].values)))

    knovel = knovel[["SMILES", "logV"]].dropna().drop_duplicates()
    knovel["in_nist"] = knovel["SMILES"].apply(lambda smi: smi in nist_all_mols)
    most_common_mols = [args["dummy_mol"]]
    knovel_mix = pd.DataFrame()

    for smi, logV, in_nist in knovel.values:
        if not in_nist:
            for dummy in most_common_mols:
                add_dict1 = {
                    "MOL_1": smi,
                    "MOL_2": dummy,
                    "MolFrac_1": 1.0,
                    "logV": logV,
                }
                add_dict2 = {
                    "MOL_1": dummy,
                    "MOL_2": smi,
                    "MolFrac_1": 0.0,
                    "logV": logV,
                }
                add_df = pd.DataFrame([add_dict1, add_dict2])
            knovel_mix = pd.concat((knovel_mix, add_df))

    knovel_mix["T"] = 298
    knovel_mix["Ref_ID"] = "ref1"
    return knovel_mix


def load_data(args):
    """
    Load and process NIST and DIPPR datasets.
    """

    nist = pd.read_csv(args["NIST"])
    dippr = pd.read_csv(args["DIPPR"])
    nist = process_nist(nist)
    dippr = process_dippr(dippr, nist, args)
    nist_knovel_all = pd.concat((nist, dippr))

    return nist_knovel_all

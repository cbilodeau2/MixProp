import numpy as np
import pandas as pd


def report_stats(nist_knovel_all, test_mols):
    """
    Prints the number of molecules/datapoints in the test and training sets.
    
    nist_knovel_all: pandas dataframe containing the dataset
    test_mols: list containing compounds to held out of the training set and assigned to the test set    
    """

    # Average over all data that looks like duplicates before checkpointing
    nist_knovel_all_nodup = nist_knovel_all.groupby(
        ["MOL_1", "MOL_2", "MolFrac_1"]
    ).mean()
    nist_knovel_all_nodup.reset_index(inplace=True)

    nist_knovel_all_nodup["test_1"] = nist_knovel_all_nodup["MOL_1"].apply(
        lambda smi: smi in test_mols
    )
    nist_knovel_all_nodup["test_2"] = nist_knovel_all_nodup["MOL_2"].apply(
        lambda smi: smi in test_mols
    )
    nist_knovel_all_nodup = nist_knovel_all_nodup[
        ["MOL_1", "MOL_2", "MolFrac_1", "logV", "test_1", "test_2"]
    ]
    test_data = nist_knovel_all_nodup[
        nist_knovel_all_nodup["test_1"] | nist_knovel_all_nodup["test_2"]
    ].dropna()
    train_data = nist_knovel_all_nodup[
        ~(nist_knovel_all_nodup["test_1"] | nist_knovel_all_nodup["test_2"])
    ].dropna()

    print(
        "Total Number of Molecules:{}".format(
            len(
                list(
                    set(
                        list(nist_knovel_all_nodup["MOL_1"].values)
                        + list(nist_knovel_all_nodup["MOL_2"].values)
                    )
                )
            )
        )
    )
    print(
        "Number of Molecules in Training Set:{}".format(
            len(
                list(
                    set(
                        list(train_data["MOL_1"].values)
                        + list(train_data["MOL_2"].values)
                    )
                )
            )
        )
    )
    print(
        "Number of Molecules in Test Set:{}".format(
            len(
                list(
                    set(
                        list(test_data["MOL_1"].values)
                        + list(test_data["MOL_2"].values)
                    )
                )
            )
        )
    )
    print("Number of Datapoints in Training Set:{}".format(len(train_data)))
    print("Number of Datapoints in Test Set:{}".format(len(test_data)))
    
    return nist_knovel_all_nodup


def series_std(data):
    """
    Obtains the standard deviation between multiple series contained in a dictionary.
    
    data: dictionary containing one or more series
    """

    flag = 0
    std_list = []
    for x_val, y_val in data.items():
        if len(y_val) > 1:
            flag = 1
            std_list.append(np.std(np.array(y_val)))
    if flag == 1:
        return np.mean(std_list)


def assign_phase(T, smi1, smi2, df_mols, mp_buffer=10, bp_buffer=0):
    bp1 = df_mols[df_mols["SMILES"] == smi1]["BP"].values[0]
    bp2 = df_mols[df_mols["SMILES"] == smi2]["BP"].values[0]
    mp1 = df_mols[df_mols["SMILES"] == smi1]["MP"].values[0]
    mp2 = df_mols[df_mols["SMILES"] == smi2]["MP"].values[0]
    if mp1 > (T - mp_buffer):
        phase1 = "solid"
    elif (mp1 <= (T - mp_buffer)) & (bp1 > (T + bp_buffer)):
        phase1 = "liquid"
    else:
        phase1 = "gas"

    if mp2 > (T - mp_buffer):
        phase2 = "solid"
    elif (mp2 <= (T - mp_buffer)) & (bp2 > (T + bp_buffer)):
        phase2 = "liquid"
    else:
        phase2 = "gas"
    return phase1, phase2


def soft_assign_phase(T, smi1, smi2, nist_knovel_all_mols_df):
    try:
        return assign_phase(T, smi1, smi2, nist_knovel_all_mols_df)
    except:
        return None, None

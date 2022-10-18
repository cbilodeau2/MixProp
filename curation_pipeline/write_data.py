import numpy as np
import pandas as pd
import os


def write_data(nist_knovel_all, test_mols, input_args):

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
        ["MOL_1", "MOL_2", "MolFrac_1", "T", "logV", "test_1", "test_2"]
    ]
    test_data = nist_knovel_all_nodup[
        nist_knovel_all_nodup["test_1"] | nist_knovel_all_nodup["test_2"]
    ].dropna()
    train_data = nist_knovel_all_nodup[
        ~(nist_knovel_all_nodup["test_1"] | nist_knovel_all_nodup["test_2"])
    ].dropna()

    train_data[["MOL_1", "MOL_2", "logV"]].to_csv(
        os.path.join(input_args["out_path"], "data.csv"), index=False
    )
    train_data[["MolFrac_1", "T"]].to_csv(
        os.path.join(input_args["out_path"], "data_features.csv"), index=False
    )
    test_data[["MOL_1", "MOL_2", "logV"]].to_csv(
        os.path.join(input_args["out_path"], "test.csv"), index=False
    )
    test_data[["MolFrac_1", "T"]].to_csv(
        os.path.join(input_args["out_path"], "test_features.csv"), index=False
    )

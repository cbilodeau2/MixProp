import warnings
import time

from .load_data import load_data
from .split_data import split_nist_dippr
from .remove_salts import remove_salts
from .remove_not_liquid import remove_not_liquid
from .T_logV_correlation import drop_flagged_data
from .remove_inconsistent import remove_inconsistent
from .pchip_interpolation import pchip_interpolation
from .write_data import write_data


warnings.filterwarnings("ignore")


def prepare_dataset(input_args):
    """
    Wrapper for data curation pipeline.
    """

    start = time.time()

    print("Loading Data: ---")
    nist_knovel_all = load_data(input_args)
    print("Time Elapsed: {:.2f}".format(time.time() - start))

    print("Splitting Data: ---")
    test_mols = split_nist_dippr(nist_knovel_all, input_args)
    print("Total Time Elapsed: {:.2f}".format(time.time() - start))

    print("Removing Salts: ---")
    nist_knovel_all = remove_salts(nist_knovel_all, test_mols)
    print("Total Time Elapsed: {:.2f}".format(time.time() - start))

    print("Removing Non-Liquid Compounds: ---")
    nist_knovel_all = remove_not_liquid(nist_knovel_all,test_mols,input_args)
    print("Time Elapsed: {}".format(time.time()-start))

    print("Removing Compounds Based on Viscosity/Temperature Correlation: ---")
    nist_knovel_all = drop_flagged_data(nist_knovel_all, test_mols)
    print("Total Time Elapsed: {:.2f}".format(time.time() - start))

    print("Removing Inconsistent Data: ---")
    nist_knovel_all = remove_inconsistent(nist_knovel_all, test_mols, input_args)
    print("Total Time Elapsed: {:.2f}".format(time.time() - start))

    print("Combining Data using PCHIP Interpolation: ---")
    nist_knovel_all = pchip_interpolation(nist_knovel_all, test_mols)
    print("Total Time Elapsed: {}".format(time.time() - start))

    print("Writing Dataset to {}: ---".format(input_args["out_path"]))
    write_data(nist_knovel_all, test_mols, input_args)

    print("Total Time Elapsed: {:.2f}".format(time.time() - start))

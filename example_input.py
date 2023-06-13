from visc_pred_wrapper import visc_pred_single, visc_pred_T_curve, visc_pred_molfrac1_curve, visc_pred_read_csv


args = {'smi1': 'O',
        'smi2': 'O.c1ccccc1',
        'molfrac1': 0.25,
        'T': 285,
        'n_models':25,
        'threshold':0.022,
        'num_workers':4,
        'check_phase':False,
        'checkpoint_dir':'pretrained_models/nist_dippr_model/nist_dippr_model',     
        'input_path':'example_input.csv',
        }


#%% Option 1: Make a prediction for a single datapoint.

# out = [viscosity (cp), reliability (bool)]
out = visc_pred_single(args)


#%% Option 2: Make a prediction for a given pair of molecules at a range of
#   temperatures, but at a fixed mole fraction. Any temperature input will be
#   ignored.

# out = [viscosity (cp), temperature (K), reliability (bool)]

out = visc_pred_T_curve(args)


#%% Option 3: Make a prediction for a given pair of molecules at a range of
#   mole fractions, but at a fixed temperature. Any mole fraction input will be
#   ignored.

# out = [viscosity (cp), mole fraction, reliability (bool)]

out = visc_pred_molfrac1_curve(args)

#%% Option 4: Make a prediction for a set of datapoints specified in a csv file
#   defined using the input_path argument. Inputs not used (smi1, smi2,
#   molfrac1, T) will be ignored. The csv file should consist of four columns
#   in the following order: SMILES 1, SMILES 2, Mole Fraction, and Temperature.
#   Columns should have headers.

# out = [viscosity (cp), mole fraction, dataframe]
# dataframe contains a pandas dataframe with predictions and reliability values
# added in columns to the end of the csv input file.

out = visc_pred_read_csv(args)


import numpy as np
import os
import pandas as pd

from chemprop.train import predict
from chemprop.data import MoleculeDataset, MoleculeDataLoader, MoleculeDatapoint
from chemprop.utils import load_args, load_checkpoint, load_scalers

from rdkit import Chem


class chemprop_model():
    
    def __init__(self, checkpoint_dir):
        self.checkpoints = []
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                if fname.endswith('.pt'):
                    fname = os.path.join(root, fname)
                    scalers =load_scalers(fname)
                    self.scaler, self.features_scaler = scalers[0], scalers[1]
                    self.train_args = load_args(fname)
                    model = load_checkpoint(fname) #, cuda=True)
                    self.checkpoints.append(model)

    def __call__(self, args):
                
        if args['n_models']==None:
            args['n_models']=len(self.checkpoints)
        assert args['n_models']<=len(self.checkpoints),'Too many models requested. {} models requested.'.format(args['n_models'])
        assert args['n_models']>1, 'Multiple models are needed for reliability analysis.'
        
        model_input= MoleculeDatapoint(smiles=[args['smi1'],args['smi2']],features=[args['molfrac1'],args['T']])
        model_input = MoleculeDataset([model_input])
        model_input_loader = MoleculeDataLoader(dataset=model_input,num_workers=args['num_workers'])

        all_model_preds = []
        for model in self.checkpoints[:args['n_models']]:
             model_preds = predict(
                            model=model,
                            data_loader=model_input_loader,
                            scaler=self.scaler,
                            disable_progress_bar=True)
             all_model_preds.append(model_preds)
        
        avg_prediction = np.mean(all_model_preds)
        reliability = np.var(all_model_preds)<args['threshold']
        return avg_prediction,reliability

def load_model(checkpoint_dir):
    return chemprop_model(checkpoint_dir)


def onepoint_assertions(args):
    
    # Requirements:
    assert type(args['smi1'])==str, 'Molecules need to be input as SMILES strings.'
    assert type(args['smi2'])==str, 'Molecules need to be input as SMILES strings.'
    assert (type(args['T'])==float)|(type(args['T'])==int)|(type(args['T'])==np.int64)|(type(args['T'])==np.float), 'Temperature needs to be input as a number.'
    assert (type(args['molfrac1'])==float)|(type(args['molfrac1'])==int)|(type(args['T'])==np.int64)|(type(args['T'])==np.float), 'Mole fraction needs to be input as a number.'
    
    assert (args['molfrac1']>=0.0)&(args['molfrac1']<=1.0), 'Mole fraction needs to be between 0 and 1.'
    
    mol1 = Chem.MolFromSmiles(args['smi1'])
    mol2 = Chem.MolFromSmiles(args['smi2'])
    assert type(mol1)==Chem.rdchem.Mol, 'Invalid SMILES entry, please check that your SMILES are valid.'
    assert type(mol2)==Chem.rdchem.Mol, 'Invalid SMILES entry, please check that your SMILES are valid.'

    
    # Warnings:
    if ('.' in args['smi1'])|('.' in args['smi2']):
        print('\nWARNING: Multiple molecules are contained within a single SMILES. Predictions may be unreliable.')
    
    if (args['T']<293)|(args['T']>323):
        print('\nWARNING: Temperature is outside of recommended range (293 < T < 323). Predictions may be unreliable.')


def visc_pred_onepoint(model, args):
          
    # Check input validity:
    onepoint_assertions(args)
    
    prediction_log,reliability = model(args)
    prediction = 10**prediction_log #Prediction must be converted cP units (without the log)
    
    return prediction, reliability


def visc_pred_single(args):
    
    model = load_model(args['checkpoint_dir'])
    prediction, reliability = visc_pred_onepoint(model, args)
    
    return prediction, reliability
    

def visc_pred_T_curve(args):
    
    model = load_model(args['checkpoint_dir'])
    
    T_low = 293
    T_high = 323
    interval = 5
    T_vals = np.arange(T_low,T_high+interval,interval)
    
    preds, rels = [], []
    for T in T_vals:
        T = float(T)
        args['T'] = T
        prediction, reliability = visc_pred_onepoint(model, args)
        preds.append(prediction)
        rels.append(reliability)

    return preds, T_vals, rels


def visc_pred_molfrac1_curve(args):
    
    model = load_model(args['checkpoint_dir'])
    
        
    frac_low = 0.0
    frac_high = 1.0
    interval = 0.1
    frac_vals = np.arange(frac_low,frac_high+interval,interval)
    
    preds, rels = [], []
    for frac in frac_vals:
        frac = float(frac)
        args['molfrac1'] = frac
        prediction, reliability = visc_pred_onepoint(model, args)
        preds.append(prediction)
        rels.append(reliability)

    return preds, frac_vals, rels


def visc_pred_read_csv(args): # Need additional path argument
    
    model = load_model(args['checkpoint_dir'])
    
    data = pd.read_csv(args['input_path'])
    cols = data.columns
    
    preds, rels = [], []
    for i in range(len(data)):
        args['smi1'] = data[cols[0]].iloc[i]
        args['smi2'] = data[cols[1]].iloc[i]
        args['molfrac1'] = data[cols[2]].iloc[i]
        args['T'] = data[cols[3]].iloc[i]

        prediction, reliability = visc_pred_onepoint(model, args)
        preds.append(prediction)
        rels.append(reliability)
        
    data['Viscoisty Predictions'] = preds
    data['Reliability'] = rels
    
    return preds, rels, data
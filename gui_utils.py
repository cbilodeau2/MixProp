# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:59:55 2022

@author: camil
"""

import numpy as np
# from rdkit import Chem
# from rdkit.Chem import AllChem
import os
# import pickle

from chemprop.train import predict
from chemprop.data import MoleculeDataset, MoleculeDataLoader, MoleculeDatapoint
# from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
# from chemprop.args import PredictArgs
# from chemprop.train.make_predictions import make_predictions

# import sys
# import matplotlib.pyplot as plt

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

    def __call__(self, smi1,smi2,molfrac1,T,threshold=0.022,n_models=None,num_workers=4):
        
        if n_models==None:
            n_models=len(self.checkpoints)
        assert n_models<=len(self.checkpoints),'Too many models requested. {} models requested.'.format(n_models)
        assert n_models>1, 'Multiple models are needed for reliability analysis.'
        
        model_input= MoleculeDatapoint(smiles=[smi1,smi2],features=[molfrac1,T])
        model_input = MoleculeDataset([model_input])
        model_input_loader = MoleculeDataLoader(dataset=model_input,num_workers=num_workers)

        all_model_preds = []
        for model in self.checkpoints[:n_models]:
             model_preds = predict(
                            model=model,
                            data_loader=model_input_loader,
                            scaler=self.scaler)
             all_model_preds.append(model_preds)
        
        avg_prediction = np.mean(all_model_preds)
        reliability = np.var(all_model_preds)<threshold
        return avg_prediction,reliability

def load_model(checkpoint_dir):
    return chemprop_model(checkpoint_dir)
    

def visc_pred_onepoint(smi1,smi2,molfrac1,T,
                      T_range = (293,323),
                      threshold = 0.022,
                      n_models=None,
                      num_workers=4,
                      checkpoint_dir='pretrained_models/nist_dippr_model/nist_dippr_model'):#'pretrained_models/nist_dippr_model'):
    
    try:
        n_models = int(n_models)
    except:
        print('Number of models needs to be a number')
    try:
        T = float(T)
    except:
        print('Temperature needs to be a number')
    
    try:
        molfrac1 = float(molfrac1)
    except:
        print('Mole fraction needs to be a number')
    
    model = load_model(checkpoint_dir)
    if (T<=T_range[0])|(T>=T_range[1]):
        print('Temperature is outside of recommended range')
    
    prediction,reliability = model(smi1,smi2,molfrac1,T,threshold=threshold,n_models=n_models,num_workers=num_workers)
    
    return 10**prediction,reliability #Prediction is in cP units

def visc_pred_onecurve(smi1,smi2,T,
                      molfrac1_range=np.arange(0,1,0.1),
                      T_range = (293,323),
                      threshold = 0.022,
                      n_models=None,
                      checkpoint_dir='pretrained_models/nist_dippr_model/nist_dippr_model'):

    try:
        n_models = int(n_models)
    except:
        print('Number of models needs to be a number')
    try:
        T = float(T)
    except:
        print('Temperature needs to be a number')
    
    model = load_model(checkpoint_dir)
    
    if (T<=T_range[0])|(T>=T_range[1]):
        print('Temperature is outside of recommended range')
    
    preds = []
    reliabilities = []
    for molfrac1 in molfrac1_range:
        pred,reliability = model(smi1,smi2,molfrac1,T,threshold=threshold,n_models=n_models)
        preds.append(pred)
        reliabilities.append(reliability)
    
    return preds,reliabilities

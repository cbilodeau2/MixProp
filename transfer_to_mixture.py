import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import sys
import os
from rdkit import Chem
from rdkit import DataStructs
from os import walk
import random
import argparse

from chemprop.models import MoleculeModel
from chemprop.utils import build_optimizer
from chemprop.utils import save_checkpoint, load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument('--mix_model_path', required=True)
parser.add_argument('--pure_model_path', required=True)
parser.add_argument('--out_model_path', required=True)
args = parser.parse_args()


# Read in args:
mix_model = torch.load(args.mix_model_path)
model_args =  mix_model['args']

# Read in pure model:
pure_model = torch.load(args.pure_model_path)

# Create new, randomly initialized model based on mixture model arguments:
new_model = MoleculeModel(model_args)

# Copy cached_zero_vector:
new_model.state_dict()['encoder.encoder.0.cached_zero_vector']=pure_model['state_dict']['encoder.encoder.0.cached_zero_vector']
new_model.state_dict()['encoder.encoder.1.cached_zero_vector']=pure_model['state_dict']['encoder.encoder.0.cached_zero_vector']


# Copy W_i weight:
new_model.state_dict()['encoder.encoder.0.W_i.weight']=pure_model['state_dict']['encoder.encoder.0.W_i.weight']
new_model.state_dict()['encoder.encoder.1.W_i.weight']=pure_model['state_dict']['encoder.encoder.0.W_i.weight']

# Coppy W_h weight:
new_model.state_dict()['encoder.encoder.0.W_h.weight']=pure_model['state_dict']['encoder.encoder.0.W_h.weight']
new_model.state_dict()['encoder.encoder.1.W_h.weight']=pure_model['state_dict']['encoder.encoder.0.W_h.weight']

# Copy W_o weight:
new_model.state_dict()['encoder.encoder.0.W_o.weight']=pure_model['state_dict']['encoder.encoder.0.W_o.weight']
new_model.state_dict()['encoder.encoder.1.W_o.weight']=pure_model['state_dict']['encoder.encoder.0.W_o.weight']

# Copy W_o bias:
new_model.state_dict()['encoder.encoder.0.W_o.bias']=pure_model['state_dict']['encoder.encoder.0.W_o.bias']
new_model.state_dict()['encoder.encoder.1.W_o.bias']=pure_model['state_dict']['encoder.encoder.0.W_o.bias']

state = {
    'args': model_args,
    'state_dict': new_model.state_dict()}

torch.save(state, args.out_model_path)

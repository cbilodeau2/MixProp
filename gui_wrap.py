# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 15:59:22 2022

@author: camil
"""

from gui_utils import visc_pred_onepoint
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--smi1',required=True)
parser.add_argument('--smi2',required=True)
parser.add_argument('--molfrac1', type=float,required=True) #Without an args file, many parameters will revert to default
parser.add_argument('--T', type=float, default=298)
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--n_models', type=str, default=2)
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()
print(args.smi1,args.smi2,type(args.molfrac1))

pred, rel = visc_pred_onepoint(args.smi1,
                                args.smi2,
                                args.molfrac1,
                                args.T,
                                args.checkpoint_dir,
                                n_models=args.n_models,
                                num_workers=args.num_workers)

print('Viscosity Prediction:{}'.format(pred))
print('Reliability:{}'.format(rel))

# input_syntax = [
#     'python gui_wrap.py',
#     '--smi1',
#     '--smi2',
#     '--molfrac1',
#     '--T',
#     'n_models'
#     ]
#!/bin/bash

export CUDA_VISIBLE_DEVICES=2


name=sol_new
data_file=data/DowData/solubility/Joback_OneMol/data.csv
checkpoint_folder=checkpoints/sol_new
#features_file=data/DowData/solubility/${feat_name}_${set_name}/features.csv


data_type=regression
split_type=random #random 

n_epochs=100
dropout=0
num_iters=20
num_folds=10
ensemble_size=1

timestamp=`date +"%s"`
checkpoint_subfolder=$checkpoint_folder/${name}-${timestamp}
mkdir $checkpoint_folder
mkdir $checkpoint_subfolder
config=$checkpoint_subfolder/config.json

python train.py --data_path $data_file --dataset_type $data_type --number_of_molecules 1 --save_dir $checkpoint_subfolder --split_type $split_type --save_smiles_splits --epochs $n_epochs --dropout $dropout --smiles_column SMILES

#python train.py --data_path $data_file --dataset_type $data_type --number_of_molecules 2 --save_dir $checkpoint_subfolder --split_type $split_type --save_smiles_splits --epochs $n_epochs --dropout $dropout --smiles_column mol1 mol2 --mpn_shared --fractions --split_sizes 0.85 0.1 0.05 --hidden_size 200 --ffn_num_layers 4 --depth 4 --batch_size 50 --checkpoint_path test_model.pt

#--features_path $features_file




#python hyperparameter_optimization.py --data_path $data_file --dataset_type $data_type --save_dir $checkpoint_subfolder --split_type $split_type --num_iters $num_iters --num_folds $num_folds --ensemble_size $ensemble_size --save_smiles_splits --config_save_path $config #--features_path $features_file



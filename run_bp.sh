#!/bin/bash

export CUDA_VISIBLE_DEVICES=1


name=bp_nofeat_100epochs
data_file=data/DowData/boiling_point/Physprop_BP_Data.csv
checkpoint_folder=checkpoints/bp_nofeat
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

python train.py --data_path $data_file --dataset_type $data_type --number_of_molecules 1 --save_dir $checkpoint_subfolder --split_type $split_type --epochs $n_epochs --dropout $dropout --hidden_size 200 --ffn_num_layers 4 --depth 4 --batch_size 50 --save_smiles_splits --smiles_columns SMILES
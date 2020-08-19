#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

name=sol_hyperopt
data_file=data/DowData/solubility/Joback_OneMol/data.csv #data/MolNet/tox21.csv #data/MolNet/tox21.csv #qm7.csv #data/DowData/solubility/Joback_OneMol/data.csv
checkpoint_folder=checkpoints/NoFeat_OneMol
features_file=data/DowData/solubility/Joback_OneMol/features.csv
config=$checkpoint_subfolder/config.json
#pretraining_model_path=None

data_type=regression
split_type=scaffold_balanced

n_epochs=5
dropout=0
num_iters=10
num_folds=1
ensemble_size=1

timestamp=`date +"%s"`
checkpoint_subfolder=$checkpoint_folder/${name}-${timestamp}
mkdir $checkpoint_folder
mkdir $checkpoint_subfolder

#echo "python train.py --data_path $data_file --dataset_type $data_type --save_dir $checkpoint_subfolder"
#python train.py --data_path $data_file --dataset_type $data_type --save_dir $checkpoint_subfolder #--split_type $split_type --save_smiles_splits

python hyperparameter_optimization.py --data_path $data_file --dataset_type $data_type --save_dir $checkpoint_subfolder --split_type $split_type --num_iters $num_iters --num_folds $num_folds --ensemble_size $ensemble_size --save_smiles_splits --config_save_path $checkpoint_subfolder #--features_path $features_file



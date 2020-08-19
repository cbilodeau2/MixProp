#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

#MAKE SURE FEATURES PATH IS COMMENTED CORRECTLY

# feat_name=Hansen #THINGS HAVE BEEN HCANGED
# set_name=G3_G5

# name=sol_hyperopt
# data_file=data/DowData/solubility/${feat_name}_${set_name}/data.csv 
# checkpoint_folder=checkpoints/${feat_name}_${set_name}_random
# features_file=data/DowData/solubility/${feat_name}_${set_name}/features.csv



name=tox21
data_file=data/MolNet/tox21.csv
checkpoint_folder=checkpoints/tox21
#features_file=data/DowData/solubility/${feat_name}_${set_name}/features.csv


#pretraining_model_path=None

data_type=classification # regression
split_type=scaffold_balanced #random #scaffold_balanced

n_epochs=30
dropout=0
num_iters=20
num_folds=10
ensemble_size=1

timestamp=`date +"%s"`
checkpoint_subfolder=$checkpoint_folder/${name}-${timestamp}
mkdir $checkpoint_folder
mkdir $checkpoint_subfolder
config=$checkpoint_subfolder/config.json

#echo "python train.py --data_path $data_file --dataset_type $data_type --save_dir $checkpoint_subfolder"
#python train.py --data_path $data_file --dataset_type $data_type --save_dir $checkpoint_subfolder --split_type $split_type --save_smiles_splits --epochs $n_epochs --dropout $dropout --features_path $features_file

python hyperparameter_optimization.py --data_path $data_file --dataset_type $data_type --save_dir $checkpoint_subfolder --split_type $split_type --num_iters $num_iters --num_folds $num_folds --ensemble_size $ensemble_size --save_smiles_splits --config_save_path $config #--features_path $features_file



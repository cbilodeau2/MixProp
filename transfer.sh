#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

mix=checkpoints/bubble_florence_split/bubble_florence_split-1601405994/fold_0/model_0/model.pt
pure=checkpoints/bp_nofeat/bp_nofeat-1601480380/fold_0/model_0/model.pt
out=test_model.pt

python transfer_to_mixture.py --mix_model_path $mix --pure_model_path $pure --out_model_path $out
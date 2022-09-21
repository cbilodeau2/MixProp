# Property Prediction for the Binary Mixtures

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/chemprop)](https://badge.fury.io/py/chemprop)
[![PyPI version](https://badge.fury.io/py/chemprop.svg)](https://badge.fury.io/py/chemprop)
[![Build Status](https://github.com/chemprop/chemprop/workflows/tests/badge.svg)](https://github.com/chemprop/chemprop)

**Warning: MixProp is currently configured for use with CUDA. If a GPU is unavailable, the user will receive an error message.**


This repository contains a modified directed message passing neural network (D-MPNN) for the prediction of binary mixtures. It is based off of the standard D-MPNN described in [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237).

**ChemProp Standard:** The original version of Chemprop for general property prediction is available here: https://github.com/chemprop/chemprop


**Documentation:** Full documentation of Chemprop Standard is available at https://chemprop.readthedocs.io/en/latest/.

## Train the Model:
To re-train the model on the training set found in ``pretrained_models/nist_dippr_data`` and store the results in a folder ``checkpoints`` , execute the following command:

```
mkdir checkpoints
python train.py --data_path pretrained_models/nist_dippr_data/data.csv --features_path pretrained_models/nist_dippr_data/data_features.csv --no_features_scaling --dataset_type regression --save_dir checkpoints --split_type random --epochs 500 --aggregation norm --gpu 0 --num_folds 25 --ensemble_size 1 --number_of_molecules 2 --mpn_shared
```
The test/train split shared in ``pretrained_models/nist_dippr_data`` was one of the three splits reported in the original paper, such that, executing the above command will reproduce the model performance reported in the paper.

## Request the Pre-trained Models:
Pre-trained models are available on request: cur5wz@virginia.edu.

## Use Pre-trained Models with Graphical User Interface:
A simple graphical user interface is available for single point predictions and can be opened using:
```
python gui.py
```
By default, the pre-trained model files are expected to be in ``pretrained_models/nist_dippr_model/nist_dippr_model``. If they are located somewhere else, the variable ``checkpoint_dir`` in ``gui.py`` should be set to the path of the pre-trained model files.

## Use Pre-trained Models to Predict Test Data:
To use the pretrained models to make predictions, below is the command for predicting viscosity for the test set used to train this model (found in ``pretrained_models/nist_dippr_data/test.csv`` and ``pretrained_models/nist_dippr_data/test_features.csv``:

```
python predict.py --test_path pretrained_models/nist_dippr_data/test.csv --features_path pretrained_models/nist_dippr_data/test_features.csv --checkpoint_dir pretrained_models/nist_dippr_model --preds_path test_preds.csv --number_of_molecules 2 --no_features_scaling
```
The test/train split shared in ``pretrained_models/nist_dippr_data`` was one of the three splits reported in the original paper, such that, executing the above command will reproduce the model performance reported in the paper.

**Uncertainty Quantification:** 

To calculate the uncertainty of a given prediction using the ensemble variance approach, add the ``--uncertainty_method ensemble`` flag:
```
python predict.py --test_path pretrained_models/nist_dippr_data/test.csv --features_path pretrained_models/nist_dippr_data/test_features.csv --checkpoint_dir pretrained_models/nist_dippr_model --preds_path test_preds.csv --number_of_molecules 2 --no_features_scaling --uncertainty_method ensemble
```






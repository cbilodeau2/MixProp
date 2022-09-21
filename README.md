# Property Prediction for the Binary Mixtures
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/chemprop)](https://badge.fury.io/py/chemprop)
[![PyPI version](https://badge.fury.io/py/chemprop.svg)](https://badge.fury.io/py/chemprop)
[![Build Status](https://github.com/chemprop/chemprop/workflows/tests/badge.svg)](https://github.com/chemprop/chemprop)

This repository contains message passing neural networks for molecular property prediction as described in the paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) and as used in the paper [A Deep Learning Approach to Antibiotic Discovery](https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1) for molecules and [Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction](https://doi.org/10.1021/acs.jcim.1c00975) for reactions.

**ChemProp Standard:** The original version of Chemprop for general property prediction is available here: https://github.com/chemprop/chemprop


**Documentation:** Full documentation of Chemprop Standard is available at https://chemprop.readthedocs.io/en/latest/.


## Use Pre-trained Models to Predict Test Data:
Pretrained models can be found in ``pretrained_models/nist_dippr_models`` and these can be used to make predictions for new datasets. To illustrate this, below is the command for predicting viscosity for the test set used to train this model (found in ``pretrained_models/nist_dippr_data/test.csv`` and ``pretrained_models/nist_dippr_data/test_features.csv``:

```
python predict.py --test_path pretrained_models/nist_dippr_data/test.csv --features_path pretrained_models/nist_dippr_data/test_features.csv --checkpoint_dir pretrained_models/nist_dippr_model --preds_path test_preds.csv --number_of_molecules 2 --no_features_scaling --dataset_type regression
```
The test/train split shared in ``pretrained_models/nist_dippr_data`` was one of the three splits reported in the original paper, such that, executing the above command will reproduce the model performance reported in the paper.

**Uncertainty Quantification:** 

To calculate the uncertainty of a given prediction using the ensemble variance approach, add the ``--uncertainty_method ensemble`` flag:
```
python predict.py --test_path pretrained_models/nist_dippr_data/test.csv --features_path pretrained_models/nist_dippr_data/test_features.csv --checkpoint_dir pretrained_models/nist_dippr_model --preds_path test_preds.csv --number_of_molecules 2 --no_features_scaling --dataset_type regression --uncertainty_method ensemble
```

## Train the Model:
To re-train the model on the training set found in ``pretrained_models/nist_dippr_data`` and store the results in a folder ``checkpoints`` , execute the following command:

```
mkdir checkpoints
python train.py --data_path pretrained_models/nist_dippr_data/data.csv --features_path pretrained_models/nist_dippr_data/data_features.csv --no_features_scaling --dataset_type regression --save_dir checkpoints --split_type random --epochs 500 --aggregation norm --gpu 0 --num_folds 25 --ensemble_size 1 --number_of_molecules 2 --mpn_shared
```
The test/train split shared in ``pretrained_models/nist_dippr_data`` was one of the three splits reported in the original paper, such that, executing the above command will reproduce the model performance reported in the paper.





import os, random, math
import copy
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

import rdkit
from rdkit.Chem import Descriptors
from rdkit import Chem

from SRGM import SRGM

from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.models import MoleculeModel
from chemprop.data import MoleculeDataset, MoleculeDatapoint, MoleculeDataLoader, StandardScaler, get_data, get_task_names, split_data
from chemprop.data.scaffold import scaffold_to_smiles, scaffold_split
from chemprop.nn_utils import get_activation_function, initialize_weights, compute_gnorm
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, makedirs, save_checkpoint
from chemprop.train import evaluate, evaluate_predictions, predict

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)


class SRGMArgs(TrainArgs):
    rgm_p : float = 0.1
    rgm_e : float = 0.1
    rgm_env_lr : float = 0.1
    multiclass : bool = False
    linear : bool = False
    scaffold : bool = True
    num_domains : int = 2
    num_groups : int = 2


def generate_scaffolds(train_data, args):
    scaffold_to_indices = scaffold_to_smiles(train_data.mols(flatten=True), use_indices=True)
    scaffolds = sorted(scaffold_to_indices.keys())
    for sid, scaf in enumerate(scaffolds):
        indices = scaffold_to_indices[scaf]
        smol = Chem.MolFromSmiles(scaf)
        if not smol: 
            scaf = train_data[i].smiles  # in case scaffold is not a valid molecule
        for i in indices:
            train_data[i].scaf = train_data[i].smiles if len(scaf) == 0 else [scaf]


def prepare_data(args):
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns, target_columns=args.target_columns, ignore_columns=args.ignore_columns)

    data = get_data(path=args.data_path, args=args, skip_none_targets=True)

    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.0, 0.2), seed=args.seed, num_folds=args.num_folds, args=args)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0), seed=args.seed, num_folds=args.num_folds, args=args)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, num_folds=args.num_folds, args=args)

    generate_scaffolds(train_data, args)
    train_holdout, _, _ = scaffold_split(data=train_data, sizes=(0.5,0.5,0.0), balanced=False, seed=args.seed)
    for d in train_data:
        d.holdout = False
    for d in train_holdout:
        d.holdout = True

    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        scaler = train_data.normalize_targets()
    else:
        scaler = None

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.output_size = train_data.num_tasks()
    args.features_size = train_data.features_size()
    args.train_data_size = len([d for d in train_data if not d.holdout])
    
    print('train, val, test:', len(train_data), len(val_data), len(test_data))
    return train_data, val_data, test_data, scaler, features_scaler


def prepare_trainer(args):
    loss_func = get_loss_func(args)
    featurizer = MoleculeModel(args)
    featurizer.create_encoder(args)
    featurizer.ffn = nn.Identity()

    trainer = SRGM(featurizer, loss_func, args)
    trainer = trainer.cuda()
    initialize_weights(trainer)

    trainer.network = MoleculeModel(args)
    trainer.network.encoder = featurizer.encoder
    trainer.network.ffn = trainer.classifier

    print(trainer.network)

    trainer.optimizer = build_optimizer(trainer, args)
    trainer.scheduler = build_lr_scheduler(trainer.optimizer, args)
    return trainer


def train(trainer, data, args):
    trainer.train()

    holdout_data = [d for d in data if d.holdout]
    data = [d for d in data if not d.holdout]  # important!
    random.shuffle(holdout_data)
    random.shuffle(data)
    print('data:', len(data), 'holdout:', len(holdout_data))

    assert len(holdout_data) > 0
    while len(holdout_data) < len(data):
        holdout_data += holdout_data

    for i in range(0, len(data), args.batch_size):
        mol_batch = data[i:i + args.batch_size]
        mol_batch = MoleculeDataset(mol_batch)
        if len(mol_batch) < args.batch_size:
            continue

        holdout_batch = holdout_data[i:i + args.batch_size]
        holdout_batch = MoleculeDataset(holdout_batch)

        smiles_batch, features_batch, target_batch = mol_batch.batch_graph(), mol_batch.features(), mol_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]).cuda()
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).cuda()
        envs = MoleculeDataset([MoleculeDatapoint(d.scaf) for d in mol_batch]).batch_graph()

        holdout_smiles, holdout_features, holdout_targets = holdout_batch.batch_graph(), holdout_batch.features(), holdout_batch.targets()
        holdout_mask = torch.Tensor([[x is not None for x in tb] for tb in holdout_targets]).cuda()
        holdout_targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in holdout_targets]).cuda()
        holdout_envs = MoleculeDataset([MoleculeDatapoint(d.scaf) for d in holdout_batch]).batch_graph()

        batches = [(smiles_batch, features_batch, targets, envs, mask), (holdout_smiles, holdout_features, holdout_targets, holdout_envs, holdout_mask)]

        trainer.optimizer.zero_grad()
        loss = trainer(batches)
        loss.backward()
        nn.utils.clip_grad_norm_(trainer.parameters(), 5.0)
        gnorm = compute_gnorm(trainer)
        trainer.optimizer.step()
        trainer.scheduler.step()

        lr = trainer.scheduler.get_lr()[0]
        gnorm = compute_gnorm(trainer)
        print(f'lr: {lr:.5f}, loss: {loss:.4f}, gnorm: {gnorm:.4f}')


def run_training(args, save_dir):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_data, val_data, test_data, scaler, features_scaler = prepare_data(args)
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=0
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=0
    )

    trainer = prepare_trainer(args)

    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch = 0
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        train(trainer, train_data, args)

        val_scores = evaluate(trainer.network, val_data_loader, args.output_size, args.metrics, args.dataset_type, scaler=scaler)
        avg_val_score = np.nanmean(val_scores[args.metric])
        print(f'Validation {args.metric} = {avg_val_score:.4f}')

        if args.minimize_score and avg_val_score < best_score or not args.minimize_score and avg_val_score > best_score:
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), trainer.network, scaler, features_scaler, args)
    
    print(f'Loading model checkpoint from epoch {best_epoch}')
    model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device)
    test_scores = evaluate(trainer.network, test_data_loader, args.output_size, args.metrics, args.dataset_type, scaler=scaler)

    avg_test_score = np.nanmean(test_scores[args.metric])
    print(f'Test {args.metric} = {avg_test_score:.4f}')
    return avg_test_score


if __name__ == "__main__":
    parser = SRGMArgs()
    args = parser.parse_args()
    args.classification = (args.dataset_type == 'classification')
    args.batch_size = args.batch_size // args.num_domains
    print(args)

    all_test_score = np.zeros((args.num_folds,))
    for i in range(args.num_folds):
        fold_dir = os.path.join(args.save_dir, f'fold_{i}')
        makedirs(fold_dir)
        args.seed = i
        all_test_score[i] = run_training(args, fold_dir)

    mean, std = np.mean(all_test_score), np.std(all_test_score)
    print(f'{args.num_folds} fold average: {mean:.4f} +/- {std:.4f}')


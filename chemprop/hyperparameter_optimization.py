"""Optimizes hyperparameters using Bayesian optimization."""

from copy import deepcopy
import json
from typing import Dict, Union
import os
from collections import defaultdict
import csv

from hyperopt import fmin, hp, tpe
import numpy as np
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch

from chemprop.args import HyperoptArgs
from chemprop.constants import HYPEROPT_LOGGER_NAME, TEST_SCORES_FILE_NAME
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count
from chemprop.train import cross_validate, run_training, predict, evaluate_predictions, evaluate
from chemprop.utils import create_logger, makedirs, timeit, save_smiles_splits, load_scalers, load_checkpoint
from chemprop.data import split_data, get_data, MoleculeDataLoader, set_cache_graph


hyperopt_space = {
    'hidden_size': hp.quniform('hidden_size', low=300, high=2400, q=100),
    # 'depth': hp.quniform('depth', low=2, high=6, q=1),
    'dropout': hp.uniform('dropout', low=0.0, high=0.4),
    # 'ffn_num_layers': hp.quniform('ffn_num_layers', low=1, high=3, q=1),
    'ffn_hidden_size': hp.quniform('ffn_hidden_size', low=300, high=2400, q=100),
    'max_lr': hp.quniform('max_lr', low = 1e-4, high = 1.5e-3, q = 1e-4),
}

raytune_space = {
    'hidden_size': tune.quniform(300,2400,100),
    # 'depth': hp.quniform('depth', low=2, high=6, q=1),
    'dropout': tune.uniform(0,0.4),
    # 'ffn_num_layers': hp.quniform('ffn_num_layers', low=1, high=3, q=1),
    'ffn_hidden_size': tune.quniform(300,2400,100),
    'max_lr': tune.loguniform(1e-3,1e-4),
}

INT_KEYS = ['hidden_size','ffn_hidden_size']#, 'depth', 'ffn_num_layers']


@timeit(logger_name=HYPEROPT_LOGGER_NAME)
def hyperopt(args: HyperoptArgs) -> None:
    """
    Runs hyperparameter optimization on a Chemprop model.

    Hyperparameter optimization optimizes the following parameters:

    * :code:`hidden_size`: The hidden size of the neural network layers is selected from {300, 400, ..., 2400}
    * :code:`depth`: The number of message passing iterations is selected from {2, 3, 4, 5, 6}
    * :code:`dropout`: The dropout probability is selected from {0.0, 0.05, ..., 0.4}
    * :code:`ffn_num_layers`: The number of feed-forward layers after message passing is selected from {1, 2, 3}

    The best set of hyperparameters is saved as a JSON file to :code:`args.config_save_path`.

    :param args: A :class:`~chemprop.args.HyperoptArgs` object containing arguments for hyperparameter
                 optimization in addition to all arguments needed for training.
    """
    # Create logger
    if args.log_dir is None:
        args.log_dir=args.save_dir
    logger = create_logger(name=HYPEROPT_LOGGER_NAME, save_dir=args.log_dir, quiet=True)

    # Run grid search
    results = []

    data = get_data(path=args.data_path,features_path=args.features_path)

    if args.num_folds ==1:
        trainval_data,val2_data,test_data = split_data(data=data,seed=args.seed,sizes=(0.81,0.09,0.1))

        train_data,val1_data, _ = split_data(data=trainval_data,seed=args.seed,sizes=(0.89,0.11,0))
        val1_data=val2_data

        save_smiles_splits(data_path=args.data_path,test_data=val2_data,train_data=train_data,val_data=val1_data,save_dir=args.save_dir)

        os.rename(os.path.join(args.save_dir,'test_smiles.csv'),os.path.join(args.save_dir, 'val2_smiles.csv'))
        os.rename(os.path.join(args.save_dir,'test_full.csv'),os.path.join(args.save_dir, 'val2_full.csv'))
        os.rename(os.path.join(args.save_dir,'val_smiles.csv'),os.path.join(args.save_dir, 'val1_smiles.csv'))
        os.rename(os.path.join(args.save_dir,'val_full.csv'),os.path.join(args.save_dir, 'val1_full.csv'))

        save_smiles_splits(data_path=args.data_path,test_data=test_data,train_data=train_data,val_data=val1_data,save_dir=args.save_dir)

        os.remove(os.path.join(args.save_dir,'val_smiles.csv'))
        os.remove(os.path.join(args.save_dir,'val_full.csv'))
        os.remove(os.path.join(args.save_dir,'split_indices.pckl'))

        args.data_path = os.path.join(args.save_dir,'train_full.csv')
        args.separate_val_path = os.path.join(args.save_dir,'val1_full.csv')
        args.separate_test_path = os.path.join(args.save_dir,'val2_full.csv')
    else: #for crossvalidation, where run_training will further split trainval
        trainval_data,val2_data,test_data = split_data(data=data,seed=args.seed,sizes=(0.81,0.09,0.1))

        save_smiles_splits(data_path=args.data_path,test_data=test_data,train_data=trainval_data,val_data=val2_data,save_dir=args.save_dir)

        os.rename(os.path.join(args.save_dir,'train_smiles.csv'),os.path.join(args.save_dir, 'trainval_smiles.csv'))
        os.rename(os.path.join(args.save_dir,'train_full.csv'),os.path.join(args.save_dir, 'trainval_full.csv'))
        os.rename(os.path.join(args.save_dir,'val_smiles.csv'),os.path.join(args.save_dir, 'val2_smiles.csv'))
        os.rename(os.path.join(args.save_dir,'val_full.csv'),os.path.join(args.save_dir, 'val2_full.csv'))

        os.remove(os.path.join(args.save_dir,'split_indices.pckl'))

        args.data_path = os.path.join(args.save_dir,'trainval_full.csv')
        args.separate_test_path = os.path.join(args.save_dir,'val2_full.csv')


    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    test_targets=test_data.targets()
    test_smiles=test_data.smiles()
    num_tasks=len(test_targets[0])

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Copy args
        hyper_args = deepcopy(args)

        # Update args with hyperparams
        if args.save_dir is not None:
            folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items())
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)

        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)

        hyper_args.ffn_hidden_size = hyper_args.hidden_size

        # Record hyperparameters
        logger.info(hyperparams)

        # Cross validate
        val2_mean_score, val2_std_score = cross_validate(args=hyper_args, train_func=run_training)

        ensemble_scores=[]

        all_scores=defaultdict(list)
        for fold_num in range(args.num_folds):

            if args.dataset_type == 'multiclass':
                sum_test_preds = np.zeros((len(test_smiles), num_tasks, args.multiclass_num_classes))
            else:
                # print('length_smiles',len(test_smiles))
                # print('num_tasks',num_tasks)
                sum_test_preds = np.zeros((len(test_smiles), num_tasks))

            for model_idx in range(args.ensemble_size):

                checkpoint_path=os.path.join(args.save_dir,folder_name,'fold_'+str(fold_num),'model_'+str(model_idx),'model.pt')

                scaler, _ = load_scalers(checkpoint_path)

                model = load_checkpoint(checkpoint_path)

                test_preds = predict(
                    model=model,
                    data_loader=test_data_loader,
                    scaler=scaler
                )
                test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=test_targets,
                    num_tasks=num_tasks,
                    metrics=args.metrics,
                    dataset_type=args.dataset_type,
                    logger=logger
                )

                if len(test_preds) != 0:
                    sum_test_preds += np.array(test_preds)

                # Average test score
                for tgt, scores in test_scores.items():
                    avg_test_score = np.nanmean(scores)

            # Evaluate ensemble on test set
            avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

            ensemble_scores = evaluate_predictions(
                preds=avg_test_preds,
                targets=test_targets,
                num_tasks=num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                logger=logger
            )
            for tgt, scores in ensemble_scores.items():
                all_scores[tgt].append(scores)
        # print(all_scores.keys())
        all_scores=dict(all_scores)
        # print(all_scores.keys())

        for tgt, scores in all_scores.items():
            all_scores[tgt] = np.array(scores)
        # print(all_scores.keys())
        avg_scores = np.nanmean(all_scores[args.metric], axis=1)
        mean_score = np.nanmean(avg_scores)
        std_score = np.nanstd(avg_scores)

        # Record results
        temp_model = MoleculeModel(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f'num params: {num_params:,}')
        logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')

        results.append({
            'test_mean_score': mean_score,
            'test_std_score': std_score,
            'hyperparams': hyperparams,
            'num_params': num_params,
            'val2_mean_score': val2_mean_score,
            'val2_std_score': val2_std_score
        })

        # Deal with nan
        if np.isnan(mean_score):
            if hyper_args.dataset_type == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')

        return (1 if hyper_args.minimize_score else -1) * val2_mean_score

    fmin(objective, hyperopt_space, algo=tpe.suggest, max_evals=args.num_iters, rstate=np.random.RandomState(args.seed))

    # Report best result
    results = [result for result in results if not np.isnan(result['test_mean_score'])]
    for result in results:
        logger.info(result)
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['val2_mean_score'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(f'val2 score {best_result["val2_mean_score"]} +/- {best_result["val2_std_score"]} {args.metric}')
    logger.info(f'test score {best_result["test_mean_score"]} +/- {best_result["test_std_score"]} {args.metric}')

    with open(os.path.join(args.log_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)

        non_param_header=list(results[0].keys())
        non_param_header.remove('hyperparams')
        param_header=list(results[0]['hyperparams'].keys())

        writer.writerow(non_param_header+param_header)
        for result in results:
            writer.writerow([result[i] for i in non_param_header]+[result['hyperparams'][i] for i in param_header])

    # Save best hyperparameter settings as JSON config file
    makedirs(args.config_save_path, isfile=True)

    with open(args.config_save_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


def chemprop_hyperopt() -> None:
    """Runs hyperparameter optimization for a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_hyperopt`.
    """
    hyperopt(args=HyperoptArgs().parse_args())

@timeit(logger_name=HYPEROPT_LOGGER_NAME)
def raytune(args: HyperoptArgs) -> None:
    """
    Runs hyperparameter optimization on a Chemprop model.

    Hyperparameter optimization optimizes the following parameters:

    * :code:`hidden_size`: The hidden size of the neural network layers is selected from {300, 400, ..., 2400}
    * :code:`depth`: The number of message passing iterations is selected from {2, 3, 4, 5, 6}
    * :code:`dropout`: The dropout probability is selected from {0.0, 0.05, ..., 0.4}
    * :code:`ffn_num_layers`: The number of feed-forward layers after message passing is selected from {1, 2, 3}

    The best set of hyperparameters is saved as a JSON file to :code:`args.config_save_path`.

    :param args: A :class:`~chemprop.args.HyperoptArgs` object containing arguments for hyperparameter
                 optimization in addition to all arguments needed for training.
    """
    # Create logger
    if args.log_dir is None:
        args.log_dir=args.save_dir
    logger = create_logger(name=HYPEROPT_LOGGER_NAME, save_dir=args.log_dir, quiet=True)

    # Run grid search
    results = []

    data = get_data(path=args.data_path,features_path=args.features_path)

    if args.num_folds ==1:
        trainval_data,val2_data,test_data = split_data(data=data,seed=args.seed,sizes=(0.81,0.09,0.1))

        train_data,val1_data, _ = split_data(data=trainval_data,seed=args.seed,sizes=(0.89,0.11,0))
        val1_data=val2_data

        save_smiles_splits(data_path=args.data_path,test_data=val2_data,train_data=train_data,val_data=val1_data,save_dir=args.save_dir)

        os.rename(os.path.join(args.save_dir,'test_smiles.csv'),os.path.join(args.save_dir, 'val2_smiles.csv'))
        os.rename(os.path.join(args.save_dir,'test_full.csv'),os.path.join(args.save_dir, 'val2_full.csv'))
        os.rename(os.path.join(args.save_dir,'val_smiles.csv'),os.path.join(args.save_dir, 'val1_smiles.csv'))
        os.rename(os.path.join(args.save_dir,'val_full.csv'),os.path.join(args.save_dir, 'val1_full.csv'))

        save_smiles_splits(data_path=args.data_path,test_data=test_data,train_data=train_data,val_data=val1_data,save_dir=args.save_dir)

        os.remove(os.path.join(args.save_dir,'val_smiles.csv'))
        os.remove(os.path.join(args.save_dir,'val_full.csv'))
        os.remove(os.path.join(args.save_dir,'split_indices.pckl'))

        args.data_path = os.path.join(args.save_dir,'train_full.csv')
        args.separate_val_path = os.path.join(args.save_dir,'val1_full.csv')
        args.separate_test_path = os.path.join(args.save_dir,'val2_full.csv')
    else: #for crossvalidation, where run_training will further split trainval
        trainval_data,val2_data,test_data = split_data(data=data,seed=args.seed,sizes=(0.81,0.09,0.1))

        save_smiles_splits(data_path=args.data_path,test_data=test_data,train_data=trainval_data,val_data=val2_data,save_dir=args.save_dir)

        os.rename(os.path.join(args.save_dir,'train_smiles.csv'),os.path.join(args.save_dir, 'trainval_smiles.csv'))
        os.rename(os.path.join(args.save_dir,'train_full.csv'),os.path.join(args.save_dir, 'trainval_full.csv'))
        os.rename(os.path.join(args.save_dir,'val_smiles.csv'),os.path.join(args.save_dir, 'val2_smiles.csv'))
        os.rename(os.path.join(args.save_dir,'val_full.csv'),os.path.join(args.save_dir, 'val2_full.csv'))

        os.remove(os.path.join(args.save_dir,'split_indices.pckl'))

        args.data_path = os.path.join(args.save_dir,'trainval_full.csv')
        args.separate_test_path = os.path.join(args.save_dir,'val2_full.csv')


    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    test_targets=test_data.targets()
    test_smiles=test_data.smiles()
    num_tasks=len(test_targets[0])

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Copy args
        hyper_args = deepcopy(args)

        # Update args with hyperparams
        if args.save_dir is not None:
            folder_name = '_'.join(f'{key}_{value}' for key, value in hyperparams.items())
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)

        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)

        hyper_args.ffn_hidden_size = hyper_args.hidden_size

        # Record hyperparameters
        logger.info(hyperparams)

        # Cross validate
        val2_mean_score, val2_std_score = cross_validate(args=hyper_args, train_func=run_training)

        ensemble_scores=[]

        all_scores=defaultdict(list)
        for fold_num in range(args.num_folds):

            if args.dataset_type == 'multiclass':
                sum_test_preds = np.zeros((len(test_smiles), num_tasks, args.multiclass_num_classes))
            else:
                # print('length_smiles',len(test_smiles))
                # print('num_tasks',num_tasks)
                sum_test_preds = np.zeros((len(test_smiles), num_tasks))

            for model_idx in range(args.ensemble_size):

                checkpoint_path=os.path.join(args.save_dir,folder_name,'fold_'+str(fold_num),'model_'+str(model_idx),'model.pt')

                scaler, _ = load_scalers(checkpoint_path)

                model = load_checkpoint(checkpoint_path)

                test_preds = predict(
                    model=model,
                    data_loader=test_data_loader,
                    scaler=scaler
                )
                test_scores = evaluate_predictions(
                    preds=test_preds,
                    targets=test_targets,
                    num_tasks=num_tasks,
                    metrics=args.metrics,
                    dataset_type=args.dataset_type,
                    logger=logger
                )

                if len(test_preds) != 0:
                    sum_test_preds += np.array(test_preds)

                # Average test score
                for tgt, scores in test_scores.items():
                    avg_test_score = np.nanmean(scores)

            # Evaluate ensemble on test set
            avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

            ensemble_scores = evaluate_predictions(
                preds=avg_test_preds,
                targets=test_targets,
                num_tasks=num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                logger=logger
            )
            for tgt, scores in ensemble_scores.items():
                all_scores[tgt].append(scores)
        # print(all_scores.keys())
        all_scores=dict(all_scores)
        # print(all_scores.keys())

        for tgt, scores in all_scores.items():
            all_scores[tgt] = np.array(scores)
        # print(all_scores.keys())
        avg_scores = np.nanmean(all_scores[args.metric], axis=1)
        mean_score = np.nanmean(avg_scores)
        std_score = np.nanstd(avg_scores)

        # Record results
        temp_model = MoleculeModel(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f'num params: {num_params:,}')
        logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')

        results.append({
            'test_mean_score': mean_score,
            'test_std_score': std_score,
            'hyperparams': hyperparams,
            'num_params': num_params,
            'val2_mean_score': val2_mean_score,
            'val2_std_score': val2_std_score
        })

        # Deal with nan
        if np.isnan(mean_score):
            if hyper_args.dataset_type == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')

        tune.report(loss=(1 if hyper_args.minimize_score else -1) * val2_mean_score)
        return (1 if hyper_args.minimize_score else -1) * val2_mean_score

    ray.init()
    algo=HyperOptSearch(metric="loss",mode='min')
    ray_opt=tune.run(objective, config=raytune_space, search_alg=algo, num_samples=args.num_iters)

    # Report best result
    results = [result for result in results if not np.isnan(result['test_mean_score'])]
    for result in results:
        logger.info(result)
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['val2_mean_score'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(f'val2 score {best_result["val2_mean_score"]} +/- {best_result["val2_std_score"]} {args.metric}')
    logger.info(f'test score {best_result["test_mean_score"]} +/- {best_result["test_std_score"]} {args.metric}')

    with open(os.path.join(args.log_dir, TEST_SCORES_FILE_NAME), 'w') as f:
        writer = csv.writer(f)

        non_param_header=list(results[0].keys())
        non_param_header.remove('hyperparams')
        param_header=list(results[0]['hyperparams'].keys())

        writer.writerow(non_param_header+param_header)
        for result in results:
            writer.writerow([result[i] for i in non_param_header]+[result['hyperparams'][i] for i in param_header])

    # Save best hyperparameter settings as JSON config file
    makedirs(args.config_save_path, isfile=True)

    with open(args.config_save_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


def chemprop_raytune() -> None:
    """Runs hyperparameter optimization for a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_hyperopt`.
    """
    raytune(args=HyperoptArgs().parse_args())
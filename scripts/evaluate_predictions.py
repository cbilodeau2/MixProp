""" Evaluates predictions. """
from typing import List
import warnings

import numpy as np
import pandas as pd
from tap import Tap

from chemprop.args import DatasetType, Metric
from chemprop.train import evaluate_predictions


class Args(Tap):
    pred_path: str  # Path to CSV file containing predictions.
    true_path: str  # Path to CSV file containing true values.
    pred_smiles_column: str = None  # Name of the column in pred_path containing SMILES (if None, uses first column).
    true_smiles_column: str = None  # Name of the column in true_path containing SMILES (if None, uses first column).
    pred_target_columns: List[str] = None  # List of columns in pred_path containing target values (if None, uses all except smiles_columns).
    true_target_columns: List[str] = None  # List of columns in true_path containing target values (if None, uses all except smiles_columns).
    dataset_type: DatasetType  # Dataset type.
    metrics: List[Metric]  # List of metrics to apply.
    metric_by_row: bool = False  # Whether to evaluate the metric row-wise rather than column-wise.


def eval_preds(args: Args) -> None:
    """ Evaluates predictions. """
    # Load pred and true
    pred = pd.read_csv(args.pred_path)
    true = pd.read_csv(args.true_path)

    # Get SMILES and target columns
    if args.pred_smiles_column is None:
        args.pred_smiles_column = pred.columns[0]

    if args.true_smiles_column is None:
        args.true_smiles_column = true.columns[0]

    if args.pred_target_columns is None:
        args.pred_target_columns = list(pred.columns)
        args.pred_target_columns.remove(args.pred_smiles_column)

    if args.true_target_columns is None:
        args.true_target_columns = list(true.columns)
        args.true_target_columns.remove(args.true_smiles_column)

    # Ensure SMILES and targets line up
    assert pred[args.pred_smiles_column].equals(true[args.true_smiles_column])

    if len(args.pred_target_columns) != len(args.true_target_columns):
        raise ValueError('Different number of targets between pred and true.')

    if set(args.pred_target_columns) != set(args.true_target_columns):
        warnings.warn(f'Target column names differ between pred and true.')
    elif args.pred_target_columns != args.true_target_columns:
        warnings.warn(f'Target column names are in a different order between pred and true.')

    # Extract predictions and true values
    preds = pred[args.pred_target_columns].to_numpy()
    targets = true[args.true_target_columns].to_numpy()

    # NaN to None
    preds = np.where(np.isnan(preds), None, preds)
    targets = np.where(np.isnan(targets), None, targets)

    # Convert to list
    preds = preds.tolist()
    targets = targets.tolist()

    # Evaluate predictions
    all_scores = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=len(preds[0]),
        metrics=args.metrics,
        dataset_type=args.dataset_type,
        metric_by_row=args.metric_by_row
    )

    # Print scores
    for metric, scores in all_scores.items():
        print(f'Overall {metric} = {np.nanmean(scores):.6f} +/- {np.nanstd(scores):.6f}')

        for task, task_score in sorted(zip(args.pred_target_columns, scores), key=lambda pair: pair[0]):
            print(f'    {task} {metric} = {task_score:.6f}')


if __name__ == '__main__':
    eval_preds(Args().parse_args())

""" Evaluates predictions. """
from typing import List

import numpy as np
import pandas as pd
from tap import Tap

from chemprop.args import DatasetType, Metric
from chemprop.train import evaluate_predictions


class Args(Tap):
    pred_path: str  # Path to CSV file containing predictions.
    true_path: str  # Path to CSV file containing true values.
    smiles_column: str = None  # Name of the column containing SMILES (if None, uses first column).
    target_columns: List[str] = None  # List of columns containing target values (if None, uses all except smiles_columns).
    dataset_type: DatasetType  # Dataset type.
    metrics: List[Metric]  # List of metrics to apply.
    metric_by_row: bool = False  # Whether to evaluate the metric row-wise rather than column-wise.


def eval_preds(args: Args) -> None:
    """ Evaluates predictions. """
    # Load pred and true
    pred = pd.read_csv(args.pred_path)
    true = pd.read_csv(args.true_path)

    # Ensure that pred and true line up
    smiles_column = args.smiles_column if args.smiles_column is not None else pred.columns[0]
    target_columns = args.target_columns if args.target_columns is not None else set(pred.columns) - {smiles_column}

    # Ensure SMILES line up
    assert pred[smiles_column].equals(true[smiles_column])

    # Extract predictions and true values
    preds = pred[target_columns].to_numpy()
    targets = true[target_columns].to_numpy()

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

        for task, task_score in sorted(zip(target_columns, scores), key=lambda pair: pair[0]):
            print(f'    {task} {metric} = {task_score:.6f}')


if __name__ == '__main__':
    eval_preds(Args().parse_args())

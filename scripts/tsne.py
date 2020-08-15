import os
import sys
import time
from typing import List, Optional

from matplotlib import offsetbox
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.manifold import TSNE
from tap import Tap
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.features import get_features_generator
from chemprop.utils import makedirs


class Args(Tap):
    smiles_paths: List[str]  # Path to .csv files containing smiles strings (with header)
    smiles_columns: List[str] = None  # Name of the column containing SMILES strings for the first data.
    """
    If None, uses the first column.
    If one is provided, uses that column for each data file.
    If more than one is provided, there must be one provided for each data file.
    """
    activity_columns: List[str] = None  # If provided, splits data into actives and inactives.
    """
    If None, does not split by activity.
    If one is provided, uses that column for each data file.
    If more than one is provided, there must be one provided for each data file.
    """
    colors: List[str] = ['red', 'blue', 'orange', 'green', 'purple']  # Colors of the points associated with each dataset
    sizes: List[float] = [1, 1, 1, 1, 1]  # Sizes of the points associated with each molecule
    scale: int = 1  # Scale of figure
    plot_molecules: bool = False  # Whether to plot images of molecules instead of points
    max_per_dataset: int = 10000  # Maximum number of molecules per dataset; larger datasets will be subsampled to this size
    save_path: str  # Path to a .png file where the t-SNE plot will be saved


def extend_arguments(arguments: Optional[List[str]], length: int) -> List[Optional[str]]:
    """Extends arguments to the provided length."""
    if arguments is None:
        return [None] * length

    if len(arguments) == 1:
        return arguments * length

    assert len(arguments) == length

    return arguments


def compare_datasets_tsne(args: Args):
    if len(args.smiles_paths) * (1 + (args.activity_columns is not None)) > len(args.colors):
        raise ValueError('Must have at least as many colors and sizes as datasets (times 2 if splitting by activity)')

    # Extend SMILES columns and activity columns
    smiles_columns = extend_arguments(arguments=args.smiles_columns, length=len(args.smiles_paths))
    activity_columns = extend_arguments(arguments=args.activity_columns, length=len(args.smiles_paths))

    # Random seed for random subsampling
    np.random.seed(0)

    # Load the smiles datasets
    print('Loading data')
    smiles, slices, labels = [], [], []
    for smiles_path, smiles_column, activity_column in zip(args.smiles_paths, smiles_columns, activity_columns):
        # Get label
        label = os.path.basename(smiles_path).replace('.csv', '')

        # Load data
        data = pd.read_csv(smiles_path)
        smiles_column = smiles_column if smiles_column is not None else data.keys()[0]

        # Get SMILES and labels
        if activity_column is not None:
            new_smiles_sets = [data[data[activity_column] == 1][smiles_column],
                               data[data[activity_column] == 0][smiles_column]]
            new_labels = [f'{label}_active', f'{label}_inactive']
        else:
            new_smiles_sets = [data[smiles_column]]
            new_labels = [label]

        for new_smiles, label in zip(new_smiles_sets, new_labels):
            new_smiles = list(new_smiles)
            print(f'{label}: {len(new_smiles):,}')

            # Subsample if dataset is too large
            if len(new_smiles) > args.max_per_dataset:
                print(f'Subsampling to {args.max_per_dataset:,} molecules')
                new_smiles = np.random.choice(new_smiles, size=args.max_per_dataset, replace=False).tolist()

            slices.append(slice(len(smiles), len(smiles) + len(new_smiles)))
            labels.append(label)
            smiles += new_smiles

    # Compute Morgan fingerprints
    print('Computing Morgan fingerprints')
    morgan_generator = get_features_generator('morgan')
    morgans = [morgan_generator(smile) for smile in tqdm(smiles, total=len(smiles))]

    print('Running t-SNE')
    start = time.time()
    tsne = TSNE(n_components=2, init='pca', random_state=0, metric='jaccard')
    X = tsne.fit_transform(morgans)
    print(f'time = {time.time() - start:.2f} seconds')

    print('Plotting t-SNE')
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    makedirs(args.save_path, isfile=True)

    plt.clf()
    fontsize = 50 * args.scale
    fig = plt.figure(figsize=(64 * args.scale, 48 * args.scale))
    plt.title('t-SNE using Morgan fingerprint with Jaccard similarity', fontsize=2 * fontsize)
    ax = fig.gca()
    handles = []
    legend_kwargs = dict(loc='upper right', fontsize=fontsize)

    for slc, color, label, size in zip(slices, args.colors, labels, args.sizes):
        if args.plot_molecules:
            # Plots molecules
            handles.append(mpatches.Patch(color=color, label=label))

            for smile, (x, y) in zip(smiles[slc], X[slc]):
                img = Draw.MolsToGridImage([Chem.MolFromSmiles(smile)], molsPerRow=1, subImgSize=(200, 200))
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), (x, y), bboxprops=dict(color=color))
                ax.add_artist(imagebox)
        else:
            # Plots points
            plt.scatter(X[slc, 0], X[slc, 1], s=150 * size, color=color, label=label)

    if args.plot_molecules:
        legend_kwargs['handles'] = handles

    plt.legend(**legend_kwargs)
    plt.xticks([]), plt.yticks([])

    print('Saving t-SNE')
    plt.savefig(args.save_path)


if __name__ == '__main__':
    compare_datasets_tsne(Args().parse_args())

from typing import Iterable, List
from typing_extensions import Literal

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Fragments
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    data_path: str
    antibiotics_path: str
    save_path: str
    score_function: Literal['rotatable_bonds', 'amine', 'smiles_len'] = 'rotatable_bonds'
    smiles_column: str = 'canonical_smiles'  # assumes canonicalized
    bin_size: int = 25000


def num_rotatable_bonds(smiles: str) -> int:
    return CalcNumRotatableBonds(Chem.MolFromSmiles(smiles))


def has_amine(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)

    num_primary = Fragments.fr_NH2(mol)
    num_secondary = Fragments.fr_NH1(mol)
    num_tertiary = Fragments.fr_NH0(mol)
    num_amines = num_primary + num_secondary + num_tertiary

    return num_amines > 0


def run_antibiotic_score(args: Args) -> None:
    # Get score function (higher means more antibiotic-likes)
    if args.score_function == 'rotatable_bonds':
        score_function = num_rotatable_bonds
    elif args.score_function == 'amine':
        score_function = has_amine
    elif args.score_function == 'smiles_len':
        score_function = len
    else:
        raise ValueError(f'Score function "{args.score_function}" not supported.')

    # Load data
    data = pd.read_csv(args.data_path)
    antibiotics = pd.read_csv(args.antibiotics_path)

    # Filter bad smiles
    data = data[~data[args.smiles_column].isna()]

    # Determine which data smiles are antibiotics
    data['is_antibiotic'] = data[args.smiles_column].isin(antibiotics[args.smiles_column])

    # Score data smiles
    data['score'] = [score_function(smiles) for smiles in tqdm(data[args.smiles_column])]

    # Sort by score
    data.sort_values(by='score', ascending=False, inplace=True)

    # Compute and plot histogram
    categorical_score = args.score_function in {'rotatable_bonds', 'amine', 'smiles_len'}

    if categorical_score:
        values = sorted(data['score'].unique())
        proportions = []

        for value in values:
            bin_data = data[data['score'] == value]
            proportion = sum(bin_data['is_antibiotic']) / len(bin_data)
            proportions.append(proportion)

        x_vals = list(range(len(values)))
        plt.bar(x_vals, proportions)
        plt.xticks(x_vals, values)
        plt.ylabel('Proportion of antibiotics')
        plt.xlabel(args.score_function)
        plt.title(f'Histogram with proprortion of antibiotics vs {args.score_function}')
        plt.savefig(args.save_path)
    else:

        # Compute histogram
        values = list(range(0, len(data) + args.bin_size, args.bin_size))
        proportions = []

        for i in range(len(values) - 1):
            bin_data = data.iloc[values[i]:values[i + 1]]
            if len(bin_data) > 0:
                proportion = sum(bin_data['is_antibiotic']) / len(bin_data)
                proportions.append(proportion)

        # Plot histogram
        size = min(len(values), len(proportions))
        values, proportions = values[:size], proportions[:size]

        plt.bar(list(range(size)), proportions, align='edge')
        plt.xticks(list(range(0, size, 4)), values[::4])
        plt.ylabel('Proportion of antibiotics')
        plt.xlabel(f'Rank according to antibiotic score (intervals of {args.bin_size:,})')
        plt.title('Histogram with proprortion of antibiotics vs antibiotic score rank')
        plt.savefig(args.save_path)
        # plt.show()


if __name__ == '__main__':
    run_antibiotic_score(Args().parse_args())

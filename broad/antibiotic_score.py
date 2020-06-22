"""See https://www.nature.com/articles/nature22308 and https://www.nature.com/articles/s41564-019-0604-5 and https://github.com/HergenrotherLab/entry-cli"""

from functools import partial
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
    score_function: Literal['rotatable_bonds', 'amine', 'primary_amine', 'carboxylic_acid', 'smiles_len']
    smiles_column: str = 'canonical_smiles'  # assumes canonicalized
    bin_size: int = 25000


def num_rotatable_bonds(smiles: str) -> int:
    return CalcNumRotatableBonds(Chem.MolFromSmiles(smiles))


def has_amine(smiles: str, primary: bool = True, secondary: bool = True, tertiary: bool = True) -> bool:
    mol = Chem.MolFromSmiles(smiles)

    num_primary = Fragments.fr_NH2(mol)
    num_secondary = Fragments.fr_NH1(mol)
    num_tertiary = Fragments.fr_NH0(mol)
    num_amines = primary * num_primary + secondary * num_secondary + tertiary * num_tertiary

    return num_amines > 0


def has_carboxylic_acid(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)

    num_coo = Fragments.fr_COO(mol)

    return num_coo > 0


def run_antibiotic_score(args: Args) -> None:
    # Get score function (higher means more antibiotic-likes)
    if args.score_function == 'rotatable_bonds':
        score_function = num_rotatable_bonds
    elif args.score_function == 'amine':
        score_function = partial(has_amine, primary=True, secondary=True, tertiary=True)
    elif args.score_function == 'primary_amine':
        score_function = partial(has_amine, primary=True, secondary=False, tertiary=False)
    elif args.score_function == 'carboxylic_acid':
        score_function = has_carboxylic_acid
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
    # TODO: adapt for non-categorical
    categorical_score = True

    if categorical_score:
        values = sorted(data['score'].unique())
        proportions = []

        for value in values:
            bin_data = data[data['score'] == value]
            num_with_value = len(bin_data)
            num_with_value_antibiotic = sum(bin_data['is_antibiotic'])
            proportion_antibiotic = num_with_value_antibiotic / num_with_value
            proportions.append(proportion_antibiotic)

            print(f'Value={value}')
            print(f'Num with value={num_with_value:,}')
            print(f'Num with value and antibiotic={num_with_value_antibiotic:,} ({100 * proportion_antibiotic:.4f}%)')

        x_vals = list(range(len(values)))
        plt.bar(x_vals, proportions)
        plt.xticks(x_vals, values)
        plt.ylabel('Proportion of antibiotics')
        plt.xlabel(args.score_function)
        plt.title(f'Histogram with proportion of antibiotics vs {args.score_function}')
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

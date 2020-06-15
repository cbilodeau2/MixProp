from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    data_path: str
    antibiotics_path: str
    save_path: str
    smiles_column: str = 'canonical_smiles'  # assumes canonicalized
    bin_size: int = 50000


def antibiotic_score(smiles: Iterable[str]) -> List[float]:
    """Higher score means more antibiotic-like"""
    return [CalcNumRotatableBonds(Chem.MolFromSmiles(smile)) for smile in tqdm(smiles)]


def run_antibiotic_score(args: Args) -> None:
    # Load data
    data = pd.read_csv(args.data_path)
    antibiotics = pd.read_csv(args.antibiotics_path)

    # Filter bad smiles
    data = data[~data[args.smiles_column].isna()]

    # Determine which data smiles are antibiotics
    data['is_antibiotic'] = data[args.smiles_column].isin(antibiotics[args.smiles_column])

    # Score data smiles
    data['score'] = antibiotic_score(data[args.smiles_column])

    # Sort by score
    data.sort_values(by='score', ascending=False, inplace=True)

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
    plt.xticks(list(range(0, size, 2)), values[::2])
    plt.ylabel('Proportion of antibiotics')
    plt.xlabel(f'Rank according to antibiotic score (intervals of {args.bin_size:,})')
    plt.title('Histogram with proprortion of antibiotics vs antibiotic score rank')
    plt.savefig(args.save_path)
    # plt.show()


if __name__ == '__main__':
    run_antibiotic_score(Args().parse_args())

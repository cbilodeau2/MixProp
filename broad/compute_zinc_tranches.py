"""Uses molecular weight and logP to determine which tranches from ZINC15 molecules belong to.
Tranches: https://zinc15.docking.org/tranches/home/#
"""

from string import ascii_uppercase

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import MolWt
from tap import Tap  # pip install typed-argument-parser https://github.com/swansonk14/typed-argument-parser
from tqdm import tqdm


class Args(Tap):
    data_path: str  # Path to CSV file containing molecular data
    save_path: str  # Path where resulting CSV file with ZINC15 tranche data will be saved
    smiles_column: str = 'SMILES'  # Name of column containing SMILES in data_path
    deduplicate_by_smiles: bool = False  # Whether to de-duplicate based on canonical SMILES


ASCII_UPPERCASE = np.array(list(ascii_uppercase))

# Each bin includes up to that value
MOLWT_TRANCHES = [200, 250, 300, 325, 350, 375, 400, 425, 450, 500, np.inf]
LOGP_TRANCHES = [-1, 0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, np.inf]


def compute_zinc_tranches(args: Args) -> None:
    # Load data
    data = pd.read_csv(args.data_path)

    # Convert SMILES to mols
    data['mols'] = [Chem.MolFromSmiles(smiles) for smiles in tqdm(data[args.smiles_column])]

    # Compute canonical smiles
    data['canonical_smiles'] = [Chem.MolToSmiles(mol) if mol is not None else None for mol in tqdm(data['mols'])]

    # Optionally de-duplicate by canonical smiles
    if args.deduplicate_by_smiles:
        data.drop_duplicates(subset='canonical_smiles', inplace=True)
        data.reset_index(drop=True, inplace=True)

    # Filter out invalid mols
    valid_indices = [i for i, mol in enumerate(data['mols']) if mol is not None]
    print(f'Total number of SMILES = {len(data):,}')
    print(f'Number of valid SMILES = {len(valid_indices):,}')
    print(f'Number of invalid SMILES = {len(data) - len(valid_indices):,}')
    mols = [mol for mol in data['mols'] if mol is not None]

    # Compute molecular weight and logp
    molwt, logp = zip(*[(MolWt(mol), MolLogP(mol)) for mol in tqdm(mols) if mol is not None])

    # Determine tranches
    molwt_tranches = ASCII_UPPERCASE[np.digitize(molwt, MOLWT_TRANCHES)]
    logp_tranches = ASCII_UPPERCASE[np.digitize(logp, LOGP_TRANCHES)]
    tranches = [f'{molwt_tranche}{logp_tranche}' for molwt_tranche, logp_tranche in zip(molwt_tranches, logp_tranches)]

    # Set tranches and remove mols
    data.loc[valid_indices, 'tranche'] = tranches
    data.drop(columns='mols', inplace=True)

    # Print tranche statistics
    for tranche, count in data['tranche'].value_counts().iteritems():
        print(f'{tranche}: {count:,}')

    # Save data
    data.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    compute_zinc_tranches(Args().parse_args())

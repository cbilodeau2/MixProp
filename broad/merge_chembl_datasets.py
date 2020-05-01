"""Merges ChEMBL datasets derived from various search terms.

De-duplicates by ID and drops compounds without SMILES strings.

https://www.ebi.ac.uk/chembl/g/#search_results/compounds/query=antibacterial
https://www.ebi.ac.uk/chembl/g/#search_results/compounds/query=antibiotic
https://www.ebi.ac.uk/chembl/g/#search_results/compounds/query=antiinfective
https://www.ebi.ac.uk/chembl/g/#search_results/compounds/query=antiseptic
"""

from typing import List

import pandas as pd
from rdkit import Chem
from tap import Tap


class Args(Tap):
    data_paths: List[str]  # Paths to CSV files containing downloaded ChEMBL data
    save_path: str  # Path where combined ChEMBL data will be saved
    id_column: str = 'ChEMBL ID'  # Name of the column containing the compound ID
    smiles_column: str = 'Smiles'  # Name of column containing SMILES in data_paths
    deduplicate_by_smiles: bool = False  # Whether to de-duplicate based on canonical SMILES


def merge_chembl_datasets(args: Args) -> None:
    # Load data
    datasets = [pd.read_csv(data_path) for data_path in args.data_paths]

    # Merge datasets, de-duplicating by ID
    data = pd.concat(datasets).drop_duplicates(subset=args.id_column)

    # Only keep compounds with SMILES
    data = data[data[args.smiles_column].notna()]

    # Compute canonical smiles
    data['canonical_smiles'] = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in data[args.smiles_column]]

    # Optionally de-duplicate by canonical SMILES
    if args.deduplicate_by_smiles:
        data.drop_duplicates(subset='canonical_smiles', inplace=True)

    print(f'Dataset size = {len(data):,}')

    # Save data
    data.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    merge_chembl_datasets(Args().parse_args())

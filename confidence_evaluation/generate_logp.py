"""Compute and save a csv file with the MolLogP feature."""
import csv
import os
import sys

from rdkit import Chem

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data.utils import get_data

if __name__ == "__main__":
    datasets = ['lipo', 'delaney', 'freesolv', 'qm7']

    full_molecule_set = set()
    for dataset in datasets:
        data = get_data(path=f'data/{dataset}.csv')
        for smile in data.smiles():
            if smile not in full_molecule_set:
                full_molecule_set.add(smile)

    full_molecule_list = list(full_molecule_set)
    logp_list = []
    for molecule in full_molecule_list:
        logp_list.append(Chem.Crippen.MolLogP(Chem.MolFromSmiles(molecule)))

    with open('data/logp.csv', 'w+') as logp_csv:
        csv_writer = csv.writer(logp_csv, delimiter=',')
        csv_writer.writerow(['smiles', 'logp'])
        csv_writer.writerows([[full_molecule_list[i],
                               logp_list[i]] for i in range(
                                len(full_molecule_list))])

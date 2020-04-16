"""Stores a list of targets associated with a set of molecules."""
from chembl_webresource_client.new_client import new_client as chembl_client
import csv
import typer
from tqdm import tqdm

from molecules import Drug
from utils import extract_column_from_csv, write_rows_to_csv

app = typer.Typer()


@app.command()
def fetch_drugs(smiles_filename: str,
                output_filename: str,
                file_delimiter: str = '\t',
                entry_delimiter: str = ', ',
                chunk_size: int = 50):
    """
    :param str smiles_filename: a CSV file with a SMILES column.
    :param str output_filename: the file to save drug info to.
    :param str file_delimiter: the delimiter used by the input CSV.
    :param str entry_delimiter: the delimiter used to split entries, if any.
    :param int chunk_size: the number of drugs to lookup on chEMBL at a time.
    """
    smiles_list = extract_column_from_csv(smiles_filename,
                                          'SMILES',
                                          file_delimiter=file_delimiter,
                                          entry_delimiter=entry_delimiter)
    drug_ids_to_info = {}

    # Find Drug basic info.
    drugs = []
    missed_count = 0
    for smiles in tqdm(smiles_list, desc='Fetching basic drug info.'):
        try:
            drugs.append(Drug(smiles=smiles))
        except ValueError:
            missed_count += 1

    print(f'Could not identify {missed_count}/{len(smiles_list)} drugs.')

    for drug in drugs:
        drug_ids_to_info[drug.chembl_id] = {'Name': drug.name,
                                            'chEMBL ID': drug.chembl_id,
                                            'SMILES': drug.smiles,
                                            'Target chEMBL IDs': set()}

    # Find drug targets.
    drug_ids = list(drug_ids_to_info.keys())
    for i in tqdm(range(0, len(drug_ids), chunk_size),
                  desc='Fetching drug targets'):
        mechanisms = chembl_client.mechanism.filter(
            molecule_chembl_id__in=drug_ids[i:i+chunk_size]).only([
                    'target_chembl_id', 'molecule_chembl_id'])

        for m in tqdm(mechanisms, desc='Processing chunk', leave=False):
            if m['target_chembl_id'] is not None:
                drug_info = drug_ids_to_info[m['molecule_chembl_id']]
                drug_info['Target chEMBL IDs'].add(m['target_chembl_id'])

    column_names = ['Name', 'chEMBL ID', 'SMILES', 'Target chEMBL IDs']
    write_rows_to_csv(output_filename,
                      column_names,
                      list(drug_ids_to_info.values()),
                      file_delimiter=file_delimiter,
                      entry_delimiter=entry_delimiter)


@app.command()
def fetch_targets(ids_filename: str,
                  output_filename: str,
                  file_delimiter: str = '\t',
                  entry_delimiter: str = ', ',
                  chunk_size: int = 50):
    """
    :param str ids_filename: a CSV file with a 'Target chEMBL IDs' column.
    :param str output_filename: the name of the file to save target info to.
    :param str file_delimiter: the delimiter used by the input CSV.
    :param str entry_delimiter: the delimiter used to split entries, if any.
    :param int chunk_size: the number of targets to lookup on chEMBL at a time.
    """
    target_ids = extract_column_from_csv(ids_filename,
                                         'Target chEMBL IDs',
                                         file_delimiter=file_delimiter,
                                         entry_delimiter=entry_delimiter)

    # Find GO terms associated with each Target.
    target_data_list = []
    for i in tqdm(range(0, len(target_ids), chunk_size),
                  desc='Fetching GO Terms'):
        targets = chembl_client.target.filter(
            target_chembl_id__in=target_ids[i:i+chunk_size]).only([
                'target_components',
                'target_chembl_id',
                'organism',
                'pref_name'])

        seen_target_ids = set()
        for target in tqdm(targets, desc='Processing chunk', leave=False):
            if target['target_chembl_id'] in seen_target_ids:
                continue

            seen_target_ids.add(target['target_chembl_id'])
            target_data = {'GoComponent': set(),
                           'GoFunction': set(),
                           'GoProcess': set()}

            for component in target['target_components']:
                xrefs = component['target_component_xrefs']
                for xref in xrefs:
                    if xref['xref_src_db'] not in ['GoComponent',
                                                   'GoFunction',
                                                   'GoProcess']:
                        continue

                    target_data[xref['xref_src_db']].add(xref['xref_id'])

            target_data['chEMBL ID'] = target['target_chembl_id']
            target_data['Organism'] = target['organism']
            target_data['Name'] = target['pref_name']
            target_data_list.append(target_data)

    column_names = ['Name',
                    'Organism',
                    'chEMBL ID',
                    'GoComponent',
                    'GoFunction',
                    'GoProcess']

    write_rows_to_csv(output_filename,
                      column_names,
                      target_data_list,
                      file_delimiter=file_delimiter,
                      entry_delimiter=entry_delimiter)


@app.command()
def join_drugs_to_targets(drugs_filename: str,
                          targets_filename: str,
                          output_filename: str,
                          file_delimiter: str = '\t',
                          entry_delimiter: str = ', '):
    """
    Matches data about fetched drugs and their targets by chEMBL id.

    :param str drugs_filename: a CSV file with drug data.
    :param str targets_filename: a CSV file with target data.
    :param str output_filename: the name of the file to save joined info to.
    :param str file_delimiter: the delimiter used by the input CSV.
    :param str entry_delimiter: the delimiter used to split entries, if any.
    """
    target_columns = ['Name',
                      'Organism',
                      'chEMBL ID',
                      'GoComponent',
                      'GoFunction',
                      'GoProcess']

    line_count = 0
    targets = {}
    with open(targets_filename) as file_:
        reader = csv.reader(file_, delimiter=file_delimiter)

        for row in reader:
            if line_count == 0:
                column_indices = {c: row.index(c) for c in target_columns}
            else:
                target = {c: row[column_indices[c]] for c in target_columns}

                targets[row[column_indices['chEMBL ID']]] = target

            line_count += 1

    drug_columns = ['Name',
                    'chEMBL ID',
                    'SMILES',
                    'Target chEMBL IDs']

    line_count = 0
    drugs = []
    with open(drugs_filename) as file_:
        reader = csv.reader(file_, delimiter=file_delimiter)

        for row in reader:
            if line_count == 0:
                column_indices = {c: row.index(c) for c in drug_columns}
            else:
                drug = {c: row[column_indices[c]] for c in drug_columns}
                drug['Organisms'] = []
                drug['GoComponent'] = []
                drug['GoFunction'] = []
                drug['GoProcess'] = []

                if drug['Target chEMBL IDs'] != '':
                    target_ids = drug['Target chEMBL IDs'].split(
                        entry_delimiter)

                    for target_id in target_ids:
                        target = targets[target_id]
                        drug['Organisms'].append(target['Organism'])

                        drug['GoComponent'] += target['GoComponent'].split(
                            entry_delimiter)
                        drug['GoFunction'] += target['GoFunction'].split(
                            entry_delimiter)
                        drug['GoProcess'] += target['GoProcess'].split(
                            entry_delimiter)

                drug['GoComponent'] = set(drug['GoComponent'])
                drug['GoFunction'] = set(drug['GoFunction'])
                drug['GoProcess'] = set(drug['GoProcess'])

                drugs.append(drug)

            line_count += 1

    write_rows_to_csv(output_filename,
                      ['Name',
                       'chEMBL ID',
                       'SMILES',
                       'Target chEMBL IDs',
                       'Organisms',
                       'GoComponent',
                       'GoFunction',
                       'GoProcess'],
                      drugs,
                      file_delimiter=file_delimiter,
                      entry_delimiter=entry_delimiter)


if __name__ == '__main__':
    app()

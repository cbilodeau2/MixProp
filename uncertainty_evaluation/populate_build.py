"""Writes the commands to execute a series of experiments to populate.sh."""
import os
from typing import Dict, TextIO, Union


def params_to_line(params: Dict[str, Union[str, int]]) -> str:
    """
    Transforms a dictionary of parameters into a command line arguments.

    :param params: A dictionary of parameter names to values.
    :return: A executable python command.
    """
    base = 'python train.py'
    for param, value in params.items():
        if value is None:
            base += f' --{param}'
        else:
            base += f' --{param} {value}'

    return base + '\n'


def write_experiments(method: str,
                      dataset: str,
                      experiment_filename: str,
                      populate_file: TextIO):
    """
    Writes the commands for a series of experiments to a shell script.

    :param method: The uncertainty estimation used in the experiment.
    :param dataset: The dataset the experiment is run on.
    :param experiment_filename: The file the experiment writes to.
    :param populate_file: The shell script file to write the command to.
    """
    for i in range(8):
        params = {'data_path': f'data/{dataset}.csv',
                  'dataset_type': 'regression',
                  'epochs': 30,
                  'split_type': 'random',
                  'seed': i,
                  'ensemble_size': 1,
                  'uncertainty': method,
                  'split_sizes': '0.5 0.2 0.3',
                  'dropout': 0}

        if method in {'ensemble', 'bootstrap', 'snapshot', 'dropout'}:
            params['ensemble_size'] = 16

        if 'dropout10' in experiment_filename:
            params['dropout'] = 0.1

        if 'dropout20' in experiment_filename:
            params['dropout'] = 0.2

        if 'ffn' in experiment_filename or 'fp' in experiment_filename:
            params['depth'] = 0
            params['features_only'] = None
            params['features_generator'] = 'morgan'

        folder = f'uncertainty_evaluation/uncalibrated/{experiment_filename}/'\
                 f'{dataset}/{params["split_type"]}'

        if not os.path.exists(folder):
            os.makedirs(folder)
        params['save_uncertainty'] = f'{folder}/{i}.txt'

        populate_file.write(params_to_line(params))

    del params['seed']

    params['split_type'] = 'scaffold'

    folder = f'uncertainty_evaluation/uncalibrated/{experiment_filename}/' \
             f'{dataset}/{params["split_type"]}'

    if not os.path.exists(folder):
        os.makedirs(folder)
    params['save_uncertainty'] = f'{folder}/{i}.txt'

    populate_file.write(params_to_line(params))
    populate_file.write('\n')
    populate_file.write('\n')


if __name__ == "__main__":
    methods_to_files = {'mve':              ['mpnn_mve',
                                             'ffn_mve'],
                        'gaussian':         ['mpnn_gaussian',
                                             'ffn_gaussian'],
                        'random_forest':    ['mpnn_random_forest',
                                             'ffn_random_forest'],
                        'tanimoto':         ['mpnn_tanimoto',
                                             'ffn_tanimoto'],
                        'latent_space':     ['mpnn_latent_space',
                                             'ffn_latent_space'],
                        'ensemble':         ['mpnn_ensemble',
                                             'ffn_ensemble'],
                        'bootstrap':        ['mpnn_bootstrap',
                                             'ffn_bootstrap'],
                        'snapshot':         ['mpnn_snapshot',
                                             'ffn_snapshot'],
                        'dropout':          ['mpnn_dropout10',
                                             'mpnn_dropout20',
                                             'ffn_dropout10',
                                             'ffn_dropout20'],
                        'fp_random_forest': ['fp_random_forest'],
                        'fp_gaussian':      ['fp_gaussian']}

    f = open('uncertainty_evaluation/populate.sh', 'w+')

    datasets = ['lipo', 'delaney', 'freesolv', 'qm7', 'logp']

    for method, files in methods_to_files.items():
        for filename in files:
            for dataset in datasets:
                write_experiments(method, dataset, filename, f)

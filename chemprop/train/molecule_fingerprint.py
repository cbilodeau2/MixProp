import csv
from typing import List, Optional, Union

import torch
from tqdm import tqdm

from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, makedirs, timeit
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.models import MoleculeModel

@timeit()
def molecule_fingerprint(args: PredictArgs, smiles: List[List[str]] = None) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to encode fingerprint vectors for the data.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :return: A list of fingerprint vectors (list of floats)
    """

    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args: Union[PredictArgs, TrainArgs]

    # Same number of molecules must be used in training as in making predictions
    assert train_args.number_of_molecules == args.number_of_molecules, (
        'A different number of molecules was used in training '
        f'model than is specified for prediction, {train_args.number_of_molecules} '
        'smiles fields must be provided')

    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator
        )
    else:
        full_data = get_data(path=args.test_path, target_columns=[], ignore_columns=[], skip_invalid_smiles=False,
                             args=args, store_row=True)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f'Encoding smiles into a fingerprint vector from a single model')
    assert len(args.checkpoint_paths) == 1, "Fingerprint generation only supports one model, cannot use an ensemble"
    # Load model
    model = load_checkpoint(args.checkpoint_paths[0], device=args.device)

    model_preds = model_fingerprint(
        model=model,
        data_loader=test_data_loader
    )

    # Save predictions
    print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(model_preds)
    makedirs(args.preds_path, isfile=True)

    # Copy predictions over to full_data
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = model_preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * (args.hidden_size*args.number_of_molecules)

        fingerprint_columns=[f'fp_{i}' for i in range(args.hidden_size*args.number_of_molecules)]
        for i in range(len(fingerprint_columns)):
            datapoint.row[fingerprint_columns[i]] = preds[i]

    # Write predictions
    with open(args.preds_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=args.smiles_columns+fingerprint_columns,extrasaction='ignore')
        writer.writeheader()
        for datapoint in full_data:
            writer.writerow(datapoint.row)
    return model_preds

def model_fingerprint(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False) -> List[List[float]]:
    """
    Encodes the provided molecules into the latent fingerprint vectors, according to the provided model.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :return: A list of fingerprint vector lists.
    """
    model.eval()

    fingerprints = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch = batch.batch_graph(), batch.features(), batch.atom_descriptors()

        # Make predictions
        with torch.no_grad():
            batch_fp = model.fingerprint(mol_batch, features_batch, atom_descriptors_batch)

        # Collect vectors
        batch_fp = batch_fp.data.cpu().tolist()

        fingerprints.extend(batch_fp)

    return fingerprints

def chemprop_fingerprint() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    molecule_fingerprint(args=PredictArgs().parse_args())
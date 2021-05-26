from typing import List

import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    preds = []
    ale_unc = []

    # if aleatoric uncertainty is enabled
    aleatoric = model.aleatoric

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features()

        if not aleatoric:
            # Make predictions
            with torch.no_grad():
                batch_preds = model(mol_batch, features_batch, atom_descriptors_batch,
                                    atom_features_batch, bond_features_batch)

            batch_preds = batch_preds.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)
        else:
            # Make predictions
            with torch.no_grad():
                batch_preds, batch_logvar = model(mol_batch, features_batch,
                    atom_descriptors_batch, atom_features_batch, bond_features_batch)
                batch_var = torch.exp(batch_logvar)

            batch_preds = batch_preds.data.cpu().numpy()
            batch_ale_unc = batch_var.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)
                batch_ale_unc = scaler.inverse_transform_variance(batch_ale_unc)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            batch_ale_unc = batch_ale_unc.tolist()
            preds.extend(batch_preds)
            ale_unc.extend(batch_ale_unc)

    if not aleatoric:
        return preds, None
    else:
        return preds, ale_unc

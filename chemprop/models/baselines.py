"""Contains baseline models."""
from typing import Any, List, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.data import MoleculeDataset
from chemprop.features import BatchMolGraph


class SimpleBaselineModel(nn.Module):
    def __init__(self, dataset_type: str, train_data: MoleculeDataset):
        super(SimpleBaselineModel, self).__init__()
        self.dataset_type = dataset_type
        self.preds = self.fit(train_data)

    def fit(self, data: MoleculeDataset) -> torch.FloatTensor:
        targets = np.array(data.targets())

        if self.dataset_type == 'regression':
            self.preds = np.nanmean(targets, axis=0)
        elif self.dataset_type == 'classification':
            num_active = (targets == 1).sum(axis=0)
            num_inactive = (targets == 0).sum(axis=0)
            self.preds = num_active / (num_active + num_inactive)
        else:
            raise ValueError(f'Dataset type "{self.dataset_type}" not supported.')

        self.preds = torch.from_numpy(self.preds)

        return self.preds

    def forward(self, batch: Union[List[str], List[Chem.Mol], BatchMolGraph], *inputs: Any) -> torch.FloatTensor:
        if type(batch) == BatchMolGraph:
            batch_size = len(batch.a_scope)
        else:
            batch_size = len(batch)

        return self.preds.repeat(batch_size, 1)

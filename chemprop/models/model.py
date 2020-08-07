from typing import List, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .dag_model import DAGModel, MLP
from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights


class CombineEncodedReadout(nn.Module):
    """Modules which combines the molecule encoding and readout to perform one additional prediction."""

    def __init__(self, args: TrainArgs):
        super(CombineEncodedReadout, self).__init__()
        self.mlp = MLP(
            in_features=args.hidden_size + args.num_tasks - 1,
            hidden_features=args.hidden_size,
            out_features=1,
            activation=args.activation
        )

    def forward(self,
                encoded: torch.FloatTensor,  # (batch_size, hidden_size)
                readout: torch.FloatTensor  # (batch_size, output_size - 1)
                ) -> torch.FloatTensor:  # (batch_size, output_size)
        input = torch.cat((encoded, readout), dim=1)  # (batch_size, hidden_size + output_size - 1)
        output = self.mlp(input)  # (batch_size, 1)
        output = torch.cat((output, readout), dim=1)  # (batch_size, output_size)

        return output


def combine_readout_only(encoded: torch.FloatTensor, readout: torch.FloatTensor) -> torch.FloatTensor:
    """Returns the readout only."""
    return readout


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e., outputting the
                           learned features from the last layer prior to prediction rather than
                           outputting the actual property predictions.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer
        self.device = args.device
        self.organism_and_go = args.organism_and_go

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.lineage_embedding_type = args.lineage_embedding_type

        self.encoder = MPN(args)

        if args.features_only:
            self.first_linear_dim = args.features_size
        else:
            self.first_linear_dim = args.hidden_size
            if args.use_input_features:
                self.first_linear_dim += args.features_size
        if args.use_taxon:
            self.first_linear_dim += args.hidden_size

        if args.use_go_dag:
            self.readout = DAGModel(
                dag=args.go_dag,
                input_size=self.first_linear_dim,
                hidden_size=args.hidden_size,
                embedding_size=args.go_embedding_size,
                layer_type=args.go_dag_layer_type,
                activation=args.activation
            )
        else:
            self.readout = self.create_ffn(args)

        if self.organism_and_go:
            self.combine = CombineEncodedReadout(args)
        else:
            self.combine = combine_readout_only

        initialize_weights(self)

        if args.use_taxon:
            self.create_lineage_embedding(args)

    def create_ffn(self, args: TrainArgs) -> nn.Module:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :return: A PyTorch module with the feed-forward layers.
        """
        output_size = self.output_size - 1 if self.organism_and_go else self.output_size
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(self.first_linear_dim, output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(self.first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, output_size),
            ])

        # Create FFN model
        ffn = nn.Sequential(*ffn)

        return ffn

    def create_lineage_embedding(self, args: TrainArgs) -> None:
        """
        Creates embedding layer(s) to embed the organism lineage.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.taxon_embedder = nn.Embedding(
            num_embeddings=args.num_taxons,
            embedding_dim=args.hidden_size,
            padding_idx=0
        )

        if args.lineage_embedding_type == 'rnn_lineage':
            self.taxon_rnn = nn.LSTM(
                input_size=self.taxon_embedder.embedding_dim,
                hidden_size=args.hidden_size,
                num_layers=1,
                bias=True,
                batch_first=True,
                dropout=args.dropout,
                bidirectional=True
            )

    def embed_lineage(self, lineage_batch: List[List[int]]) -> torch.FloatTensor:
        """
        Embeds a batch of lineages.

        :param lineage_batch: A list of list of taxonomy indices representing organism lineages.
        :return: A PyTorch FloatTensor containing lineage embeddings.
        """
        # Embed taxon
        if self.lineage_embedding_type == 'taxon_only':
            taxons = torch.LongTensor([lineage[-1] for lineage in lineage_batch]).to(self.device)
            taxon_embedding = self.taxon_embedder(taxons)

            return taxon_embedding

        # Determine lengths
        lengths = torch.FloatTensor([len(lineage) for lineage in lineage_batch]).to(self.device)

        # Tensorize lineages
        lineage_batch = [torch.LongTensor(lineage).to(self.device) for lineage in lineage_batch]

        # Pad lineages
        lineage_batch = torch.nn.utils.rnn.pad_sequence(lineage_batch, batch_first=True, padding_value=0.0)

        # Embed lineages
        lineage_embeddings = self.taxon_embedder(lineage_batch)

        # Post-process lineage embeddings
        if self.lineage_embedding_type in ['average_lineage', 'sum_lineage']:
            sum_lineage_embedding = torch.sum(lineage_embeddings, dim=1)

            if self.lineage_embedding_type == 'sum_lineage':
                return sum_lineage_embedding

            average_lineage_embedding = sum_lineage_embedding / lengths.unsqueeze(dim=1)

            return average_lineage_embedding
        elif self.lineage_embedding_type == 'max_lineage':
            max_lineage_embedding, _ = torch.max(lineage_embeddings, dim=1)

            return max_lineage_embedding
        elif self.lineage_embedding_type == 'rnn_lineage':
            # Pack lineage embeddings
            lineage_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
                input=lineage_embeddings,
                lengths=lengths,
                batch_first=True,
                enforce_sorted=False
            )

            # Run RNN
            _, (hidden, _) = self.taxon_rnn(lineage_embeddings)
            rnn_lineage_embedding = hidden.sum(dim=0)

            return rnn_lineage_embedding
        else:
            raise ValueError(f'Lineage embedding type "{self.lineage_embedding_type}" not supported.')

    def featurize(self,
                  batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                  features_batch: List[np.ndarray] = None,
                  lineage_batch: List[List[int]] = None) -> torch.FloatTensor:
        """
        Computes feature vectors of the input by running the model except for the last layer.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :param lineage_batch: A list of list of taxonomy indices representing organism lineages.
        :return: The feature vectors computed by the :class:`MoleculeModel`.
        """
        return self.readout[:-1](self.encoder(batch, features_batch, lineage_batch))

    def forward(self,
                batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                lineage_batch: List[List[int]] = None) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of SMILES, a list of RDKit molecules, or a
                      :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional features.
        :param lineage_batch: A list of list of taxonomy indices representing organism lineages.
        :return: The output of the :class:`MoleculeModel`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        if lineage_batch is not None:
            lineage_batch = self.embed_lineage(lineage_batch)

        if self.featurizer:
            return self.featurize(batch, features_batch, lineage_batch)

        encoded = self.encoder(batch, features_batch, lineage_batch)
        readout = self.readout(encoded)
        output = self.combine(encoded, readout)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output

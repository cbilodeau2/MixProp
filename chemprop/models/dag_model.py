"""Contains a model which can make predictions on the nodes of a DAG."""
import torch.nn as nn


class DAGModel(nn.Module):
    def __init__(self):
        super(DAGModel, self).__init__()

"""Calibrations classification model predictions.

Currently uses temperature scaling from https://arxiv.org/pdf/1706.04599.pdf
https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
"""
import logging

import torch
from torch import nn, optim

from chemprop.data import MoleculeDataLoader
from chemprop.models import MoleculeModel
from chemprop.train.predict import predict


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.

    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins: int = 15):
        """
        :param n_bins: number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits: torch.FloatTensor, mask: torch.BoolTensor, targets: torch.FloatTensor) -> float:
        probs = torch.sigmoid(logits)
        predictions = torch.round(probs).byte()
        confidences = torch.where(predictions, probs, 1 - probs)
        accuracies = predictions.eq(targets)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item()) * mask
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()


def fit_temperature(model: MoleculeModel,
                    data_loader: MoleculeDataLoader,
                    logger: logging.Logger = None) -> None:
    """
    Fits the temperature parameter to calibration model predictions.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param logger: A logger for recording output.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    model.train()
    device = model.temperatures.device
    nll_criterion = nn.BCEWithLogitsLoss(reduction='none')
    ece_criterion = ECELoss()

    def compute_nll(preds: torch.FloatTensor,
                    mask: torch.FloatTensor,
                    targets: torch.FloatTensor) -> torch.FloatTensor:
        loss = nll_criterion(preds, targets) * mask
        loss = loss.sum() / mask.sum()

        return loss

    # Get predictions and targets
    logits = predict(model=model, data_loader=data_loader, eval_mode=False)
    logits = torch.FloatTensor(logits).to(device)
    targets = data_loader.targets
    mask = torch.Tensor([[x is not None for x in t] for t in targets]).to(device)
    mask_bool = mask.bool()
    targets = torch.Tensor([[0 if x is None else x for x in t] for t in targets]).to(device)

    # Compute ECE prior to training
    ece_before = ece_criterion(logits, mask_bool, targets)
    nll_before = compute_nll(logits, mask, targets)

    # Fit temperatures
    optimizer = optim.LBFGS([model.temperatures], lr=0.01, max_iter=50)

    def evaluate():
        loss = compute_nll(model.temperature_scale(logits), mask, targets)
        loss.backward()
        return loss

    optimizer.step(evaluate)

    preds = model.temperature_scale(logits)
    nll_after = compute_nll(preds, mask, targets)
    ece_after = ece_criterion(preds, mask_bool, targets)

    info(f'ECE before calibration = {ece_before:.6f}')
    info(f'ECE after calibration = {ece_after:.6f}')
    info(f'NLL before calibration = {nll_before:.6f}')
    info(f'NLL after calibration = {nll_after:.6f}')
    debug(f'Optimal temperatures = {model.temperatures.tolist()}')

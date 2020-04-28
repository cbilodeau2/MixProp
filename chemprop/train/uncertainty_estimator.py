from argparse import Namespace
import GPy
import heapq
import numpy as np
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
import torch.nn as nn
from typing import Any, Callable, List, Tuple

from .predict import predict
from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.features import morgan_binary_features_generator as morgan


class UncertaintyEstimator:
    """
    An UncertaintyEstimator calculates uncertainty when passed a model.
    Certain UncertaintyEstimators also augment the model and alter prediction
    values. Note that many UncertaintyEstimators compute unscaled uncertainty
    values. These are only meaningful relative to one another.
    """
    def __init__(self,
                 train_data: MoleculeDataset,
                 val_data: MoleculeDataset,
                 test_data: MoleculeDataset,
                 scaler: StandardScaler,
                 args: Namespace):
        """
        Constructs an UncertaintyEstimator.

        :param train_data: The data a model was trained on.
        :param val_data: The validation/supplementary data for a model.
        :param test_data: The data to test the model with.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.scaler = scaler
        self.args = args

    def process_model(self, model: nn.Module):
        """Perform initialization using model and prior data.

        :param model: The model to learn the uncertainty of.
        """
        pass

    def compute_uncertainty(self,
                            val_predictions: np.ndarray,
                            test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute uncertainty on self.val_data and self.test_data predictions.

        :param val_predictions: The predictions made on self.val_data.
        :param test_predictions: The predictions made on self.test_data.
        :return: Validation set predictions, validation set uncertainty,
                 test set predictions, and test set uncertainty.
        """
        pass

    def _scale_uncertainty(self, uncertainty: float) -> float:
        """
        Rescale uncertainty estimates to account for scaled input.

        :param uncertainty: An unscaled uncertainty estimate.
        :return: A scaled uncertainty estimate.
        """
        return self.scaler.stds * uncertainty


class EnsembleEstimator(UncertaintyEstimator):
    """
    An EnsembleEstimator trains a collection of models.
    Each model is exposed to all training data.

    On any input, a single prediction is calculated by taking the mean of
    model outputs. Reported uncertainty is the variance of outputs.
    """
    def __init__(self,
                 train_data: MoleculeDataset,
                 val_data: MoleculeDataset,
                 test_data: MoleculeDataset,
                 scaler: StandardScaler,
                 args: Namespace):
        super().__init__(train_data, val_data, test_data, scaler, args)
        self.all_val_preds = None
        self.all_test_preds = None

    def process_model(self, model: nn.Module):
        val_preds = predict(
            model=model,
            data=self.val_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
        )

        test_preds = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
        )

        reshaped_val_preds = np.array(val_preds).reshape(
            (len(self.val_data.smiles()), self.args.num_tasks, 1))
        if self.all_val_preds is not None:
            self.all_val_preds = np.concatenate(
                (self.all_val_preds, reshaped_val_preds), axis=2)
        else:
            self.all_val_preds = reshaped_val_preds

        reshaped_test_preds = np.array(test_preds).reshape(
            (len(self.test_data.smiles()), self.args.num_tasks, 1))
        if self.all_test_preds is not None:
            self.all_test_preds = np.concatenate(
                (self.all_test_preds, reshaped_test_preds), axis=2)
        else:
            self.all_test_preds = reshaped_test_preds

    def compute_uncertainty(self,
                            val_predictions: np.ndarray,
                            test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        val_uncertainty = np.sqrt(np.var(self.all_val_preds, axis=2))
        test_uncertainty = np.sqrt(np.var(self.all_test_preds, axis=2))

        return (val_predictions,
                val_uncertainty,
                test_predictions,
                test_uncertainty)


class BootstrapEstimator(EnsembleEstimator):
    """
    A BootstrapEstimator trains a collection of models.
    Each model is exposed to only a subset of training data.

    On any input, a single prediction is calculated by taking the mean of
    model outputs. Reported uncertainty is the variance of outputs.
    """
    def __init__(self,
                 train_data: MoleculeDataset,
                 val_data: MoleculeDataset,
                 test_data: MoleculeDataset,
                 scaler: StandardScaler,
                 args: Namespace):
        super().__init__(train_data, val_data, test_data, scaler, args)


class SnapshotEstimator(EnsembleEstimator):
    """
    A SnapshotEstimator trains a collection of models.
    Each model is produced by storing a single NN's
    weight at a different epochs in training.

    On any input, a single prediction is calculated by taking the mean of
    model outputs. Reported uncertainty is the variance of outputs.
    """
    def __init__(self,
                 train_data: MoleculeDataset,
                 val_data: MoleculeDataset,
                 test_data: MoleculeDataset,
                 scaler: StandardScaler,
                 args: Namespace):
        super().__init__(train_data, val_data, test_data, scaler, args)


class DropoutEstimator(EnsembleEstimator):
    """
    A DropoutEstimator trains a collection of models.
    The prediction of each 'model' is calculating by dropping out a random
    subset of nodes from a single NN.

    On any input, a single prediction is calculated by taking the mean of
    model outputs. Reported uncertainty is the variance of outputs.
    """
    def __init__(self,
                 train_data: MoleculeDataset,
                 val_data: MoleculeDataset,
                 test_data: MoleculeDataset,
                 scaler: StandardScaler,
                 args: Namespace):
        super().__init__(train_data, val_data, test_data, scaler, args)


class MVEEstimator(UncertaintyEstimator):
    """
    An MVEEstimator alters NN structure to produce twice as many outputs.
    Half correspond to predicted labels and half correspond to uncertainties.
    """
    def __init__(self,
                 train_data: MoleculeDataset,
                 val_data: MoleculeDataset,
                 test_data: MoleculeDataset,
                 scaler: StandardScaler,
                 args: Namespace):
        super().__init__(train_data, val_data, test_data, scaler, args)

        self.sum_val_uncertainty = np.zeros(
            (len(val_data.smiles()), args.num_tasks))

        self.sum_test_uncertainty = np.zeros(
            (len(test_data.smiles()), args.num_tasks))

    def process_model(self, model: nn.Module):
        val_preds, val_uncertainty = predict(
            model=model,
            data=self.val_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            uncertainty=True
        )

        if len(val_preds) != 0:
            self.sum_val_uncertainty += np.array(val_uncertainty).clip(min=0)

        test_preds, test_uncertainty = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=self.scaler,
            uncertainty=True
        )

        if len(test_preds) != 0:
            self.sum_test_uncertainty += np.array(test_uncertainty).clip(min=0)

    def compute_uncertainty(self,
                            val_predictions: np.ndarray,
                            test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (val_predictions,
                np.sqrt(self.sum_val_uncertainty / self.args.ensemble_size),
                test_predictions,
                np.sqrt(self.sum_test_uncertainty / self.args.ensemble_size))


class ExposureEstimator(UncertaintyEstimator):
    """
    An ExposureEstimator drops the output layer
    of the provided model after training.

    The "exposed" final hidden-layer is used to calculate uncertainty.
    """
    def __init__(self,
                 train_data: MoleculeDataset,
                 val_data: MoleculeDataset,
                 test_data: MoleculeDataset,
                 scaler: StandardScaler,
                 args: Namespace):
        super().__init__(train_data, val_data, test_data, scaler, args)

        self.sum_last_hidden_train = np.zeros(
            (len(self.train_data.smiles()), self.args.last_hidden_size))

        self.sum_last_hidden_val = np.zeros(
            (len(self.val_data.smiles()), self.args.last_hidden_size))

        self.sum_last_hidden_test = np.zeros(
            (len(self.test_data.smiles()), self.args.last_hidden_size))

    def process_model(self, model: nn.Module):
        model.eval()
        model.use_last_hidden = False

        last_hidden_train = predict(
            model=model,
            data=self.train_data,
            batch_size=self.args.batch_size,
            scaler=None
        )

        self.sum_last_hidden_train += np.array(last_hidden_train)

        last_hidden_val = predict(
            model=model,
            data=self.val_data,
            batch_size=self.args.batch_size,
            scaler=None
        )

        self.sum_last_hidden_val += np.array(last_hidden_val)

        last_hidden_test = predict(
            model=model,
            data=self.test_data,
            batch_size=self.args.batch_size,
            scaler=None
        )

        self.sum_last_hidden_test += np.array(last_hidden_test)

    def _compute_hidden_vals(self):
        ensemble_size = self.args.ensemble_size
        avg_last_hidden_train = self.sum_last_hidden_train / ensemble_size
        avg_last_hidden_val = self.sum_last_hidden_val / ensemble_size
        avg_last_hidden_test = self.sum_last_hidden_test / ensemble_size

        return avg_last_hidden_train, avg_last_hidden_val, avg_last_hidden_test


class GaussianProcessEstimator(ExposureEstimator):
    """
    A GaussianProcessEstimator trains a Gaussian process to
    operate on data transformed by the provided model.

    Uncertainty and predictions are calculated using
    the output of the Gaussian process.
    """
    def compute_uncertainty(self,
                            val_predictions: np.ndarray,
                            test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (_,
         avg_last_hidden_val,
         avg_last_hidden_test) = self._compute_hidden_vals()

        val_predictions = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))
        val_uncertainty = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_uncertainty = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        transformed_val = self.scaler.transform(
            np.array(self.val_data.targets()))

        for task in range(self.args.num_tasks):
            kernel = GPy.kern.Linear(input_dim=self.args.last_hidden_size)
            gaussian = GPy.models.SparseGPRegression(
                avg_last_hidden_val,
                transformed_val[:, task:task + 1], kernel)
            gaussian.optimize()

            avg_val_preds, avg_val_var = gaussian.predict(
                avg_last_hidden_val)

            val_predictions[:, task:task + 1] = avg_val_preds
            val_uncertainty[:, task:task + 1] = np.sqrt(avg_val_var)

            avg_test_preds, avg_test_var = gaussian.predict(
                avg_last_hidden_test)

            test_predictions[:, task:task + 1] = avg_test_preds
            test_uncertainty[:, task:task + 1] = np.sqrt(avg_test_var)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_uncertainty(val_uncertainty),
                test_predictions, self._scale_uncertainty(test_uncertainty))


class FPGaussianProcessEstimator(UncertaintyEstimator):
    """
    An FPGaussianProcessEstimator trains a Gaussian process on the
    morgan fingerprints of provided training data.

    Uncertainty and predictions are calculated using
    the output of the Gaussian process.
    """
    def compute_uncertainty(self,
                            val_predictions: np.ndarray,
                            test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_smiles = self.train_data.smiles()
        val_smiles = self.val_data.smiles()
        test_smiles = self.test_data.smiles()

        # Train targets are already scaled.
        scaled_train_targets = np.array(self.train_data.targets())

        train_fps = np.array([morgan(s) for s in train_smiles])
        val_fps = np.array([morgan(s) for s in val_smiles])
        test_fps = np.array([morgan(s) for s in test_smiles])

        val_predictions = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))
        val_uncertainty = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_uncertainty = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        for task in range(self.args.num_tasks):
            kernel = GPy.kern.Linear(input_dim=train_fps.shape[1])
            gaussian = GPy.models.SparseGPRegression(
                train_fps,
                scaled_train_targets[:, task:task + 1], kernel)
            gaussian.optimize()

            val_preds, val_var = gaussian.predict(
                val_fps)

            val_predictions[:, task:task + 1] = val_preds
            val_uncertainty[:, task:task + 1] = np.sqrt(val_var)

            test_preds, test_var = gaussian.predict(
                test_fps)

            test_predictions[:, task:task + 1] = test_preds
            test_uncertainty[:, task:task + 1] = np.sqrt(test_var)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_uncertainty(val_uncertainty),
                test_predictions, self._scale_uncertainty(test_uncertainty))


class RandomForestEstimator(ExposureEstimator):
    def compute_uncertainty(self,
                            val_predictions: np.ndarray,
                            test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        A RandomForestEstimator trains a random forest to
        operate on data transformed by the provided model.

        Predictions are calculated using the output of the random forest.
        Reported uncertainty is the variance of trees in the forest.
        """
        (_,
         avg_last_hidden_val,
         avg_last_hidden_test) = self._compute_hidden_vals()

        val_predictions = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))
        val_uncertainty = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_uncertainty = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        transformed_val = self.scaler.transform(
            np.array(self.val_data.targets()))

        n_trees = 128
        for task in range(self.args.num_tasks):
            forest = RandomForestRegressor(n_estimators=n_trees)
            forest.fit(avg_last_hidden_val, transformed_val[:, task])

            avg_val_preds = forest.predict(avg_last_hidden_val)
            val_predictions[:, task] = avg_val_preds

            individual_val_predictions = np.array([estimator.predict(
                avg_last_hidden_val) for estimator in forest.estimators_])
            val_uncertainty[:, task] = np.std(individual_val_predictions,
                                              axis=0)

            avg_test_preds = forest.predict(avg_last_hidden_test)
            test_predictions[:, task] = avg_test_preds

            individual_test_predictions = np.array([estimator.predict(
                avg_last_hidden_test) for estimator in forest.estimators_])
            test_uncertainty[:, task] = np.std(individual_test_predictions,
                                               axis=0)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_uncertainty(val_uncertainty),
                test_predictions, self._scale_uncertainty(test_uncertainty))


class FPRandomForestEstimator(UncertaintyEstimator):
    """
    An FPRandomForestEstimator trains a random forest on the
    morgan fingerprints of provided training data.

    Predictions are calculated using the output of the random forest.
    Reported uncertainty is the variance of trees in the forest.
    """
    def compute_uncertainty(self,
                            val_predictions: np.ndarray,
                            test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_smiles = self.train_data.smiles()
        val_smiles = self.val_data.smiles()
        test_smiles = self.test_data.smiles()

        # Train targets are already scaled.
        scaled_train_targets = np.array(self.train_data.targets())

        train_fps = np.array([morgan(s) for s in train_smiles])
        val_fps = np.array([morgan(s) for s in val_smiles])
        test_fps = np.array([morgan(s) for s in test_smiles])

        val_predictions = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))
        val_uncertainty = np.ndarray(
            shape=(len(self.val_data.smiles()), self.args.num_tasks))

        test_predictions = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))
        test_uncertainty = np.ndarray(
            shape=(len(self.test_data.smiles()), self.args.num_tasks))

        n_trees = 128
        for task in range(self.args.num_tasks):
            forest = RandomForestRegressor(n_estimators=n_trees)
            forest.fit(train_fps, scaled_train_targets[:, task])

            avg_val_preds = forest.predict(val_fps)
            val_predictions[:, task] = avg_val_preds

            individual_val_predictions = np.array([estimator.predict(
                val_fps) for estimator in forest.estimators_])
            val_uncertainty[:, task] = np.std(individual_val_predictions,
                                              axis=0)

            avg_test_preds = forest.predict(test_fps)
            test_predictions[:, task] = avg_test_preds

            individual_test_predictions = np.array([estimator.predict(
                test_fps) for estimator in forest.estimators_])
            test_uncertainty[:, task] = np.std(individual_test_predictions,
                                               axis=0)

        val_predictions = self.scaler.inverse_transform(val_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        return (val_predictions, self._scale_uncertainty(val_uncertainty),
                test_predictions, self._scale_uncertainty(test_uncertainty))


class LatentSpaceEstimator(ExposureEstimator):
    """
    A LatentSpaceEstimator uses the latent space distance between
    a molecule and its kNN in the training set to calculate uncertainty.
    """
    def compute_uncertainty(self,
                            val_predictions: np.ndarray,
                            test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (avg_last_hidden_train,
         avg_last_hidden_val,
         avg_last_hidden_test) = self._compute_hidden_vals()

        val_uncertainty = np.zeros((len(self.val_data.smiles()),
                                   self.args.num_tasks))
        test_uncertainty = np.zeros((len(self.test_data.smiles()),
                                    self.args.num_tasks))

        for val_input in range(len(avg_last_hidden_val)):
            distances = np.zeros(len(avg_last_hidden_train))
            for train_input in range(len(avg_last_hidden_train)):
                difference = avg_last_hidden_val[
                    val_input] - avg_last_hidden_train[train_input]
                distances[train_input] = np.sqrt(
                    np.sum(difference * difference))

            val_uncertainty[val_input, :] = kNN(distances, 8)

        for test_input in range(len(avg_last_hidden_test)):
            distances = np.zeros(len(avg_last_hidden_train))
            for train_input in range(len(avg_last_hidden_train)):
                difference = avg_last_hidden_test[
                    test_input] - avg_last_hidden_train[train_input]
                distances[train_input] = np.sqrt(
                    np.sum(difference * difference))

            test_uncertainty[test_input, :] = kNN(distances, 8)

        return (val_predictions,
                val_uncertainty,
                test_predictions,
                test_uncertainty)


class TanimotoEstimator(UncertaintyEstimator):
    """
    A TanimotoEstimator uses the tanimoto distance between
    a molecule and its kNN in the training set to calculate uncertainty.
    """
    def compute_uncertainty(self,
                            val_predictions: np.ndarray,
                            test_predictions: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_smiles = self.train_data.smiles()
        val_smiles = self.val_data.smiles()
        test_smiles = self.test_data.smiles()

        val_uncertainty = np.ndarray(
            shape=(len(val_smiles), self.args.num_tasks))
        test_uncertainty = np.ndarray(
            shape=(len(test_smiles), self.args.num_tasks))

        train_smiles_sfp = [morgan(s) for s in train_smiles]

        for i in range(len(val_smiles)):
            val_uncertainty[i, :] = np.ones((self.args.num_tasks)) * tanimoto(
                val_smiles[i], train_smiles_sfp, lambda x: kNN(x, 8))

        for i in range(len(test_smiles)):
            test_uncertainty[i, :] = np.ones((self.args.num_tasks)) * tanimoto(
                test_smiles[i], train_smiles_sfp, lambda x: kNN(x, 8))

        return (val_predictions,
                val_uncertainty,
                test_predictions,
                test_uncertainty)


def tanimoto(smile: str, train_smiles_sfp: np.ndarray, operation: Callable) \
        -> Any:
    """
    Computes the tanimoto distances between a
    molecule and elements of the training set.

    :param smile: The SMILES string of the molecule of interest.
    :param train_smiles_sfp: The fingerprints of training set elements.
    :param operation: Some function used to process computed distances.
    """
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
    fp = morgan(smiles)
    tanimoto_distance = []

    for sfp in train_smiles_sfp:
        tsim = np.dot(fp, sfp) / (fp.sum() +
                                  sfp.sum() - np.dot(fp, sfp))
        tanimoto_distance.append(-np.log2(max(0.0001, tsim)))

    return operation(tanimoto_distance)


def kNN(distances: List[float], k: int):
    """
    Computes k-nearest-neighbors.

    :param distances: A collection of distances from some point of interest.
    :param k: The number of neighbors to average over.
    :return: The mean of the smallest 'k' values in 'distances'.
    """
    return sum(heapq.nsmallest(k, distances))/k


def uncertainty_estimator_builder(uncertainty_method: str) \
        -> UncertaintyEstimator:
    """
    Maps strings to UncertaintyEstimators for easy command line arguments.

    :param uncertainty_method: The name of an UncertaintyEstimator.
    :return: An UncertaintyEstimator object of the specified type.
    """
    return {
        'ensemble': EnsembleEstimator,
        'bootstrap': BootstrapEstimator,
        'snapshot': SnapshotEstimator,
        'dropout': DropoutEstimator,
        'mve': MVEEstimator,
        'gaussian': GaussianProcessEstimator,
        'random_forest': RandomForestEstimator,
        'latent_space': LatentSpaceEstimator,
        'tanimoto': TanimotoEstimator,
        'fp_random_forest': FPRandomForestEstimator,
        'fp_gaussian': FPGaussianProcessEstimator
    }[uncertainty_method]

from argparse import Namespace
from scipy import stats
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize


class EvaluationMethod:
    def __init__(self):
        self.name = None

    def evaluate(self, data):
        pass

    def visualize(self, task, data):
        evaluation = self.evaluate(data)

        sns.set()

        self._visualize(task, evaluation)


class Cutoffs(EvaluationMethod):
    def __init__(self):
        self.name = 'cutoff'

    def evaluate(self, data):
        cutoff = []
        rmse = []
        ideal_rmse = []

        square_error = [set_['error']**2 for set_ in data['sets_by_uncertainty']]
        ideal_square_error = [set_['error']**2 for set_ in data['sets_by_error']]

        total_square_error = np.sum(square_error)
        ideal_total_square_error = np.sum(ideal_square_error)

        for i in range(len(square_error)):
            cutoff.append(data['sets_by_uncertainty'][i]['uncertainty'])

            rmse.append(np.sqrt(total_square_error/len(square_error[i:])))
            total_square_error -= square_error[i]

            ideal_rmse.append(np.sqrt(ideal_total_square_error / len(square_error[i:])))
            ideal_total_square_error -= ideal_square_error[i]

        return {'cutoff': cutoff, 'rmse': rmse, 'ideal_rmse': ideal_rmse}

    def _visualize(self, task, evaluation):
        percentiles = np.linspace(0, 100, len(evaluation['rmse']))

        plt.plot(percentiles, evaluation['rmse'])
        plt.plot(percentiles, evaluation['ideal_rmse'])

        plt.xlabel('Percent of Data Discarded')
        plt.ylabel('RMSE')
        plt.legend(['uncertainty Discard', 'Ideal Discard'])
        plt.title(task)

        plt.show()


class Scatter(EvaluationMethod):
    def __init__(self):
        self.name = 'scatter'
        self.x_axis_label = 'uncertainty'
        self.y_axis_label = 'Error'

    def evaluate(self, data):
        uncertainty = [self._x_filter(set_['uncertainty'])
                      for set_ in data['sets_by_uncertainty']]
        error = [self._y_filter(set_['error'])
                 for set_ in data['sets_by_uncertainty']]

        slope, intercept, _, _, _ = stats.linregress(uncertainty, error)

        return {'uncertainty': uncertainty,
                'error': error,
                'best_fit_y': slope * np.array(uncertainty) + intercept}

    def _x_filter(self, x):
        return x

    def _y_filter(self, y):
        return y

    def _visualize(self, task, evaluation):
        plt.scatter(evaluation['uncertainty'], evaluation['error'], s=0.3)
        plt.plot(evaluation['uncertainty'], evaluation['best_fit_y'])

        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        plt.title(task)

        plt.show()


class AbsScatter(Scatter):
    def __init__(self):
        self.name = 'abs_scatter'
        self.x_axis_label = 'uncertainty'
        self.y_axis_label = 'Absolute Value of Error'

    def _y_filter(self, y):
        return np.abs(y)


class LogScatter(Scatter):
    def __init__(self):
        self.name = 'log_scatter'
        self.x_axis_label = 'Log uncertainty'
        self.y_axis_label = 'Log Absolute Value of Error'

    def _x_filter(self, x):
        return np.log(x)

    def _y_filter(self, y):
        return np.log(np.abs(y))


class Spearman(EvaluationMethod):
    def __init__(self):
        self.name = 'spearman'

    def evaluate(self, data):
        uncertainty = [set_['uncertainty']
                      for set_ in data['sets_by_uncertainty']]
        error = [set_['error']
                 for set_ in data['sets_by_uncertainty']]

        rho, p = stats.spearmanr(uncertainty, np.abs(error))

        return {'rho': rho, 'p': p}

    def _visualize(self, task, evaluation):
        print(task, '-', 'Spearman Rho:', evaluation['rho'])
        print(task, '-', 'Spearman p-value:', evaluation['p'])


class LogLikelihood(EvaluationMethod):
    def __init__(self):
        self.name = 'log_likelihood'

    def evaluate(self, data):
        log_likelihood = 0
        optimal_log_likelihood = 0
        for set_ in data['sets_by_uncertainty']:
            # Encourage small standard deviations.
            log_likelihood -= np.log(2 * np.pi * max(0.00001, set_['uncertainty']**2)) / 2
            optimal_log_likelihood -= np.log(2 * np.pi * set_['error']**2) / 2

            # Penalize for large error.
            log_likelihood -= set_['error']**2/(2 * max(0.00001, set_['uncertainty']**2))
            optimal_log_likelihood -= 1 / 2 # set_['error']**2/(2 * set_['error']**2)

        return {'log_likelihood': log_likelihood,
                'optimal_log_likelihood': optimal_log_likelihood,
                'average_log_likelihood': log_likelihood / len(data['sets_by_uncertainty']),
                'average_optimal_log_likelihood': optimal_log_likelihood / len(data['sets_by_uncertainty'])}

    def _visualize(self, task, evaluation):
        print(task, '-', 'Sum of Log Likelihoods:', evaluation['log_likelihood'])


class CalibrationAUC(EvaluationMethod):
    def __init__(self):
        self.name = 'calibration_auc'

    def evaluate(self, data):
        standard_devs = [np.abs(set_['error'])/set_['uncertainty'] for set_ in data['sets_by_uncertainty']]
        probabilities = [2 * (stats.norm.cdf(standard_dev) - 0.5) for standard_dev in standard_devs]
        sorted_probabilities = sorted(probabilities)

        fraction_under_thresholds = []
        threshold = 0

        for i in range(len(sorted_probabilities)):
            while sorted_probabilities[i] > threshold:
                fraction_under_thresholds.append(i/len(sorted_probabilities))
                threshold += 0.001

        # Condition used 1.0001 to catch floating point errors.
        while threshold < 1.0001:
            fraction_under_thresholds.append(1)
            threshold += 0.001

        thresholds = np.linspace(0, 1, num=1001)
        miscalibration = [np.abs(fraction_under_thresholds[i] - thresholds[i]) for i in range(len(thresholds))]
        miscalibration_area = 0
        for i in range(1, 1001):
            miscalibration_area += np.average([miscalibration[i-1], miscalibration[i]]) * 0.001


        
        return {'fraction_under_thresholds': fraction_under_thresholds,
                'thresholds': thresholds,
                'miscalibration_area': miscalibration_area}
    
    def _visualize(self, task, evaluation):
        # Ideal curve.
        plt.plot(evaluation['thresholds'], evaluation['thresholds'])

        # True curve.
        plt.plot(evaluation['thresholds'], evaluation['fraction_under_thresholds'])
        print(task, '-', 'Miscalibration Area', evaluation['miscalibration_area'])

        plt.title(task)

        plt.show()


class Boxplot(EvaluationMethod):
    def __init__(self):
        self.name = 'boxplot'

    def evaluate(self, data):
        errors_by_uncertainty = [[] for _ in range(10)]

        min_uncertainty = data['sets_by_uncertainty'][-1]['uncertainty']
        max_uncertainty = data['sets_by_uncertainty'][0]['uncertainty']
        uncertainty_range = max_uncertainty - min_uncertainty

        for pair in data['sets_by_uncertainty']:
            errors_by_uncertainty[min(
                int((pair['uncertainty'] - min_uncertainty)//(uncertainty_range / 10)),
                9)].append(pair['error'])

        return {'errors_by_uncertainty': errors_by_uncertainty,
                'min_uncertainty': min_uncertainty,
                'max_uncertainty': max_uncertainty,
                'data': data}

    def _visualize(self, task, evaluation):
        errors_by_uncertainty = evaluation['errors_by_uncertainty']
        x_vals = list(np.linspace(evaluation['min_uncertainty'],
                                  evaluation['max_uncertainty'],
                                  num=len(errors_by_uncertainty),
                                  endpoint=False) + (evaluation['max_uncertainty'] - evaluation['min_uncertainty'])/(len(errors_by_uncertainty) * 2))
        plt.boxplot(errors_by_uncertainty, positions=x_vals, widths=(0.02))

        names = (
            f'{len(errors_by_uncertainty[i])} points' for i in range(10))
        plt.xticks(x_vals, names)
        plt.xlim((evaluation['min_uncertainty'], evaluation['max_uncertainty']))
        Scatter().visualize(task, evaluation['data'])


class UncertaintyEvaluator:
    methods = [Cutoffs(), AbsScatter(), LogScatter(), Spearman(), LogLikelihood(), Boxplot(), CalibrationAUC()]

    @staticmethod
    def save(val_predictions, val_targets, val_uncertainty, test_predictions, test_targets, test_uncertainty, args):
        f = open(args.save_uncertainty, 'w+')

        val_data = UncertaintyEvaluator._log(val_predictions, val_targets, val_uncertainty, args)
        test_data = UncertaintyEvaluator._log(test_predictions, test_targets, test_uncertainty, args)

        json.dump({'validation': val_data, 'test': test_data}, f)
        f.close()

    @staticmethod
    def _log(predictions, targets, uncertainty, args):
        log = {}

        # Loop through all subtasks.    
        for task in range(args.num_tasks):
            mask = targets[:, task] != None

            task_predictions = np.extract(mask, predictions[:, task])
            task_targets = np.extract(mask, targets[:, task])
            task_uncertainty = np.extract(mask, uncertainty[:, task])
            task_error = list(task_predictions - task_targets)

            task_sets = [{'prediction': task_set[0],
                          'target': task_set[1],
                          'uncertainty': task_set[2],
                          'error': task_set[3]} for task_set in zip(
                                        task_predictions,
                                        task_targets,
                                        task_uncertainty,
                                        task_error)]

            sets_by_uncertainty = sorted(task_sets,
                                        key=lambda pair: pair['uncertainty'],
                                        reverse=True)

            sets_by_error = sorted(task_sets,
                                   key=lambda pair: np.abs(pair['error']),
                                   reverse=True)

            log[args.task_names[task]] = {
                'sets_by_uncertainty': sets_by_uncertainty,
                'sets_by_error': sets_by_error}

        return log

    @staticmethod
    def visualize(file_path, methods):
        f = open(file_path)
        log = json.load(f)['test']

        for task, data in log.items():
            for method in UncertaintyEvaluator.methods:
                if method.name in methods:
                    method.visualize(task, data)

        f.close()

    @staticmethod
    def evaluate(file_path, methods):
        f = open(file_path)
        log = json.load(f)['test']

        all_evaluations = {}
        for task, data in log.items():
            task_evaluations = {}
            for method in UncertaintyEvaluator.methods:
                if method.name in methods:
                    task_evaluations[method.name] = method.evaluate(data)
            all_evaluations[task] = task_evaluations

        f.close()

        return all_evaluations

    @staticmethod
    def calibrate(lambdas, beta_init, file_path):
        def objective_function(beta, uncertainty, errors, lambdas):
            # Construct prediction through lambdas and betas.
            pred_vars = np.zeros(len(uncertainty))
            
            for i in range(len(beta)):
                pred_vars += np.abs(beta[i]) * lambdas[i](uncertainty**2)
            pred_vars = np.clip(pred_vars, 0.001, None)
            costs = np.log(pred_vars) / 2 + errors**2 / (2 * pred_vars)

            return(np.sum(costs))
        
        def calibrate_sets(sets, sigmas, lambdas):
            calibrated_sets = []
            for set_ in sets:
                calibrated_set = set_.copy()
                calibrated_set['uncertainty'] = 0
                
                for i in range(len(sigmas)):
                    calibrated_set['uncertainty'] += sigmas[i] * lambdas[i](set_['uncertainty']**2)
                calibrated_sets.append(calibrated_set)
            return calibrated_sets

        f = open(file_path)
        full_log = json.load(f)
        val_log = full_log['validation']
        test_log = full_log['test']

        scaled_val_log = {}
        scaled_test_log = {}

        calibration_coefficients = {}
        for task in val_log:
            # Sample from validation data.
            sampled_data = val_log[task]['sets_by_error']

            # Calibrate based on sampled data.
            uncertainty = np.array([set_['uncertainty'] for set_ in sampled_data])
            errors = np.array([set_['error'] for set_ in sampled_data])

            result = minimize(objective_function, beta_init, args=(uncertainty, errors, lambdas),
                            method='BFGS', options={'maxiter': 500})
            
            calibration_coefficients[task] = np.abs(result.x)

            scaled_val_data = {}
            scaled_val_data['sets_by_error'] = calibrate_sets(val_log[task]['sets_by_error'], np.abs(result.x), lambdas)
            scaled_val_data['sets_by_uncertainty'] = calibrate_sets(val_log[task]['sets_by_uncertainty'], np.abs(result.x), lambdas)
            scaled_val_log[task] = scaled_val_data

            scaled_test_data = {}
            scaled_test_data['sets_by_error'] = calibrate_sets(test_log[task]['sets_by_error'], np.abs(result.x), lambdas)
            scaled_test_data['sets_by_uncertainty'] = calibrate_sets(test_log[task]['sets_by_uncertainty'], np.abs(result.x), lambdas)
            scaled_test_log[task] = scaled_test_data
        
        f.close()

        return {'validation': scaled_val_log, 'test': scaled_test_log}, calibration_coefficients

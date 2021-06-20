from math import sqrt
from typing import List, Dict, Any


class Accuracy:
    def __init__(self, mae, rmse, mape, smape, mase, ias):
        self.mae = mae
        self.rmse = rmse
        self.mape = mape
        self.smape = smape
        self.mase = mase
        self.ias = ias


class AccuracyCalculator:
    def __init__(self, accuracy_config: Dict[str, Any]):
        self.ias_quantile = float(accuracy_config["ias_quantile"])
        self.ias_sig_alpha = float(accuracy_config["ias_sig_alpha"])

    def compute_accuracy_metrics(self, real_counts: List[int], forecasts: List[int], forecasts_train: List[int]):
        return Accuracy(
            mae=AccuracyCalculator.mean_absolute_error(real_counts, forecasts),
            rmse=AccuracyCalculator.root_mean_squared_error(real_counts, forecasts),
            mape=AccuracyCalculator.root_mean_squared_error(real_counts, forecasts),
            smape=AccuracyCalculator.symmetric_mean_absolute_percentage_error(real_counts, forecasts),
            mase=AccuracyCalculator.mean_absolute_scaled_error(real_counts, forecasts, forecasts_train),
            ias=AccuracyCalculator.interval_accuracy_score(real_counts, forecasts, self.ias_quantile, self.ias_sig_alpha),
        )

    @staticmethod
    def mean_absolute_error(ys_real: List[int], ys_pred: List[int]) -> float:
        errors = [abs(y - y_hat) for y, y_hat in zip(ys_real, ys_pred)]
        return sum(errors) / len(errors)

    @staticmethod
    def root_mean_squared_error(ys_real: List[int], ys_pred: List[int]) -> float:
        errors = [(y - y_hat) ** 2 for y, y_hat in zip(ys_real, ys_pred)]
        return sqrt(sum(errors) / len(errors))

    @staticmethod
    def mean_absolute_percentage_error(ys_real: List[int], ys_pred: List[int]) -> float:
        errors = [abs(100 * (y - y_hat) / y_hat) for y, y_hat in zip(ys_real, ys_pred)]
        return sum(errors) / len(errors)

    @staticmethod
    def symmetric_mean_absolute_percentage_error(ys_real: List[int], ys_pred: List[int]) -> float:
        errors = [200 * abs(y - y_hat) / (y + y_hat) for y, y_hat in zip(ys_real, ys_pred)]
        return sum(errors) / len(errors)

    @staticmethod
    def mean_absolute_scaled_error(ys_real: List[int], ys_pred: List[int], ys_train_pred: List[int]) -> float:
        base = AccuracyCalculator.mean_absolute_error(ys_train_pred[1:], ys_train_pred[:-1])
        forecasts_error_mean = AccuracyCalculator.mean_absolute_error(ys_real, ys_pred)
        return forecasts_error_mean / base

    @staticmethod
    def interval_accuracy_score(ys_real: List[int], ys_pred: List[int], quantile: float, sig_alpha: float) -> float:
        mean = sum(ys_real) / len(ys_real)
        st_dev = sqrt(sum([(y - mean) ** 2 for y in ys_real]) / (len(ys_real) - 1))
        lb = mean - quantile * st_dev
        ub = mean + quantile * st_dev

        def compute_single_ias(x: float, lb: float, ub: float, alpha: float) -> float:
            lb_penalty = 1 if x < lb else 0
            lb_penalty = lb_penalty * (2 / alpha) * (lb - x)
            ub_penalty = 1 if ub < x else 0
            ub_penalty = ub_penalty * (2 / alpha) * (x - ub)
            return (ub - lb) + lb_penalty + ub_penalty

        return sum([compute_single_ias(y_hat, lb, ub, sig_alpha) for y_hat in ys_pred])

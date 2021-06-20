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
            mae=AccuracyCalculator.compute_mae(real_counts, forecasts),
            rmse=AccuracyCalculator.compute_rmse(real_counts, forecasts),
            mape=AccuracyCalculator.compute_rmse(real_counts, forecasts),
            smape=AccuracyCalculator.compute_smape(real_counts, forecasts),
            mase=AccuracyCalculator.compute_mase(real_counts, forecasts, forecasts_train),
            ias=AccuracyCalculator.compute_ias(real_counts, forecasts, self.ias_quantile, self.ias_sig_alpha),
        )

    @staticmethod
    def compute_mae(ys_real: List[int], ys_pred: List[int]) -> float:
        errors = [abs(y - y_hat) for y, y_hat in zip(ys_real, ys_pred)]
        return sum(errors) / len(errors)

    @staticmethod
    def compute_rmse(ys_real: List[int], ys_pred: List[int]) -> float:
        errors = [(y - y_hat) ** 2 for y, y_hat in zip(ys_real, ys_pred)]
        return sqrt(sum(errors) / len(errors))

    @staticmethod
    def compute_mape(ys_real: List[int], ys_pred: List[int]) -> float:
        errors = [abs(100 * (y - y_hat) / y_hat) for y, y_hat in zip(ys_real, ys_pred)]
        return sum(errors) / len(errors)

    @staticmethod
    def compute_smape(ys_real: List[int], ys_pred: List[int]) -> float:
        errors = [200 * abs(y - y_hat) / (y + y_hat) for y, y_hat in zip(ys_real, ys_pred)]
        return sum(errors) / len(errors)

    @staticmethod
    def compute_mase(ys_real: List[int], ys_pred: List[int], ys_train_pred: List[int]) -> float:
        base = AccuracyCalculator.compute_mae(ys_train_pred[1:], ys_train_pred[:-1])
        forecasts_error_mean = AccuracyCalculator.compute_mae(ys_real, ys_pred)
        return forecasts_error_mean / base

    @staticmethod
    def compute_ias(ys_real: List[int], ys_pred: List[int], quantile: float, sig_alpha: float) -> float:
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

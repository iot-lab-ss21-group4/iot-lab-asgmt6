from typing import Dict

from util.accuracy import Accuracy
from util.forecast_model_select_strategy import ForecastModelSelectStrategy, SelectMostAccurateStrategy


class ForecastModelSelector:
    def __init__(self, strategy: ForecastModelSelectStrategy):
        self.strategy = strategy

    def select(self, model_accuracies: Dict[str, Accuracy]) -> str:
        return self.strategy.apply(model_accuracies)

    @staticmethod
    def build_from_config(strategy: str):
        if strategy == "select_most_accurate":
            return ForecastModelSelector(strategy=SelectMostAccurateStrategy())

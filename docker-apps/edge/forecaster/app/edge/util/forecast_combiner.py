from typing import Dict, Tuple

from edge.util.accuracy import Accuracy
from edge.util.forecast_combine_strategy import (
    ForecastCombineStrategy,
    SelectMostAccurateStrategy,
    WeightedCombinationStrategy,
    MajorityRuleStrategy,
)


class ForecastCombiner:
    def __init__(self, strategy: ForecastCombineStrategy):
        self.strategy = strategy

    def combine(self, model_results: Dict[str, Tuple[int, int, Accuracy]]) -> int:
        return self.strategy.apply(model_results)

    @staticmethod
    def build_from_config(strategy: str):
        if strategy == "select_most_accurate":
            return ForecastCombiner(strategy=SelectMostAccurateStrategy())
        if strategy == "weighted_combination":
            return ForecastCombiner(strategy=WeightedCombinationStrategy())
        if strategy == "majority_rule":
            return ForecastCombiner(strategy=MajorityRuleStrategy())

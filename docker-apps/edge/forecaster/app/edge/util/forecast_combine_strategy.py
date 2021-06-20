import operator
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Iterable

from edge.util.accuracy import Accuracy


class ForecastCombineStrategy(ABC):
    @abstractmethod
    def apply(self, model_results: Dict[str, Tuple[int, int, Accuracy]]) -> int:
        pass


class SelectMostAccurateStrategy(ForecastCombineStrategy):
    def apply(self, model_results: Dict[str, Tuple[int, int, Accuracy]]) -> int:
        scoring = {}
        for key in model_results.keys():
            scoring[key] = 0

        for model_metric_winner in list(self.find_best_per_metric(model_results)):
            scoring[model_metric_winner] = scoring[model_metric_winner] + 1

        winner = max(scoring.items(), key=operator.itemgetter(1))[0]
        return model_results[winner][1]

    def find_best_per_metric(self, model_results: Dict[str, Tuple[int, int, Accuracy]]) -> Iterable[str]:
        model_results_list = list(model_results.items())
        first_model, result = model_results_list[0]
        accuracy: Accuracy = result[2]
        # iterate over the accuracy fields using the first model_result as reference
        for accuracy_name in accuracy._fields:
            # init wih result of first model
            model_winner, best_accuracy = first_model, getattr(accuracy, accuracy_name)
            for i in range(1, len(model_results_list)):
                model, model_result = model_results_list[i]
                other_accuracy = getattr(model_result[2], accuracy_name)
                # check if better accuracy exists and overwrite
                if other_accuracy < best_accuracy:
                    model_winner = model
                    best_accuracy = other_accuracy
            yield model_winner


class WeightedCombinationStrategy(ForecastCombineStrategy):
    def apply(self, model_results: Dict[str, Tuple[int, int, Accuracy]]) -> int:
        forecasts = [result[1] for result in model_results]
        return round(sum(forecasts) / len(forecasts))


class MajorityRuleStrategy(ForecastCombineStrategy):
    def apply(self, model_results: Dict[str, Tuple[int, int, Accuracy]]) -> int:
        pass

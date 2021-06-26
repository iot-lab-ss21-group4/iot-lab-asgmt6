import operator
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple

from util.accuracy import Accuracy


class ForecastModelSelectStrategy(ABC):
    @abstractmethod
    def apply(self, model_results: Dict[str, Accuracy]) -> str:
        pass


class SelectMostAccurateStrategy(ForecastModelSelectStrategy):
    def apply(self, model_results: Dict[str, Accuracy]) -> str:
        scoring = {}
        for key in model_results.keys():
            scoring[key] = 0

        for model_metric_winner in list(self.find_best_per_metric(model_results)):
            scoring[model_metric_winner] = scoring[model_metric_winner] + 1

        winner = max(scoring.items(), key=operator.itemgetter(1))[0]
        return winner

    def find_best_per_metric(self, model_results: Dict[str, Accuracy]) -> Iterable[str]:
        model_results_list = list(model_results.items())
        model_1, accuracy_1 = model_results_list[0]
        # iterate over the accuracy fields using the first model_result as reference
        for accuracy_name in accuracy_1._fields:
            # init wih result of first model
            model_winner, best_accuracy = model_1, getattr(accuracy_1, accuracy_name)
            for i in range(1, len(model_results_list)):
                model_other, accuracy_other = model_results_list[i]
                other_accuracy = getattr(accuracy_other, accuracy_name)
                # check if better accuracy exists and overwrite
                if other_accuracy < best_accuracy:
                    model_winner = model_other
                    best_accuracy = other_accuracy
            yield model_winner

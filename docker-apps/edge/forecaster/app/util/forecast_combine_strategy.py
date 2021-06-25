import operator
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Tuple

from util.accuracy import Accuracy


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
    class Bucket:
        def __init__(self, fixpoint: int, bucket_length: int):
            self.fixpoint = fixpoint
            self.bucket_length = bucket_length
            self.elements = 0

        @property
        def min(self):
            return self.fixpoint - self.bucket_length

        @property
        def max(self):
            if self.bucket_length == 0:
                return self.fixpoint
            return self.fixpoint + self.bucket_length - 1

        def fill(self, point):
            # the < operator ensures that growing to the left starts earlier and avoids two buckets with same
            # number of elements this gives higher fixpoints/forecasts an advantage
            if self.fixpoint - self.bucket_length <= point < self.fixpoint + self.bucket_length:
                self.elements += 1

        def intersect(self, other):
            return other.min <= self.max and other.max <= self.max or self.min <= other.min <= self.max

    def apply(self, model_results: Dict[str, Tuple[int, int, Accuracy]]) -> int:
        forecasts = [model_result[1] for model_result in model_results.values()]
        for bucket_length in range(1, 45):
            buckets = self.init_buckets(bucket_length, forecasts)
            for f in forecasts:
                for b in buckets:
                    b.fill(f)
            max_elements = max(bucket.elements for bucket in buckets)
            winner_buckets = [bucket for bucket in buckets if bucket.elements == max_elements]
            if len(winner_buckets) == 1:
                return winner_buckets[0].fixpoint

        # TODO: use logger
        print("Error in strategy")
        return list(model_results.values())[0][1]

    def init_buckets(self, bucket_length: int, fixpoints: List[int]) -> List[Bucket]:
        return [MajorityRuleStrategy.Bucket(fp, bucket_length) for fp in fixpoints]

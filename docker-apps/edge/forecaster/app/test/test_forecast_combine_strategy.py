import unittest

from edge.util.accuracy import AccuracyCalculator
from edge.util.forecast_combine_strategy import SelectMostAccurateStrategy
from edge.util.forecast_combiner import ForecastCombiner


class TestForecastCombineStrategy(unittest.TestCase):

    def setUp(self):
        accCalculator = AccuracyCalculator()
        timestamp = 42
        real_counts = [1,1,1,2,3,4,5,6,7,8,7,8,9]
        self.forecast_model_1 = 8
        self.forecast_model_2 = 7
        forecasts_1 = [1,1,1,1,2,4,5,6,7,8,8,8,self.forecast_model_1]
        forecasts_2 = [1,2,3,4,5,6,7,7,7,7,7,7,self.forecast_model_2]
        self.accuracy_1 = accCalculator.compute_accuracy_metrics(
            real_counts=real_counts,
            forecasts=forecasts_1
        )
        self.accuracy_2 = accCalculator.compute_accuracy_metrics(
            real_counts=real_counts,
            forecasts=forecasts_2
        )
        self.model_1 = "better_model"
        self.model_2 = "weaker_model"
        self.model_results = {
            self.model_1: (timestamp, self.forecast_model_1, self.accuracy_1),
            self.model_2: (timestamp, self.forecast_model_2, self.accuracy_2)
        }


    def test_select_most_accurate_strategy(self):
        combiner = ForecastCombiner(SelectMostAccurateStrategy())
        y = combiner.combine(self.model_results)
        self.assertEqual(self.forecast_model_1, y)
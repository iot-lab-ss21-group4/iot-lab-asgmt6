from typing import Optional

import optuna
import pandas as pd
from iotlab_utils.data_manager import (
    DEFAULT_LAG_ORDER,
    DERIVATIVE_COLUMN,
    LAG_FEATURE_TEMPLATE,
    TIME_COLUMN,
    prepare_data_with_features,
)
from sklearn.linear_model import Lasso


class LrWrapper:
    def __init__(self, model: Lasso, lag_order: int, look_back_buffer: Optional[pd.DataFrame] = None):
        self.model = model
        self.lag_order = lag_order
        self.look_back_buffer: Optional[pd.DataFrame] = None
        if look_back_buffer is not None:
            self.update_look_back_buffer(look_back_buffer)

    @property
    def look_back_length(self) -> int:
        return self.lag_order

    def update_look_back_buffer(self, look_back_buffer: pd.DataFrame):
        assert look_back_buffer.shape[0] == self.look_back_length, "LR lookback buffer must be of length {}".format(
            self.look_back_length
        )
        self.look_back_buffer = look_back_buffer

    def forecast(self, ts: pd.DataFrame) -> pd.DataFrame:
        ts = pd.concat([self.look_back_buffer, ts], axis=0)
        y_column, x_columns, ts, useless_rows = prepare_data_with_features(
            ts, detailed_seasonality=False, lag_order=self.lag_order
        )

        for i in range(useless_rows, ts.shape[0]):
            pred_i = self.model.predict(ts[x_columns].iloc[i : i + 1])
            ts.loc[ts.index[i], y_column] = pred_i
            for lag in range(1, self.lag_order + 1):
                if i + lag >= ts.shape[0]:
                    break
                ts.loc[ts.index[i + lag], LAG_FEATURE_TEMPLATE.format(lag)] = pred_i
            for lag in range(1, 3):
                if i < 1 or i + lag >= ts.shape[0]:
                    break
                ts.loc[ts.index[i + lag], DERIVATIVE_COLUMN] = (
                    ts.loc[ts.index[i + lag - 1], y_column] - ts.loc[ts.index[i + lag - 2], y_column]
                ) / (ts.loc[ts.index[i + lag - 1], TIME_COLUMN] - ts.loc[ts.index[i + lag - 2], TIME_COLUMN])

        ts.loc[ts.index[useless_rows:], y_column] = ts.loc[ts.index[useless_rows:], y_column].round()
        return ts.loc[ts.index[self.look_back_length :], [TIME_COLUMN, y_column]]


def train(ts: pd.DataFrame) -> LrWrapper:
    lag_order = DEFAULT_LAG_ORDER
    y_column, x_columns, ts, useless_rows = prepare_data_with_features(ts, detailed_seasonality=False, lag_order=lag_order)
    ts = ts.iloc[useless_rows:]

    train_len = int(ts.shape[0] * 0.9)
    train_ts, test_ts = ts.iloc[:train_len], ts.iloc[train_len:]

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_uniform("alpha", 0.0, 1.0)
        forecast_model = Lasso(alpha=alpha)
        model_fit = forecast_model.fit(train_ts[x_columns].to_numpy(), train_ts[y_column].to_numpy())
        return model_fit.score(test_ts[x_columns].to_numpy(), test_ts[y_column].to_numpy())

    study = optuna.create_study(
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50, n_jobs=1)
    alpha = study.best_params["alpha"]
    forecast_model = Lasso(alpha=alpha)
    model_fit = forecast_model.fit(train_ts[x_columns].to_numpy(), train_ts[y_column].to_numpy())

    wrapped_model = LrWrapper(model_fit, lag_order)
    wrapped_model.update_look_back_buffer(ts.loc[ts.index[-wrapped_model.look_back_length :], [TIME_COLUMN, y_column]])
    return wrapped_model

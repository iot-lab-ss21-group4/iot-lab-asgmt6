import optuna
import pandas as pd
from iotlab_utils.data_manager import (
    prepare_data_with_features,
    DEFAULT_LAG_ORDER,
    LAG_FEATURE_TEMPLATE,
    DERIVATIVE_COLUMN,
    TIME_COLUMN,
)
from sklearn.linear_model import Lasso


class LrWrapper:
    def __init__(self, model: Lasso):
        self.model = model

    @property
    def look_back_length(self):
        # TODO: make this more configurable. LAG order depends on training
        return DEFAULT_LAG_ORDER

    def forecast(self, ts: pd.DataFrame) -> pd.DataFrame:
        y_column, x_columns, ts, useless_rows = prepare_data_with_features(ts, detailed_seasonality=False)

        for i in range(useless_rows, ts.shape[0]):
            pred_i = self.model.predict(ts[x_columns].iloc[i : i + 1])
            ts.loc[ts.index[i], y_column] = pred_i
            for lag in range(1, DEFAULT_LAG_ORDER + 1):
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
        return ts[[TIME_COLUMN, y_column]]


def train(ts: pd.DataFrame) -> LrWrapper:
    y_column, x_columns, ts, useless_rows = prepare_data_with_features(ts, detailed_seasonality=False)
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

    return LrWrapper(model_fit)

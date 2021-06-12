import time

import optuna
import pandas as pd
from iotlab_utils.data_manager import prepare_data_with_features
from sklearn.linear_model import Lasso


def train(data: pd.DataFrame) -> Lasso:
    y_column, x_columns, ts, useless_rows = prepare_data_with_features(data, detailed_seasonality=False)
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

    return model_fit


def forecast():
    ts = time.time()


def predict() -> str:
    pass


def periodic_forecast():
    pass

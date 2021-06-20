import threading
from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
from iotlab_utils.data_manager import (
    DEFAULT_FLOAT_TYPE,
    TIME_COLUMN,
    extract_features,
    prepare_data_with_features,
    regularize_data,
)
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

# TODO
# LAG and number of elements in times might not work together if times list is too small


class SARIMAXWrapper:
    def __init__(
        self,
        model: SARIMAXResultsWrapper,
        avg_dt: pd.DateOffset,
        exog_columns: List[str],
        look_back_buffer: Optional[pd.DataFrame] = None,
    ):
        self.model = model
        self.avg_dt = avg_dt
        self.exog_columns = exog_columns
        self.look_back_buffer: Optional[pd.DataFrame] = None
        if look_back_buffer is not None:
            self.update_look_back_buffer(look_back_buffer)

    @property
    def look_back_length(self) -> int:
        return 1

    def update_look_back_buffer(self, look_back_buffer: pd.DataFrame):
        assert look_back_buffer.shape[0] == self.look_back_length, "SARIMAX lookback buffer must be of length {}".format(
            self.look_back_length
        )
        if self.look_back_buffer is None:
            self.look_back_buffer = look_back_buffer

    def forecast(self, ts: pd.DataFrame) -> pd.DataFrame:
        y_column, _, ts, _ = prepare_data_with_features(
            ts, seasonality_features=False, detailed_seasonality=False, extra_features=False, lag_order=0
        )
        pred_time = ts.loc[ts.index[-1], TIME_COLUMN]
        # Uses freq and last_t to forecast beyond the last datapoint and
        # uses interpolation to predict exactly at the datapoint times.
        last_t = self.look_back_buffer[self.look_back_buffer.index[-1], TIME_COLUMN]
        last_y = self.look_back_buffer[self.look_back_buffer.index[-1], y_column]
        avg_dt_sec = self.avg_dt.delta.total_seconds()
        forecast_size = int(np.ceil((pred_time - last_t) / avg_dt_sec))
        times, ys = [last_t], [last_y] + [DEFAULT_FLOAT_TYPE()] * forecast_size
        for i in range(1, forecast_size + 1):
            times.append(last_t + int(np.round(i * avg_dt_sec)))
        regular_ts = pd.DataFrame(
            {TIME_COLUMN: pd.Series(times), y_column: pd.Series(ys, dtype=DEFAULT_FLOAT_TYPE)},
            columns=[TIME_COLUMN, y_column],
        )
        _, regular_ts, _ = extract_features(regular_ts, y_column, detailed_seasonality=False, extra_features=False)
        pred: pd.Series = self.model.forecast(forecast_size, exog=regular_ts.loc[regular_ts.index[1:], self.exog_columns])
        regular_ts.loc[regular_ts.index[1:], y_column] = pred.to_numpy()
        # Use regular time predictions and linear interpolation to estimate at the exact times.
        univariate_f = interp1d(regular_ts[TIME_COLUMN].to_numpy(), regular_ts[y_column].to_numpy())
        ts[y_column] = univariate_f(ts[TIME_COLUMN].to_numpy()).astype(DEFAULT_FLOAT_TYPE)
        ts[y_column] = ts[y_column].round()

        return ts.loc[ts.index, [TIME_COLUMN, y_column]]


def train(data: pd.DataFrame) -> SARIMAXWrapper:
    y_column, exog_columns, ts, _ = prepare_data_with_features(data, detailed_seasonality=False, extra_features=False)
    avg_dt, ts = regularize_data(ts, y_column)
    # Remove constant columns
    constant_columns = ~((ts != ts.iloc[0]).any())
    exog_columns_set = set(exog_columns)
    for col in exog_columns:
        if constant_columns[col]:
            ts.drop(columns=[col], inplace=True)
            exog_columns_set.remove(col)
    exog_columns = list(exog_columns_set)

    train_len = int(ts.shape[0] * 0.9)
    train_ts, test_ts = ts.iloc[:train_len], ts.iloc[train_len:]

    trial_dict = {}
    trial_lock = threading.RLock()

    def objective(trial: optuna.Trial) -> float:
        # past_days, past_weeks = 1, 0
        # past_day_offsets = np.zeros(4 * 7, dtype=np.int)
        # past_day_offsets[np.concatenate([np.arange(past_days), 7 * (1 + np.arange(past_weeks))])] = 1
        # past_day_offsets: List = past_day_offsets.tolist()
        # while len(past_day_offsets) > 0 and past_day_offsets[-1] == 0:
        #     past_day_offsets.pop()
        # P, D, Q, S = past_day_offsets, 0, past_day_offsets, 1440
        p, d = trial.suggest_int("p", 0, 5), trial.suggest_int("d", 0, 3)
        q = trial.suggest_int("q", 0 if p + d > 0 else 1, 5)
        with trial_lock:
            if (p, d, q) in trial_dict:
                return trial_dict[(p, d, q)]
        forecast_model = SARIMAX(train_ts[y_column], exog=train_ts[exog_columns], trend="c", order=(p, d, q))
        model_fit = forecast_model.fit(disp=False)
        test_pred = model_fit.forecast(test_ts.shape[0], exog=test_ts[exog_columns])
        loss = mean_squared_error(test_ts[y_column].to_numpy(), test_pred.to_numpy())
        # loss = model_fit.info_criteria("bic")
        with trial_lock:
            trial_dict[(p, d, q)] = loss
        return loss

    study = optuna.create_study(
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50, n_jobs=1)
    p, d, q = study.best_params["p"], study.best_params["d"], study.best_params["q"]
    forecast_model = SARIMAX(train_ts[y_column], exog=train_ts[exog_columns], trend="c", order=(p, d, q))
    model_fit = forecast_model.fit(disp=False)

    wrapped_model = SARIMAXWrapper(model_fit, avg_dt, exog_columns)
    wrapped_model.look_back_buffer(train_ts.loc[train_ts.index[-wrapped_model.look_back_length :], [TIME_COLUMN, y_column]])
    return wrapped_model

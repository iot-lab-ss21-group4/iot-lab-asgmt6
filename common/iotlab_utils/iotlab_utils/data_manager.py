from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

DEFAULT_FLOAT_TYPE = np.float32
TIME_COLUMN = "t"
UNIVARIATE_DATA_COLUMN = "count"
DT_COLUMN = "dT"
DERIVATIVE_COLUMN = "d_{}/dT".format(UNIVARIATE_DATA_COLUMN)
LAG_FEATURE_TEMPLATE = "lag{}_" + UNIVARIATE_DATA_COLUMN
# Lag order must be >= 1.
DEFAULT_LAG_ORDER = 2


def index_from_time_column(ts: pd.DataFrame, freq: Optional[pd.Timedelta] = None) -> pd.DataFrame:
    datetimes = pd.to_datetime(ts[TIME_COLUMN], utc=True, unit="s")
    if freq is not None:
        ts.index = pd.DatetimeIndex(datetimes, freq=pd.tseries.frequencies.to_offset(freq))
    else:
        ts.index = pd.DatetimeIndex(datetimes)
    ts.index = ts.index.tz_convert("Europe/Berlin")
    return ts


# def prepare_data(ts: pd.DataFrame, detailed_seasonality: bool = True) -> Tuple[str, List[str], pd.DataFrame]:
#     ts.drop_duplicates(subset=TIME_COLUMN, inplace=True)
#     ts = index_from_time_column(ts)
#     ts["hour_of_day"] = ts.index.hour.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
#     ts["day_of_week"] = ts.index.dayofweek.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
#     ts["month_of_year"] = ts.index.month.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
#     extra_columns = ["hour_of_day", "day_of_week", "month_of_year"]
#     if detailed_seasonality:
#         ts["minute_of_hour"] = ts.index.minute.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
#         ts["minute_of_day"] = ts["minute_of_hour"] + ts["hour_of_day"] * 60
#         extra_columns.extend(["minute_of_hour", "minute_of_day"])
#     return UNIVARIATE_DATA_COLUMN, extra_columns, ts


def prepare_data(ts: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    ts.drop_duplicates(subset=TIME_COLUMN, inplace=True)
    return UNIVARIATE_DATA_COLUMN, ts


def extract_features(
    ts: pd.DataFrame,
    y_column: str,
    seasonality_features: bool = True,
    detailed_seasonality: bool = True,
    extra_features: bool = True,
    lag_order: int = DEFAULT_LAG_ORDER,
) -> Tuple[List[str], pd.DataFrame, int]:
    """Extracts features from the univariate time series data.

    Args:
      ts: A pandas data frame containing 'TIME_COLUMN' and 'y_column' as the column names.
      y_column: Name of the univariate output column.
      seasonality_features: Flag for extracting seasonality features such as "hour_of_day".
      detailed_seasonality: Flag for extracting detailed seasonality features such as "minute_of_hour".
      extra_features: Flag for extracting lag features, dT and derivative columns.
      lag_order: Order of lag variables to be extracted into a new column.

    Returns:
      (lc, ts, ur) tuple where 'lc' is a list of added column names, 'ts' is the modified
      time series pd.DataFrame and 'ur' is the number of useless rows due to NaN values.
    """
    assert ts.shape[0] > lag_order, "there is not enough data for lag order {}".format(lag_order)
    cols_added: List[str] = []
    useless_rows = 0
    if seasonality_features:
        ts = index_from_time_column(ts)
        ts["hour_of_day"] = ts.index.hour.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
        ts["day_of_week"] = ts.index.dayofweek.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
        ts["month_of_year"] = ts.index.month.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
        cols_added.extend(["hour_of_day", "day_of_week", "month_of_year"])
        if detailed_seasonality:
            ts["minute_of_hour"] = ts.index.minute.to_series(index=ts.index).astype(DEFAULT_FLOAT_TYPE)
            ts["minute_of_day"] = ts["minute_of_hour"] + ts["hour_of_day"] * 60
            cols_added.extend(["minute_of_hour", "minute_of_day"])
    if extra_features:
        # Extract lag-k outputs
        for lag in range(1, lag_order + 1):
            lag_col = LAG_FEATURE_TEMPLATE.format(lag)
            ts[lag_col] = ts[y_column].shift(lag)
            cols_added.append(lag_col)
        # Extract dT and d(count)/dT.
        DT_COLUMN = "dT"
        ts[DT_COLUMN] = (ts[TIME_COLUMN] - ts[TIME_COLUMN].shift(1)).astype(DEFAULT_FLOAT_TYPE)
        cols_added.append(DT_COLUMN)

        ts[DERIVATIVE_COLUMN] = (ts[y_column].shift(1) - ts[y_column].shift(2)) / (
            ts[TIME_COLUMN].shift(1) - ts[TIME_COLUMN].shift(2)
        ).astype(DEFAULT_FLOAT_TYPE)
        cols_added.append(DERIVATIVE_COLUMN)
        useless_rows = max(useless_rows, max(lag_order, 2))

    return cols_added, ts, useless_rows


def prepare_data_with_features(
    data: pd.DataFrame,
    seasonality_features: bool = True,
    detailed_seasonality: bool = True,
    extra_features: bool = True,
    lag_order: int = DEFAULT_LAG_ORDER,
) -> Tuple[str, List[str], pd.DataFrame, int]:
    y_column, ts = prepare_data(data)
    ts[y_column] = ts[y_column].astype(DEFAULT_FLOAT_TYPE)
    ts[TIME_COLUMN] = ts[TIME_COLUMN].astype(np.int64)
    x_columns, ts, useless_rows = extract_features(
        ts, y_column, seasonality_features, detailed_seasonality, extra_features, lag_order=lag_order
    )

    return y_column, x_columns, ts, useless_rows


def time_series_interpolate(t: np.ndarray, y: np.ndarray, t2: np.ndarray) -> np.ndarray:
    # TODO: hotfix: add extrapolate flag
    # add also concatenation of time sequences so that every value will be in range [0,1]
    t_concat = np.concatenate((t, t2))
    np.sort(t_concat)
    time_scaler = MinMaxScaler().fit(t_concat.reshape(-1, 1))
    scaled_t: np.ndarray = time_scaler.transform(t.reshape(-1, 1))
    univariate_f = interp1d(scaled_t.flatten(), y, fill_value='extrapolate')
    scaled_t2: np.ndarray = time_scaler.transform(t2.reshape(-1, 1))
    return univariate_f(scaled_t2.flatten()).astype(DEFAULT_FLOAT_TYPE)


def regularize_data(ts: pd.DataFrame, y_column: str) -> Tuple[pd.offsets.DateOffset, pd.DataFrame]:
    if ts.shape[0] <= 1:
        return ts
    avg_dt = (
        pd.Timestamp(ts.loc[ts.index[-1], TIME_COLUMN], unit="s") - pd.Timestamp(ts.loc[ts.index[0], TIME_COLUMN], unit="s")
    ) / (ts.shape[0] - 1)
    time_column = ts[TIME_COLUMN].to_numpy().reshape(-1, 1)
    time_scaler = MinMaxScaler().fit(time_column)
    regular_time_col = "regular_{}".format(TIME_COLUMN)
    ts[regular_time_col] = time_scaler.transform(time_column)
    univariate_f = interp1d(ts[regular_time_col].to_numpy(), ts[y_column].to_numpy())
    ts[regular_time_col] = np.linspace(
        ts.loc[ts.index[0], regular_time_col], ts.loc[ts.index[-1], regular_time_col], num=ts.shape[0]
    )

    ts[y_column] = univariate_f(ts[regular_time_col].to_numpy()).astype(DEFAULT_FLOAT_TYPE)
    ts[TIME_COLUMN] = np.round(time_scaler.inverse_transform(ts[regular_time_col].to_numpy().reshape(-1, 1))).astype(np.int64)
    ts.drop(columns=[regular_time_col], inplace=True)
    ts = index_from_time_column(ts)
    return pd.tseries.frequencies.to_offset(avg_dt), ts

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from common.iotlab_utils.iotlab_utils.data_manager import TIME_COLUMN, UNIVARIATE_DATA_COLUMN
from iotlab_utils.data_manager import prepare_data_with_features
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


class TimeseriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = th.tensor(X).float()
        self.y = th.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - (self.seq_len - 1)

    def __getitem__(self, index: int) -> Tuple[th.Tensor, th.Tensor]:
        return (self.X[index : index + self.seq_len], self.y[index + self.seq_len - 1])


class StudentCountPredictor(nn.Module):

    x_columns = ["lag1_count", "dT", "minute_of_day", "day_of_week", "month_of_year"]
    y_column = UNIVARIATE_DATA_COLUMN
    useless_rows = 1
    prediction_count_lower_bound = 0
    prediction_count_higher_bound = 45

    def __init__(self, config: Dict[str, Any], look_back_buffer: Optional[pd.DataFrame] = None):
        super().__init__()

        self.lr: float = config["lr"]

        self.n_features: int = config["n_features"]
        self.n_hidden: int = config["n_hidden"]
        self.n_layers: int = config["n_layers"]
        self.n_outputs: int = config["n_outputs"]

        self.batch_size: int = config["batch_size"]
        self.seq_len: int = config["seq_len"]

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.n_hidden,
            batch_first=True,
            num_layers=self.n_layers,
        )
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.n_hidden, self.n_hidden)
        self.regressor = nn.Linear(self.n_hidden, self.n_outputs)
        self.criterion = nn.MSELoss()
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.lr)

        self.look_back_buffer: Optional[pd.DataFrame] = None
        if look_back_buffer is not None:
            self.update_look_back_buffer(look_back_buffer)

    @property
    def look_back_length(self) -> int:
        return self.seq_len + self.useless_rows

    def update_look_back_buffer(self, look_back_buffer: pd.DataFrame):
        assert look_back_buffer.shape[0] == self.look_back_length, "LSTM lookback buffer must be of length {}".format(
            self.look_back_length
        )
        self.look_back_buffer = look_back_buffer

    def forward(self, x: th.Tensor, labels: Optional[th.Tensor] = None) -> th.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = lstm_out[:, -1]
        out = self.linear(out)
        out = self.relu(out)
        return self.regressor(out)

    def general_step(
        self, batch: Tuple[th.Tensor, th.Tensor], batch_idx: Optional[int] = None, mode: Optional[str] = None
    ) -> th.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def optimize(self, loss: th.Tensor):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def fit_scaler_to_data(self, ts: pd.DataFrame):
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features.fit(ts[self.x_columns])
        self.scaler_count = MinMaxScaler(feature_range=(0, 1))
        self.scaler_count.fit(ts[[self.y_column]])

    def scale_data(self, ts: pd.DataFrame):
        ts.loc[:, self.x_columns] = self.scaler_features.transform(ts.loc[:, self.x_columns])
        ts.loc[:, [self.y_column]] = self.scaler_count.transform(ts.loc[:, [self.y_column]])

    def inverse_data(self, ts: pd.DataFrame):
        ts.loc[:, self.x_columns] = self.scaler_features.inverse_transform(ts.loc[:, self.x_columns])
        ts.loc[:, [self.y_column]] = self.scaler_count.inverse_transform(ts.loc[:, [self.y_column]])

    def prepare_data(self, ts: pd.DataFrame) -> TimeseriesDataset:
        _, _, ts, _ = prepare_data_with_features(ts)
        ts = ts.iloc[self.useless_rows :]
        self.fit_scaler_to_data(ts)
        self.scale_data(ts)
        X_train = ts[self.x_columns].to_numpy()
        y_train = ts[[self.y_column]].to_numpy()
        y_train = y_train.reshape((-1, 1))
        training_dataset = TimeseriesDataset(X_train, y_train, seq_len=self.seq_len)
        return training_dataset

    def predict_single(self, sequence: th.Tensor) -> float:
        prediction = self(sequence).item()
        if prediction < self.prediction_count_lower_bound:
            prediction = self.prediction_count_lower_bound
        if self.prediction_count_higher_bound < prediction:
            prediction = self.prediction_count_higher_bound
        return prediction

    def forecast(self, ts: pd.DataFrame, update_lag1_count: bool = True) -> pd.DataFrame:
        ts = pd.concat([self.look_back_buffer, ts], axis=0)
        ts.reset_index(drop=True, inplace=True)
        _, _, ts, _ = prepare_data_with_features(ts)
        self.scale_data(ts)

        # look_back_length counts are still known
        for i in range(self.useless_rows, self.look_back_length):
            if update_lag1_count:
                ts.loc[ts.index[i], "lag1_count"] = ts.loc[ts.index[i - 1], self.y_column]

        # window starts at second count because first always has NaN data
        for i in range(self.useless_rows, len(ts.index) - self.seq_len):
            # update last count
            if update_lag1_count:
                last_count = ts.loc[ts.index[i + self.seq_len - 1], self.y_column]
                ts.loc[ts.index[i + self.seq_len], "lag1_count"] = last_count

            # get sequence and convert to tensor
            X_sequence = ts.loc[ts.index[i] : ts.index[i + self.seq_len], self.x_columns].to_numpy()
            X_sequence = th.tensor(X_sequence).float()
            X_sequence = th.unsqueeze(X_sequence, 0)

            # predict
            value = self.predict_single(X_sequence)
            ts.loc[ts.index[i + self.seq_len], self.y_column] = value

        self.inverse_data(ts)
        ts[self.y_column] = ts[self.y_column].round()
        return ts.loc[ts.index[self.look_back_length :], [TIME_COLUMN, self.y_column]]

    def plot_after_train(self, ts: pd.DataFrame):
        all_target = ts[[self.y_column]]
        self.forecast(ts, update_lag1_count=True)
        all_pred = pd.Series(ts[[self.y_column]].values.reshape(-1), index=all_target.index, name="predicted_count")
        all_target.index.freq = None
        all_pred.index.freq = None
        all_target.plot(legend=True)
        all_pred.plot(legend=True, linestyle="dotted")
        plt.show()


def train(ts: pd.DataFrame, config: Dict[str, Any]):
    model = StudentCountPredictor(config)
    model.update_look_back_buffer(ts.loc[ts.index[-model.look_back_length :], [TIME_COLUMN, model.y_column]])
    dataset = model.prepare_data(ts)
    dataloader = DataLoader(
        dataset, batch_size=8, persistent_workers=config["persistent_workers"], num_workers=config["n_workers"]
    )
    for epoch in range(config["n_epochs"]):
        for train_batch in dataloader:
            loss = model.general_step(train_batch)
            model.optimize(loss)
    if config["plot_after_train"]:
        model.plot_after_train(ts)
    return model

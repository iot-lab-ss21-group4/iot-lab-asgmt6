import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from iotlab_utils.data_manager import prepare_data_with_features
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

pd.options.mode.chained_assignment = None


class TimeseriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return (self.X[index : index + self.seq_len], self.y[index + self.seq_len - 1])


class StudentCountPredictor(nn.Module):

    x_columns = ["lag1_count", "dT", "minute_of_day", "day_of_week", "month_of_year"]
    y_column = ["count"]
    useless_rows = 1

    def __init__(self, config):
        super().__init__()

        self.lr = config["lr"]

        self.n_features = config["n_features"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.n_outputs = config["n_outputs"]

        self.batch_size = config["batch_size"]
        self.seq_len = config["seq_len"]

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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, labels=None):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = lstm_out[:, -1]
        out = self.linear(out)
        out = self.relu(out)
        return self.regressor(out)

    def general_step(self, batch, batch_idx=None, mode=None):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def optimize(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def fit_scaler_to_data(self, ts: pd.DataFrame):
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features.fit(ts[self.x_columns])
        self.scaler_count = MinMaxScaler(feature_range=(0, 1))
        self.scaler_count.fit(ts[self.y_column])

    def scale_data(self, ts: pd.DataFrame):
        ts.loc[:, self.x_columns] = self.scaler_features.transform(ts.loc[:, self.x_columns])
        ts.loc[:, self.y_column] = self.scaler_count.transform(ts.loc[:, self.y_column])

    def inverse_data(self, ts: pd.DataFrame):
        ts.loc[:, self.x_columns] = self.scaler_features.inverse_transform(ts.loc[:, self.x_columns])
        ts.loc[:, self.y_column] = self.scaler_count.inverse_transform(ts.loc[:, self.y_column])

    def prepare_data(self, ts: pd.DataFrame) -> TimeseriesDataset:
        _, _, ts, _ = prepare_data_with_features(ts)
        ts = ts.iloc[self.useless_rows :]
        self.fit_scaler_to_data(ts)
        self.scale_data(ts)
        X_train = ts[self.x_columns].to_numpy()
        y_train = ts[self.y_column].to_numpy()
        y_train = y_train.reshape((-1, 1))
        training_dataset = TimeseriesDataset(X_train, y_train, seq_len=self.seq_len)
        return training_dataset

    def predict_single(self, sequence: torch.Tensor) -> float:
        prediction = self(sequence).item()
        if prediction < 0:
            prediction = 0
        if 1 < prediction:
            prediction = 1
        return prediction

    def predict(self, ts: pd.DataFrame, update_lag1_count=True):

        self.scale_data(ts)

        # first seq_len + 1 counts are still known
        for i in range(self.useless_rows, self.seq_len + 1):
            if update_lag1_count:
                ts.loc[ts.index[i], "lag1_count"] = ts.loc[ts.index[i - 1], "count"]

        # window starts at second count because first always has NaN data
        for i in range(self.useless_rows, len(ts.index) - self.seq_len):

            # update last count
            if update_lag1_count:
                last_count = ts.loc[ts.index[i + self.seq_len - 1], "count"]
                ts.loc[ts.index[i + self.seq_len], "lag1_count"] = last_count

            # get sequence and convert to tensor
            X_sequence = ts.loc[ts.index[i] : ts.index[i + self.seq_len], self.x_columns].to_numpy()
            X_sequence = torch.tensor(X_sequence).float()
            X_sequence = torch.unsqueeze(X_sequence, 0)

            # predict
            value = self.predict_single(X_sequence)
            ts.loc[ts.index[i + self.seq_len], "count"] = value

        self.inverse_data(ts)
        ts["count"] = ts["count"].round(decimals=0)
        ts["count"] = ts["count"].astype(np.int)

    def plot_after_train(self, data: pd.DataFrame):
        # TODO: will be removed after some testing
        _, _, ts, _ = prepare_data_with_features(data)
        all_target = ts[self.y_column]
        self.predict(ts, update_lag1_count=True)
        all_pred = pd.Series(ts[self.y_column].values.reshape(-1), index=all_target.index, name="predicted_count")
        all_target.index.freq = None
        all_pred.index.freq = None
        all_target.plot(legend=True)
        all_pred.plot(legend=True, linestyle="dotted")
        plt.show()


def train(data: pd.DataFrame, config):
    model = StudentCountPredictor(config)
    dataset = model.prepare_data(data)
    dataloader = DataLoader(dataset, batch_size=8)
    for epoch in range(config["n_epochs"]):
        for train_batch in dataloader:
            loss = model.general_step(train_batch)
            model.optimize(loss)
    model.plot_after_train(data)
    return model
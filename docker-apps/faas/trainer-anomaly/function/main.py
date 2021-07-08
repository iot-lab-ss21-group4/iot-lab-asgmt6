import io
import json
import pickle
import time
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
from iotlab_utils.data_loader import load_data
from iotlab_utils.data_manager import prepare_data_with_features
from minio import Minio
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def main(params: Dict[str, Any]):
    try:
        minio_client = Minio(
            endpoint="{}:{}".format(params["minio_host"], str(params["minio_port"])),
            access_key=params["minio_access_key"],
            secret_key=params["minio_secret_key"],
            secure=False,
        )
        ts = load_data(
            params["iot_platform_consumer_host"],
            params["iot_platform_consumer_id"],
            params["iot_platform_consumer_key"],
            post_processing=params["data_post_processing"],
        )
        y_column, x_columns, ts, useless_rows = prepare_data_with_features(
            ts, detailed_seasonality=False, extra_features=False
        )
        ts: pd.DataFrame = ts.iloc[useless_rows:]

        ts["is_anomaly"] = np.int64(0)
        # Any count bigger than 0 at the weekends is anomaly
        ts.loc[(ts.index.weekday >= 5) & (ts[y_column] > 0), "is_anomaly"] = 1
        # Anything over 45 is also an anomaly
        ts.loc[ts[y_column] > 45, "is_anomaly"] = 1
        outliers_fraction = (ts["is_anomaly"] == 1).sum().item() / ts.shape[0] + 1e-7

        data = ts.loc[:, x_columns + [y_column]]
        scaler = StandardScaler()
        scaler.fit(data)
        scaled_data = pd.DataFrame(scaler.transform(data))

        def objective_isolation_forest(trial: optuna.Trial) -> float:
            kwargs = {}
            kwargs["n_estimators"] = trial.suggest_int("n_estimators", 50, 500, 25)
            kwargs["max_samples"] = trial.suggest_uniform("max_samples", 0.0, 1.0)
            kwargs["max_features"] = trial.suggest_uniform("max_features", 0.0, 1.0)
            kwargs["bootstrap"] = trial.suggest_categorical("bootstrap", [False, True])
            kwargs["warm_start"] = trial.suggest_categorical("warm_start", [False, True])
            # train isolation forest
            model = IsolationForest(contamination=outliers_fraction, **kwargs)
            model.fit(scaled_data)
            anomaly_pred = pd.Series(model.predict(scaled_data), index=ts.index, dtype=np.int64)
            # BEFORE: -1 means anomaly, 1 means normal. AFTER: 1 means anomaly, 0 means normal.
            anomaly_pred.replace({-1: 1, 1: 0}, inplace=True)
            return f1_score(ts["is_anomaly"].to_numpy(), anomaly_pred.to_numpy())

        number_of_training_points = scaled_data.shape[0]
        start = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_isolation_forest, n_trials=100)
        model = IsolationForest(contamination=outliers_fraction, **study.best_params)
        model.fit(scaled_data)
        latency = time.time() - start
        # Pick the latest available pickle protocol version for ibmfunctions/action-python-v3.7:master
        bytes_file = pickle.dumps((model, scaler), protocol=4)
        minio_client.put_object(
            bucket_name=params["model_bucket"],
            object_name=params["model_blob_name"],
            data=io.BytesIO(bytes_file),
            length=len(bytes_file),
        )
        return {"latency": latency, "datapoints": number_of_training_points}
    except Exception as e:
        return {"Error": "Training failed. Reason: {}".format(e)}


if __name__ == "__main__":
    with open("template/anomaly_train.json", "rt") as f:
        params = json.load(f)
    result = main(params)
    print(result)

import io
import json
import pickle
from time import time
from typing import Any, Dict

from iotlab_lstm.lstm_model import train
from iotlab_utils.data_loader import load_data
from minio import Minio


def main(params: Dict[str, Any]):
    try:
        minio_client = Minio(
            endpoint="{}:{}".format(params["minio_host"], str(params["minio_port"])),
            access_key=params["minio_access_key"],
            secret_key=params["minio_secret_key"],
            secure=False,
        )
        data = load_data(
            params["iot_platform_consumer_host"], params["iot_platform_consumer_id"], params["iot_platform_consumer_key"]
        )
        number_of_data_points = data.shape[0]
        start = time()
        model = train(data, params["model_config"])
        latency = time() - start
        # Pick the latest available pickle protocol version for ibmfunctions/action-python-v3.7:latest
        bytes_file = pickle.dumps(model, protocol=4)
        minio_client.put_object(
            bucket_name=params["model_bucket"],
            object_name=params["model_blob_name"],
            data=io.BytesIO(bytes_file),
            length=len(bytes_file),
        )
        return {"latency": latency, "data_points": number_of_data_points}
    except Exception as e:
        return {"Error": "Training failed. Reason: " + str(e)}


# copy until here

if __name__ == "__main__":
    with open("template/input_lstm_train.json", "rt") as f:
        params = json.load(f)
    result = main(params)
    print(result)

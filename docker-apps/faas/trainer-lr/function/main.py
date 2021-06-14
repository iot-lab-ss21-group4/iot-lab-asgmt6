import io
import json
import pickle
from time import time

from minio import Minio
from iotlab_lr.lr_model import train
from iotlab_utils.data_loader import load_data


def main(params):
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
        number_of_training_points = data.shape[0]
        start = time()
        model = train(data)
        latency = time() - start
        bytes_file = pickle.dumps(model)
        minio_client.put_object(
            bucket_name=params["model_bucket"],
            object_name=params["model_blob_name"],
            data=io.BytesIO(bytes_file),
            length=len(bytes_file),
        )
        return {"latency": latency, "datapoints": number_of_training_points}
    except Exception as e:
        return {"Error": "Training failed. Reason: " + str(e)}


# copy until here

if __name__ == "__main__":
    with open("template/input_lr_train.json", "rt") as f:
        params = json.load(f)
    result = main(params)
    print(result)

import io
import json
import pickle
from time import time

from minio import Minio
from iotlab_lr.lr_model import train
from iotlab_utils.data_loader import load_data

# TODO
minioClient = Minio(
    "x.x.x.x:9000", access_key="AKIAIOSFODNN7EXAMPLE", secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", secure=False
)


def main(params):
    try:
        data = load_data(
            params["iot_platform_consumer_host"], params["iot_platform_consumer_id"], params["iot_platform_consumer_key"]
        )
        start = time()
        model = train(data)
        latency = time() - start
        bytes_file = pickle.dumps(model)
        minioClient.put_object(
            bucket_name=params["model_bucket"],
            object_name=params["model_blob_name"],
            data=io.BytesIO(bytes_file),
            length=len(bytes_file),
        )
        return {"latency": latency}
    except Exception as e:
        return {"Error": "Training failed. " + str(e)}


# copy until here

if __name__ == "__main__":
    with open("template/input_lr_train.json", "rt") as f:
        params = json.load(f)
    result = main(params)
    print(result)

from typing import Any, Dict

from minio import Minio


def setup_minio_client(json_data: Dict[str, Any]) -> Minio:
    minio_client = Minio(
        endpoint="{}:{}".format(json_data["minio_host"], str(json_data["minio_port"])),
        access_key=json_data["minio_access_key"],
        secret_key=json_data["minio_secret_key"],
        secure=False,
    )
    return minio_client

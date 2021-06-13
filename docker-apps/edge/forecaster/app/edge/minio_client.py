from typing import Any, Dict

from minio import Minio


def setup_minio_client(config: Dict[str, Any]) -> Minio:
    minio_client = Minio(
        endpoint="{}:{}".format(config["minio_host"], str(config["minio_port"])),
        access_key=config["minio_access_key"],
        secret_key=config["minio_secret_key"],
        secure=False,
    )
    return minio_client

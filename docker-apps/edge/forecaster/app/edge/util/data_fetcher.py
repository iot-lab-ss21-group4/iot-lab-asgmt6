from typing import Any, Dict, Optional, Tuple

import pandas as pd
from iotlab_utils.data_loader import load_data_time_window, load_latest_data
from iotlab_utils.data_manager import TIME_COLUMN, UNIVARIATE_DATA_COLUMN


class DataFetcher:
    def __init__(self, config: Dict[str, Any]):
        self.iot_platform_consumer_host = config["iot_platform_consumer_host"]
        self.iot_platform_consumer_id = config["iot_platform_consumer_id"]
        self.iot_platform_consumer_key = config["iot_platform_consumer_key"]

    def fetch_latest(self, look_back_length: int) -> Tuple[pd.DataFrame, str]:
        if look_back_length <= 0:
            return pd.DataFrame(columns=[TIME_COLUMN, UNIVARIATE_DATA_COLUMN])

        return (
            load_latest_data(
                self.iot_platform_consumer_host,
                self.iot_platform_consumer_id,
                self.iot_platform_consumer_key,
                look_back_length,
            ),
            UNIVARIATE_DATA_COLUMN,
        )

    def fetch_time_window(
        self, lower_bound: Optional[int] = None, upper_bound: Optional[int] = None
    ) -> Tuple[pd.DataFrame, str]:
        return (
            load_data_time_window(
                self.iot_platform_consumer_host,
                self.iot_platform_consumer_id,
                self.iot_platform_consumer_key,
                lower_bound,
                upper_bound,
            ),
            UNIVARIATE_DATA_COLUMN,
        )

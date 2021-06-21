from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from iotlab_utils.data_loader import load_data, load_latest_data
from iotlab_utils.data_manager import DEFAULT_FLOAT_TYPE, TIME_COLUMN, UNIVARIATE_DATA_COLUMN


class DataFetcher:
    def __init__(self, config: Dict[str, Any]):
        self.iot_platform_consumer_host = config["iot_platform_consumer_host"]
        self.iot_platform_consumer_id = config["iot_platform_consumer_id"]
        self.iot_platform_consumer_key = config["iot_platform_consumer_key"]

    def fetch_latest(self, look_back_length: int) -> Tuple[pd.DataFrame, str]:
        if look_back_length <= 0:
            return (
                pd.DataFrame(
                    {TIME_COLUMN: pd.Series(dtype=np.int64), UNIVARIATE_DATA_COLUMN: pd.Series(dtype=DEFAULT_FLOAT_TYPE)},
                    columns=[TIME_COLUMN, UNIVARIATE_DATA_COLUMN],
                ),
                UNIVARIATE_DATA_COLUMN,
            )

        return (
            load_latest_data(
                self.iot_platform_consumer_host,
                self.iot_platform_consumer_id,
                self.iot_platform_consumer_key,
                look_back_length,
            ),
            UNIVARIATE_DATA_COLUMN,
        )

    def fetch(
        self,
        lower_bound: Optional[int] = None,
        upper_bound: Optional[int] = None,
        query_size: Optional[int] = None,
        query_time_order: str = "asc",
    ) -> Tuple[pd.DataFrame, str]:
        return (
            load_data(
                consumer_host=self.iot_platform_consumer_host,
                consumer_id=self.iot_platform_consumer_id,
                consumer_key=self.iot_platform_consumer_key,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                query_size=query_size,
                query_time_order=query_time_order,
            ),
            UNIVARIATE_DATA_COLUMN,
        )

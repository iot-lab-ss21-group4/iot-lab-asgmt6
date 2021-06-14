import pandas as pd
from iotlab_utils.data_loader import load_latest_data
from iotlab_utils.data_manager import TIME_COLUMN, UNIVARIATE_DATA_COLUMN


class DataInitializer():

    def __init__(self, config):
        self.iot_platform_consumer_host=config["iot_platform_consumer_host"]
        self.iot_platform_consumer_id=config["iot_platform_consumer_id"]
        self.iot_platform_consumer_key=config["iot_platform_consumer_key"]

    def initialize_data(self, look_back_length):
        if look_back_length <= 0:
            return pd.DataFrame(columns=[TIME_COLUMN, UNIVARIATE_DATA_COLUMN])

        return load_latest_data(self.iot_platform_consumer_host,
                                self.iot_platform_consumer_id,
                                self.iot_platform_consumer_key,
                                look_back_length)

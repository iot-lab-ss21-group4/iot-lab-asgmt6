import json
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests as requests

from .data_manager import TIME_COLUMN, UNIVARIATE_DATA_COLUMN

scroll_open_timeout = "1m"
consumer_scroll_api_template = "https://{}:443/api/consumers/consume/{}/_search?scroll={}"
entries_per_request = 10000

consumer_search_api_template = "https://{}:443/api/consumers/consume/{}/_search?size={}"
search_api_max_entries_per_request = 10000


def recursive_dict_update(d: Dict, u: Dict) -> Dict:
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def create_data_frame_from_hits(hits_list: List[Dict[str, Any]], data_range: Iterable[int]) -> pd.DataFrame:
    counts_df = pd.DataFrame(index=data_range, columns=[TIME_COLUMN, UNIVARIATE_DATA_COLUMN])
    for i in data_range:
        timestamp_ms, value = hits_list[i]["_source"]["timestamp"], hits_list[i]["_source"]["value"]
        # Filter / Change ts values here
        counts_df.loc[i] = [int(np.round(timestamp_ms / 1000)), value]

    return counts_df


def load_data_time_window(
    consumer_host: str,
    consumer_id: int,
    consumer_key: str,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
) -> pd.DataFrame:
    """
    Uses the scroll API to get all data from an index within the given time window.
    Args:
        consumer_host: host of the consumer
        consumer_id: id of the consumer
        consumer_key: bearer token of the consumer. Must start with 'Bearer ey'
        lower_bound: The lower threshold which our data has to be >= . If not given then there is no lower bound.
        upper_bound: The upper threshold which our data has to be <= . If not given then there is no upper bound.

    Returns: Count and timestamp in a pandas data frame

    """
    consumer_scroll_api_header = {
        "Authorization": consumer_key,
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    consumer_scroll_api = consumer_scroll_api_template.format(consumer_host, consumer_id, scroll_open_timeout)
    query = {}
    if lower_bound is not None:
        lower_bound_query = {"range": {"timestamp": {"gte": int(lower_bound)}}}
        query = recursive_dict_update(query, lower_bound_query)
    if upper_bound is not None:
        upper_bound_query = {"range": {"timestamp": {"lte": int(upper_bound)}}}
        query = recursive_dict_update(query, upper_bound_query)
    if "range" not in query:
        query = {"match_all": {}}
    response = requests.get(
        url=consumer_scroll_api,
        headers=consumer_scroll_api_header,
        verify=False,
        data=json.dumps(
            {
                "size": entries_per_request,
                "query": query,
                "sort": {"timestamp": "asc"},
            }
        ),
    )
    payload = response.json()
    scroll_id = payload["body"]["_scroll_id"]
    hits = payload["body"]["hits"]
    total_hits = hits["total"]
    while True:
        if total_hits <= len(hits["hits"]):
            break
        response = requests.get(
            url=consumer_scroll_api,
            headers=consumer_scroll_api_header,
            verify=False,
            data=json.dumps({"scroll_id": scroll_id, "scroll": scroll_open_timeout}),
        )
        next_hits = response.json()["body"]["hits"]
        hits["hits"].extend(next_hits["hits"])
        hits["total"] = next_hits["total"]
        scroll_id = payload["body"]["_scroll_id"]

    hits_list = hits["hits"]
    data_range = range(total_hits)
    return create_data_frame_from_hits(hits_list, data_range)


def load_data(consumer_host: str, consumer_id: int, consumer_key: str) -> pd.DataFrame:
    """
    Uses the scroll API to get all data from an index
    Args:
        consumer_host: host of the consumer
        consumer_id: id of the consumer
        consumer_key: bearer token of the consumer. Must start with 'Bearer ey'

    Returns: Count and timestamp in a pandas data frame

    """
    return load_data_time_window(consumer_host, consumer_id, consumer_key)


def load_latest_data(consumer_host: str, consumer_id: int, consumer_key: str, latest_entries: int) -> pd.DataFrame:
    """
    Uses search API to get latest data from an index
    Args:
        consumer_host: host of the consumer
        consumer_id: id of the consumer
        consumer_key: bearer token of the consumer. Must start with 'Bearer ey'
        latest_entries: number of latest entries to retrieve
    Returns: Count and timestamp in a pandas data frame

    """
    if search_api_max_entries_per_request < latest_entries:
        raise Exception(
            "Desired number of latest entries with {} exceeds limit of {} allowed entries for this request.".format(
                str(latest_entries), str(search_api_max_entries_per_request)
            )
        )
    consumer_search_api_header = {
        "Authorization": consumer_key,
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    consumer_search_api = consumer_search_api_template.format(consumer_host, consumer_id, latest_entries)
    response = requests.get(
        url=consumer_search_api,
        headers=consumer_search_api_header,
        verify=False,
        data=json.dumps(
            {
                "query": {"match_all": {}},
                "sort": {"timestamp": "desc"},
            }
        ),
    )
    payload = response.json()
    hits = payload["body"]["hits"]
    # we must reverse. Maybe there is a way to do this better in the request
    hits_list = hits["hits"][::-1]
    data_range = range(len(hits_list))
    return create_data_frame_from_hits(hits_list, data_range)

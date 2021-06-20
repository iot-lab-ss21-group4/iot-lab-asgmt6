import json
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests as requests

from .data_manager import TIME_COLUMN, UNIVARIATE_DATA_COLUMN

scroll_open_timeout = "1m"
consumer_scroll_api_template = "https://{}:443/api/consumers/consume/{}/_search?scroll={}"
entries_per_request = 10000


def recursive_dict_update(d: Dict, u: Dict) -> Dict:
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def create_data_frame_from_hits(
    hits_list: List[Dict[str, Any]], data_range: Iterable[int], is_reversed: bool = False
) -> pd.DataFrame:
    if is_reversed:
        hits_list = reversed(hits_list)
    counts_df = pd.DataFrame(index=data_range, columns=[TIME_COLUMN, UNIVARIATE_DATA_COLUMN])
    for i, hit in zip(data_range, hits_list):
        timestamp_ms, value = hit["_source"]["timestamp"], hit["_source"]["value"]
        # Filter / Change ts values here
        counts_df.loc[i] = [int(np.round(timestamp_ms / 1000)), value]

    return counts_df


def load_data(
    consumer_host: str,
    consumer_id: int,
    consumer_key: str,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
    query_size: Optional[int] = None,
    query_time_order: str = "asc",
) -> pd.DataFrame:
    """
    Uses the scroll API to get all data from an index.
    Args:
        consumer_host: host of the consumer
        consumer_id: id of the consumer
        consumer_key: bearer token of the consumer. Must start with 'Bearer ey'
        lower_bound: The lower threshold which our data has to be >= . If not given then there is no lower bound.
        upper_bound: The upper threshold which our data has to be <= . If not given then there is no upper bound.
        query_size: The query size. If not given then there is no limit. If given, it MUST be >= 0.
        query_time_order: Time order of the query. Possible values are {"asc", "desc"}.
            Note that the returned time order is always ascending.

    Returns: Count and timestamp in a pandas data frame

    """
    assert query_time_order in {"asc", "desc"}, 'Query time order can either be "asc" or "desc"!'
    consumer_scroll_api = consumer_scroll_api_template.format(consumer_host, consumer_id, scroll_open_timeout)
    consumer_scroll_api_header = {
        "Authorization": consumer_key,
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }
    query = {}
    if lower_bound is not None:
        lower_bound_query = {"range": {"timestamp": {"gte": int(lower_bound) * 1000}}}
        query = recursive_dict_update(query, lower_bound_query)
    if upper_bound is not None:
        upper_bound_query = {"range": {"timestamp": {"lte": int(upper_bound) * 1000}}}
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
                "sort": {"timestamp": query_time_order},
            }
        ),
    )
    payload_body: Dict[str, Any] = response.json()["body"]
    scroll_id: str = payload_body["_scroll_id"]
    hits: Dict[str, Any] = payload_body["hits"]
    total_hits: int = hits["total"]
    while True:
        if total_hits <= len(hits["hits"]) or (query_size is not None and query_size <= len(hits["hits"])):
            break
        response = requests.get(
            url=consumer_scroll_api,
            headers=consumer_scroll_api_header,
            verify=False,
            data=json.dumps({"scroll_id": scroll_id, "scroll": scroll_open_timeout}),
        )
        next_payload_body: Dict[str, Any] = response.json()["body"]
        next_hits: Dict[str, Any] = next_payload_body["hits"]
        hits["hits"].extend(next_hits["hits"])
        total_hits = hits["total"] = next_hits["total"]
        scroll_id = next_payload_body["_scroll_id"]

    hits_list: List[Dict[str, Any]] = hits["hits"] if query_size is None else hits["hits"][:query_size]
    data_range = range(len(hits_list))
    return create_data_frame_from_hits(hits_list, data_range, query_time_order == "desc")


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
    return load_data(consumer_host, consumer_id, consumer_key, query_size=latest_entries, query_time_order="desc")

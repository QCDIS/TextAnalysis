#!/usr/bin/env python3


"""Description:  """


import time
import requests
import json
import pandas as pd
from pathlib import Path
from xml.etree.ElementTree import fromstring, ElementTree
from utilities import toolbox as tb

__author__ = "U. Odyurt"
__copyright__ = "Copyright 2021"
__credits__ = ["N/A"]
__license__ = "???"
__version__ = "1.0.0"
__maintainer__ = "N/A"
__email__ = "N/A"
__status__ = "Prototype"


# Global constants
EXECUTION_TIMESTAMP = time.strftime("%Y-%m-%d_%H%M%S")
ANCHOR_PATH = Path('/Volumes/Trace/mns_survey')
SEARCH_RESULT_PATH = ANCHOR_PATH / 'search_results'
DEBUG_PATH = ANCHOR_PATH / 'debug'
XPLORE_URL = 'http://ieeexploreapi.ieee.org/api/v1/search/articles'


# =============================================================================
# =============================================================================
def main():
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH


def compose_query(focus_topic_list, context_topic_list):
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH
    pass


def query_xplore(query, output_data_format, output_type):
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    url = compose_url(query)
    response = requests.get(url, allow_redirects=True)
    returned_data = response.content

    if output_data_format == 'raw':
        return returned_data
    elif output_data_format == 'object':
        if output_type == 'xml':
            xml_object = ElementTree(fromstring(returned_data))
            return xml_object
        elif output_type == 'json':
            json_object = json.loads(returned_data)
            return json_object
    elif output_data_format == 'structured':
        return returned_data


def compose_url(query):
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH
    global XPLORE_URL

    generated_url = XPLORE_URL

    return generated_url


if __name__ == "__main__":
    main()

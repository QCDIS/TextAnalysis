#!/usr/bin/env python3


"""Description:  """


import time
import pandas as pd
from pathlib import Path
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


# =============================================================================
# =============================================================================
def main():
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH


def compose_query(glossary_terms):
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH
    pass


def query_scopus(query, ):
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH


if __name__ == "__main__":
    main()

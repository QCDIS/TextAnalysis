"""Description:  """


import itertools
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
DEBUG_PATH = ANCHOR_PATH / 'debug'


# =============================================================================
# =============================================================================
# def main():
#     global EXECUTION_TIMESTAMP
#     global DEBUG_PATH
#
#     pruned_path = SEARCH_RESULT_PATH / 'pruned'
#     papers_path = ANCHOR_PATH / 'papers'
#
#     pruned_df = load_pruned_csv(pruned_path)
#     # Call a worker that goes through the pruned data frame and downloads papers
#     literature_download_worker(pruned_df, papers_path)


def set_paths(anchor_path):
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    paths_dict = {}
    search_result_path = anchor_path / 'search_results'
    if not search_result_path.exists():
        search_result_path.mkdir()
    paths_dict['search_results'] = search_result_path

    analysis_path = anchor_path / 'text_analysis'
    if not analysis_path.exists():
        analysis_path.mkdir()
    paths_dict['text_analysis'] = analysis_path

    for repository in ['ieee', 'scopus', 'acm', 'springer']:
        repository_path = search_result_path / repository
        if not repository_path.exists():
            repository_path.mkdir()
        paths_dict[repository] = repository_path

    papers_path = anchor_path / 'papers'
    if not papers_path.exists():
        papers_path.mkdir()
    paths_dict['papers'] = papers_path

    uber_search_path = search_result_path / 'uber'
    if not uber_search_path.exists():
        uber_search_path.mkdir()
    paths_dict['uber'] = uber_search_path

    pruned_path = search_result_path / 'pruned'
    if not pruned_path.exists():
        pruned_path.mkdir()
    paths_dict['pruned'] = pruned_path

    glossary_path = analysis_path / 'glossary'
    if not glossary_path.exists():
        glossary_path.mkdir()
    paths_dict['glossary'] = glossary_path

    key_term_hits_path = analysis_path / 'key_term_hits'
    if not key_term_hits_path.exists():
        key_term_hits_path.mkdir()
    paths_dict['key_term_hits'] = key_term_hits_path

    related_papers_path = analysis_path / 'high_value_papers'
    if not related_papers_path.exists():
        related_papers_path.mkdir()
    paths_dict['high_value_papers'] = related_papers_path

    visualisations_path = analysis_path / 'visualisations'
    if not visualisations_path.exists():
        visualisations_path.mkdir()
    paths_dict['visualisations'] = visualisations_path

    return paths_dict


def compose_queries(focus_topics, context_topics):
    """
    Description
    :param focus_topics:
    :param context_topics:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    query_list = []
    query_items = list(itertools.product(focus_topics, context_topics))
    for item in query_items:
        query_list.append(" AND ".join(item))

    return query_list


def literature_download(high_value_path):
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH


def load_pruned_csv(load_path):
    """
    Description
    :param load_path:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    pruned_file_path = load_path / 'pruned_search_results.csv'
    pruned_df = pd.read_csv(pruned_file_path, sep=',')

    return pruned_df


def load_pruned_data_frame(load_path):
    """
    Description
    :param load_path:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    pruned_file_path = load_path / 'pruned_search_results.pkl'
    pruned_df = tb.load_object(pruned_file_path)

    return pruned_df


def literature_download_worker(pruned_df, papers_path):
    """
    Description
    :param pruned_df:
    :param papers_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    for index, row in pruned_df.iterrows():
        paper_id = row['Paper ID']
        first_author_name = row['Authors'].split('; ')[0].split(' ')[1]
        publication_year = row['Publication Year']
        paper_title = row['Document Title']
        paper_url = row['PDF Link']
        paper_filename = f'{paper_id}_{first_author_name}_{publication_year}_{paper_title}.pdf'

        tb.file_downloader(paper_url, paper_filename, 'PDF', papers_path)

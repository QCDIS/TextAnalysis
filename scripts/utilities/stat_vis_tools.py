"""Description:  """


import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS
import analyser
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
def stats_analysis(uber_df, glossary_pruned_df):
    """Description
    :param uber_df:
    :param glossary_pruned_df:
    """

    global EXECUTION_TIMESTAMP
    global ANCHOR_PATH
    global DEBUG_PATH

    visualisations_path = ANCHOR_PATH / 'text_analysis' / 'visualisations'

    plot_key_term_yearly_distribution(uber_df, glossary_pruned_df)

    column_name = 'Publication Year'
    column_stats_dict = column_stats_analysis(uber_df, column_name)
    #
    plot_column_distribution(uber_df, column_name, visualisations_path)
    #
    generate_word_cloud(uber_df, visualisations_path)


def column_stats_analysis(uber_df, column):
    """Description
    :param uber_df:
    :param column:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    column_stats_dict = {}
    column_data = uber_df[column]

    return column_stats_dict


def plot_column_distribution(uber_df, column, plot_path):
    """Description
    :param uber_df:
    :param column:
    :param plot_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    queries = uber_df['Query'].unique()
    for query in queries:
        query_df = uber_df[uber_df['Query'] == query]
        sns.displot(query_df, x=column)
        plot_file = plot_path / f'{query}_{column}_distribution.pdf'
        plt.savefig(plot_file, format='pdf', dpi=1200)
    plt.close()


def plot_key_term_yearly_distribution(uber_df, glossary_key_terms):
    """Description
    :param uber_df:
    :param glossary_key_terms:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    queries = glossary_key_terms['query'].unique()
    for query in queries:
        query_df = uber_df[uber_df['Query'] == query]
        query_glossary_df = glossary_key_terms[glossary_key_terms['query'] == query]
        key_terms_by_year_dict = {element: 0 for element in query_glossary_df['keyword']}
        for index, row in query_df.iterrows():
            title = row['Document Title']
            abstract = row['Abstract']
            row_field = row['Author Keywords']
            keywords_author = row_field.split(';') if pd.notnull(row_field) else []
            row_field = row['IEEE Terms']
            keywords_ieee = row_field.split(';') if pd.notnull(row_field) else []
            row_field = row['INSPEC Controlled Terms']
            keywords_inspec_controlled = row_field.split(';') if pd.notnull(row_field) else []
            row_field = row['INSPEC Non-Controlled Terms']
            keywords_inspec_non_controlled = row_field.split(';') if pd.notnull(row_field) else []

            # Concatenation for any type of iterable data structure
            keywords_full_list = [*keywords_author, *keywords_ieee,
                                  *keywords_inspec_controlled, *keywords_inspec_non_controlled]

            key_term_frequencies_in_title = analyser.key_term_frequency_per_glossary(query_glossary_df,
                                                                                     title,
                                                                                     'counter')
            print(key_term_frequencies_in_title)
            key_term_frequencies_in_abstract = analyser.key_term_frequency_per_glossary(query_glossary_df,
                                                                                        abstract,
                                                                                        'counter')
            print(key_term_frequencies_in_abstract)
            key_term_frequencies_in_keywords = analyser.key_term_frequency_per_glossary(query_glossary_df,
                                                                                        keywords_full_list,
                                                                                        'counter')
            print(key_term_frequencies_in_keywords)
            # for key_term in key_terms_by_year_dict


def generate_word_cloud(uber_df, plot_path):
    """Description
    :param uber_df:
    :param plot_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    queries = uber_df['query'].unique()
    for query in queries:
        query_df = uber_df[uber_df['Query'] == query]
        text_titles = ', '.join(str(element) for element in query_df['Document Title'])
        text_abstracts = ', '.join(str(element) for element in query_df['Abstract'])
        # Concatenate all the mini-lists generated from splitting of keywords in every row of the column
        keywords_author = [*(keywords.split(';')
                             for keywords in query_df['Author Keywords'])]
        keywords_ieee = [*(keywords.split(';')
                           for keywords in query_df['IEEE Terms'])]
        keywords_inspec_controlled = [*(keywords.split(';')
                                        for keywords in query_df['INSPEC Controlled Terms'])]
        keywords_inspec_non_controlled = [*(keywords.split(';')
                                            for keywords in query_df['INSPEC Non-Controlled Terms'])]
        # Concatenation all types of keywords in one list
        keywords_full = [*keywords_author, *keywords_ieee,
                         *keywords_inspec_controlled, *keywords_inspec_non_controlled]
        # Make a string out of all keywords in 'keywords_full', separated by a comma
        text_keywords_full = ', '.join(str(element) for element in keywords_full)

        word_cloud_title = WordCloud().generate(text_titles)
        plot_word_cloud(query, 'title', plot_path)
        word_cloud_abstract = WordCloud().generate(text_abstracts)
        plot_word_cloud(query, 'abstract', plot_path)
        word_cloud_keyword = WordCloud().generate(text_keywords_full)
        plot_word_cloud(query, 'keyword', plot_path)


def plot_word_cloud(query, field, plot_path):
    """Description
    :param query:
    :param field:
    :param plot_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    plot_file = plot_path / f'{query}_{field}_word_cloud.pdf'
    plt.savefig(plot_file, format='pdf', dpi=1200)
    plt.close()

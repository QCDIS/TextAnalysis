#!/usr/bin/env python3


"""
Description: What we mean by a 'key term' is a word or a phrase that we are interested in, and we will look for its
presence in the metadata to be analysed. A 'keyword' on the other hand, is the terms included within the metadata,
e.g., Author Keywords, or IEEE Terms.
"""


import time
import collections
import pandas as pd
from pathlib import Path
from textblob import TextBlob
from utilities import toolbox as tb
from utilities import gb_to_us as ref
from utilities import analyser_tools as at
from utilities import stat_vis_tools as svt

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
# Values
SELECTED_REPOSITORY = 'ieee'
HIGH_VALUE_PAPERS_COUNT = 40


# =============================================================================
# =============================================================================
def main():
    global EXECUTION_TIMESTAMP
    global ANCHOR_PATH
    global DEBUG_PATH
    global SELECTED_REPOSITORY
    global HIGH_VALUE_PAPERS_COUNT

    # For manual terms, especially short abbreviations, we have to consider brackets or spaces, to prevent false
    # positives when looking for hits.
    manual_key_terms_dict = {
        'layer_1': ['cognitive', 'cognitive computing', 'cognitive infrastructure'],
        'layer_2': ['networking', 'artificial intelligence', ' ai ', '(ai)', 'cloud', 'classical cloud',
                    'serverless', 'faas'],
        'layer_3': ['software defined networking', 'sdn', 'telecom', 'telecom cloud', 'machine learning',
                    ' ml ', '(ml)', 'ai-enabled', 'ai-based', 'ai-assisted', 'ai-enhanced', 'ai-augmented',
                    'ai-powered', 'ai-driven', 'application', 'function', 'cloud application',
                    'serverless application', 'serverless function', 'implementation', 'infrastructure',
                    'fog', 'edge'],
        'layer_4': ['optimisation', 'scheduling', 'deployment', 'scaling', 'scalability', 'elasticity',
                    'auto-scaling']
    }
    
    subtopics_list = ['artificial intelligence', 'cloud computing']
    queries_list = ['"cloud computing" AND challenge']
    # queries_list = ['"distributed learning"']
    # queries_list = ['"artificial intelligence" AND optimization',
    #                 '"resource allocation" AND cloud',
    #                 '"resource allocation" AND edge',
    #                 '"resource allocation" AND fog',
    #                 'serverless']

    focus_topics = ['"artificial intelligence"', '"cloud computing"']
    context_topics = ['trend', 'challenge']

    # Setting the folder structure
    paths_dict = at.set_paths(ANCHOR_PATH)
    # Compose a query list, based on focus and context topics
    queries_list = at.compose_queries(focus_topics, context_topics)

    # ==================================================
    # ================= Flags section ==================
    # ==================================================
    build_base_data_frame_flag = True
    build_glossary_flag = True
    # Manual key term usage is deprecated
    # manual_key_term_analysis_flag = False
    glossary_key_terms_analysis_flag = True
    stats_analysis_flag = False
    download_publication_flag = False
    # ==================================================
    # ==================================================
    # ==================================================

    if build_base_data_frame_flag:
        # Loading different search results from CSV files into individual and unified data frames
        if SELECTED_REPOSITORY == 'ieee':
            search_dfs, uber_df = build_data_frame_ieee(paths_dict[SELECTED_REPOSITORY])
        elif SELECTED_REPOSITORY == 'scopus':
            search_dfs, uber_df = build_data_frame_scopus(paths_dict[SELECTED_REPOSITORY])
        else:
            uber_df = None
        # Saving the unified data frame
        save_uber_search(uber_df, SELECTED_REPOSITORY, paths_dict['uber'])
        save_uber_search_csv(uber_df, SELECTED_REPOSITORY, paths_dict['uber'])
    else:
        uber_df = load_uber_data_frame(paths_dict['uber'], SELECTED_REPOSITORY)

    if build_glossary_flag:
        glossary_df, query_paper_counts_dict = build_glossary(uber_df)
        save_glossary(glossary_df, 'glossary', paths_dict['glossary'])
        save_glossary_csv(glossary_df, 'glossary_query', paths_dict['glossary'])
        glossary_pruned_df = prune_glossary(glossary_df, query_paper_counts_dict)
        save_glossary(glossary_pruned_df, 'glossary_pruned', paths_dict['glossary'])
        # TODO glossary_pruned_query?
        save_glossary_csv(glossary_pruned_df, 'glossary_pruned_query', paths_dict['glossary'])
    else:
        glossary_df = load_glossary('glossary', paths_dict['glossary'])
        glossary_pruned_df = load_glossary('glossary_pruned', paths_dict['glossary'])

    # ==================================================
    # Manual key term usage is deprecated
    # if manual_key_term_analysis_flag:
    #     key_term_hit_stats, key_term_hit_words = metadata_analysis_manual_key_terms(uber_df, manual_key_terms_dict)
    #     save_analysis_result_csv(key_term_hit_stats, 'manual_key_term_hit_stats', paths_dict['key_term_hits'])
    #     save_analysis_result_csv(key_term_hit_words, 'manual_key_term_hit_words', paths_dict['key_term_hits'])
    #     save_analysis_result_report(key_term_hit_stats, key_term_hits_path)
    # ==================================================

    if glossary_key_terms_analysis_flag:
        key_term_hit_stats, key_term_hit_words = metadata_analysis_glossary_key_terms(uber_df, glossary_pruned_df)
        high_value_papers_df = collect_high_value_papers(uber_df, key_term_hit_stats, HIGH_VALUE_PAPERS_COUNT)
        save_analysis_result_csv(key_term_hit_stats, 'glossary_key_term_hit_stats', paths_dict['key_term_hits'])
        save_analysis_result_csv(key_term_hit_words, 'glossary_key_term_hit_words', paths_dict['key_term_hits'])
        save_high_value_papers_csv(high_value_papers_df, 'high_value_papers', paths_dict['high_value_papers'])
        # save_analysis_result_report(key_term_hit_stats, key_term_hits_path)

    if stats_analysis_flag:
        # Perform different statistical analyses
        svt.stats_analysis(uber_df, glossary_pruned_df)

    if download_publication_flag:
        pass
        # download_publications()

    tb.log_message(EXECUTION_TIMESTAMP, DEBUG_PATH, 4, "Execution of this script has finished.")


def build_data_frame_scopus(trace_path):
    """
    For Scopus search responses, the following columns will be present:
    'Authors', 'Author(s) ID', 'Title', 'Year', 'Source title', 'Volume', 'Issue', 'Art. No.', 'Page start',
    'Page end', 'Page count', 'Cited by', 'DOI', 'Link', 'Affiliations', 'Authors with affiliations', 'Abstract',
    'Author Keywords', 'Index Keywords', 'Correspondence Address', 'Editors', 'Publisher', 'ISSN', 'ISBN', 'CODEN',
    'PubMed ID', 'Language of Original Document', 'Abbreviated Source Title', 'Document Type', 'Publication Stage',
    'Open Access', 'Source', 'EID'

    This function also adds the following columns, derived from the CSV file's name:
    'Paper ID', 'Source ID', 'Query'

    'Paper ID' values are generated at the end, as a unique identifier.
    :param trace_path:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    search_dfs = []
    for entry in trace_path.iterdir():
        if entry.is_file() and entry.suffix == '.csv' and not entry.stem.startswith('.'):
            source_query_split = entry.stem.split('_')
            search_df = pd.read_csv(entry, sep=',')
            # Get rid of 'NaN's
            search_df.fillna('', inplace=True)

            for column in ['Title', 'Abstract', 'Author Keywords', 'Index Keywords']:
                column_to_transform = search_df[column]
                # Converting to lower case and translating any British English entries to American English
                column_to_transform = [element.lower() for element in column_to_transform]
                search_df[column] = convert_to_american_en(column_to_transform)

            # Adding a 'Query' column, as it is useful to have the query within the CSV structure
            insert_column(search_df, 'Query', source_query_split[1])
            # Adding a 'Source ID' column, as it is useful to have the relevant journal/conference within the CSV
            # structure
            insert_column(search_df, 'Source ID', source_query_split[0])
            search_dfs.append(search_df)

    # Concatenating all individual search data frames into one
    uber_search_df = pd.concat(search_dfs, ignore_index=True)
    # Adding a 'Paper ID' column to be able to keep track of the analysis results per paper later on
    insert_column(uber_search_df, 'Paper ID', range(1, len(uber_search_df) + 1))

    return search_dfs, uber_search_df


def build_data_frame_ieee(trace_path):
    """
    For IEEE search responses, the following columns will be present:
    'Document Title', 'Authors', 'Author Affiliations', 'Publication Title', 'Date Added To Xplore',
    'Publication Year', 'Volume', 'Issue', 'Start Page', 'End Page', 'Abstract', 'ISSN', 'ISBNs', 'DOI',
    'Funding Information', 'PDF Link', 'Author Keywords', 'IEEE Terms', 'INSPEC Controlled Terms',
    'INSPEC Non-Controlled Terms', 'Mesh_Terms', 'Article Citation Count', 'Patent Citation Count', 'Reference Count',
    'License', 'Online Date', 'Issue Date', 'Meeting Date', 'Publisher', 'Document Identifier'

    This function also adds the following columns, derived from the CSV file's name:
    'Paper ID', 'Source ID', 'Query'

    'Paper ID' values are generated at the end, as a unique identifier.
    :param trace_path:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    search_dfs = []
    for entry in trace_path.iterdir():
        if entry.is_file() and entry.suffix == '.csv' and not entry.stem.startswith('.'):
            source_query_split = entry.stem.split('_')
            search_df = pd.read_csv(entry, sep=',')
            # Get rid of 'NaN's
            search_df.fillna('', inplace=True)
            # This line removes quotation marks from the query (not needed)
            # source_query_split[1] = source_query_split[1].replace('"', '')

            for column in ['Document Title', 'Abstract', 'Author Keywords', 'IEEE Terms',
                           'INSPEC Controlled Terms', 'INSPEC Non-Controlled Terms']:
                column_to_transform = search_df[column]
                # Converting to lower case and translating any British English entries to American English
                column_to_transform = [element.lower() for element in column_to_transform]
                search_df[column] = convert_to_american_en(column_to_transform)

            # Adding a 'Query' column, as it is useful to have the query within the CSV structure
            insert_column(search_df, 'Query', source_query_split[1])
            # Adding a 'Source ID' column, as it is useful to have the relevant journal/conference within the CSV
            # structure
            insert_column(search_df, 'Source ID', source_query_split[0])
            search_dfs.append(search_df)

    # Concatenating all individual search data frames into one
    uber_search_df = pd.concat(search_dfs, ignore_index=True)
    # Adding a 'Paper ID' column to be able to keep track of the analysis results per paper later on
    insert_column(uber_search_df, 'Paper ID', range(1, len(uber_search_df) + 1))

    return search_dfs, uber_search_df


def save_uber_search(uber_df, repository, save_path):
    """
    Saves the uber dataframe as a byte stream on disk, using pickle module.
    :param uber_df:
    :param repository:
    :param save_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    save_file = save_path / f'uber_search_results_{repository}.pkl'
    tb.save_object(save_file, uber_df)


def save_uber_search_csv(uber_df, repository, save_path):
    """
    Saves the uber dataframe as a CSV to have a human readable alternative.
    :param uber_df:
    :param repository:
    :param save_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    save_file_path = save_path / f'uber_search_results_{repository}.csv'
    uber_df.to_csv(save_file_path, index=None, header=True)


def load_uber_data_frame(load_path, repository):
    """
    Description
    :param load_path:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    uber_file_path = load_path / f'uber_search_results_{repository}.pkl'
    uber_df = tb.load_object(uber_file_path)

    return uber_df


def build_glossary(uber_df):
    """
    Description
    :param uber_df:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    query_paper_counts_dict = {}
    query_dfs_list = []
    queries = uber_df['Query'].unique()
    for query in queries:
        query_glossary_list = []
        query_df = uber_df[uber_df['Query'] == query]
        query_paper_count = query_df.shape[0]
        query_paper_counts_dict[query] = query_paper_count
        for index, row in query_df.iterrows():
            row_field = row['Author Keywords']
            keywords_author = row_field.split(';') if pd.notnull(row_field) else []
            row_field = row['IEEE Terms']
            keywords_ieee = row_field.split(';') if pd.notnull(row_field) else []
            row_field = row['INSPEC Controlled Terms']
            keywords_inspec_controlled = row_field.split(';') if pd.notnull(row_field) else []
            row_field = row['INSPEC Non-Controlled Terms']
            keywords_inspec_non_controlled = row_field.split(';') if pd.notnull(row_field) else []

            # Concatenation for any type of iterable data structure
            # This list is related to one row and could have duplicates
            row_glossary_list = [*keywords_author, *keywords_ieee,
                                 *keywords_inspec_controlled, *keywords_inspec_non_controlled]
            # Add the glossary from this row to the glossary from the rest of the processed rows
            query_glossary_list = [*query_glossary_list, *row_glossary_list]

        # Converting to lower case and translating any British English entries to American English
        query_glossary_lower_list = [element.lower() for element in query_glossary_list]
        query_glossary_lower_list = convert_to_american_en(query_glossary_lower_list)

        keyword_frequency = collections.Counter(query_glossary_lower_list)
        # glossary_dict[query] = dict(keyword_frequency.most_common())
        query_df = pd.DataFrame.from_records(keyword_frequency.most_common(), columns=['Keyword', 'Count'])

        # Get the core terms or phrases from the query, which may include logic operators.
        query_core_list = get_query_core(query)

        # Get rid of empty strings in glossary and
        # Get rid of strings exactly equal to the core terms or phrases from the query itself
        query_df = query_df[(query_df['Keyword'] != '') & (~query_df['Keyword'].isin(query_core_list))]
        insert_column(query_df, 'Query', query)
        query_dfs_list.append(query_df)

    # Concatenating all individual search data frames into one
    glossary_df = pd.concat(query_dfs_list, ignore_index=True)

    return glossary_df, query_paper_counts_dict


def get_query_core(query):
    """
    This function extracts the core terms and/or phrases making up the query. The goal is to exclude these from the
    generated glossary later on, since obviously these terms and phrases will have high frequencies. We do expect a
    certain format for queries containing logic operators as follows:
    (term1 OR term2 OR ...) AND (term3 OR term4 OR ...) AND ...
    The above format considers the most extensive form of a query. Queries do not have to be that complex and single
    term queries will be handled too, resulting in a single element list as the return value. For phrases, i.e.,
    multi-word terms, we expect these to be surrounded with quotation marks. Quotation marks do not affect the
    processing within this function, but are needed to control repository search engine behaviour, resulting in
    accurate responses to queries.
    :param query:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    # This line removes quotation marks from the query
    query_no_quote = query.replace('"', '')

    or_substring_list_total = []
    if 'AND' in query_no_quote:
        and_substring_list = query_no_quote.split(' AND ')
        for and_substring in and_substring_list:
            if 'OR' in and_substring:
                and_substring = and_substring.replace('(', '')
                and_substring = and_substring.replace(')', '')
                or_substring_list = and_substring.split(' OR ')
                or_substring_list_total = [*or_substring_list_total, *or_substring_list]
            else:
                or_substring_list_total.append(and_substring)
        query_core = or_substring_list_total

    elif 'OR' in query_no_quote:
        or_substring_list_total = query_no_quote.split(' OR ')
        query_core = or_substring_list_total

    else:
        query_core = [query_no_quote]

    return query_core


def prune_glossary(glossary_df, query_paper_counts_dict):
    """
    Description
    :param glossary_df:
    :param query_paper_counts_dict:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    paper_total_percentage = 5
    query_pruned_dfs_list = []
    queries = glossary_df['Query'].unique()
    for query in queries:
        query_df = glossary_df[glossary_df['Query'] == query]
        top_portion = int((paper_total_percentage * query_paper_counts_dict[query]) / 100)
        query_pruned_df = query_df[query_df['Count'] >= top_portion]
        query_pruned_dfs_list.append(query_pruned_df)

    # Concatenating all individual pruned query data frames into one
    glossary_pruned_df = pd.concat(query_pruned_dfs_list, ignore_index=True)

    return glossary_pruned_df


def save_glossary(glossary_df, file_name, save_path):
    """
    Saves the glossary dataframe as a byte stream on disk, using pickle module.
    :param glossary_df:
    :param file_name:
    :param save_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    save_file = save_path / f'{file_name}.pkl'
    tb.save_object(save_file, glossary_df)


def save_glossary_csv(glossary_df, file_name, save_path):
    """
    Saves the glossary dataframe as multiple CSVs to have a human readable alternative. Every query within the
    glossary dataframe is saved as a separate CSV.
    :param glossary_df:
    :param file_name:
    :param save_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    queries = glossary_df['Query'].unique()
    for query in queries:
        query_df = glossary_df[glossary_df['Query'] == query]
        save_file_path = save_path / f'{file_name}_{query}.csv'
        query_df.to_csv(save_file_path, index=None, header=True)


def load_glossary(file_name, load_path):
    """
    Loads the glossary dataframe from a previously saved byte stream on disk, using pickle module.
    :param file_name:
    :param load_path:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    glossary_file_path = load_path / f'{file_name}.pkl'
    glossary_df = tb.load_object(glossary_file_path)

    return glossary_df


def insert_column(data_frame, column_name, fill_value, reference=0):
    """
    Given any dataframe, inserts a column to the left hand side of another reference column, populating the newly
    created column with desired data.
    :param data_frame:
    :param column_name:
    :param fill_value:
    :param reference:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    data_frame.insert(loc=reference, column=column_name, value=fill_value)


def metadata_analysis_manual_key_terms(uber_df, reference_key_terms):
    """
    Performs text analysis for designated fields within the collected metadata from previous search queries. Designated
    fields are 'Document Title', 'Abstract' and the collection of all keywords from 'Author Keywords', 'IEEE Terms',
    'INSPEC Controlled Terms' and 'INSPEC Non-Controlled Terms'. The main goal here is to detect the number of cases
    and the cases where our key terms are present.
    This function specifically considers our manually composed key terms.
    :param uber_df:
    :param reference_key_terms:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    analysis_columns_stats = ['Paper ID', 'Source ID', 'Query',
                              'Title L1 Hits', 'Title L2 Hits', 'Title L3 Hits', 'Title L4 Hits',
                              'Abstract L1 Hits', 'Abstract L2 Hits', 'Abstract L3 Hits', 'Abstract L4 Hits',
                              'Keywords L1 Hits', 'Keywords L2 Hits', 'Keywords L3 Hits', 'Keywords L4 Hits',
                              'Title Total Hits', 'Abstract Total Hits', 'Keywords Total Hits', 'Total Hits']
    analysis_columns_words = ['Paper ID', 'Source ID', 'Query',
                              'Title L1 Hits', 'Title L2 Hits', 'Title L3 Hits', 'Title L4 Hits',
                              'Abstract L1 Hits', 'Abstract L2 Hits', 'Abstract L3 Hits', 'Abstract L4 Hits',
                              'Keywords L1 Hits', 'Keywords L2 Hits', 'Keywords L3 Hits', 'Keywords L4 Hits']
    # ==================================================
    # To test if the NLP library gives similar results to string-wise comparison
    # Options: 'nlp' and 'counter'
    # ==================================================
    analysis_method = 'counter'
    # analysis_method = 'nlp'
    # ==================================================
    # ==================================================

    stats_rows_list = []
    words_rows_list = []
    for index, row in uber_df.iterrows():
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
        keywords_full = [*keywords_author, *keywords_ieee,
                         *keywords_inspec_controlled, *keywords_inspec_non_controlled]
        # Make a string out of all keywords in 'keywords_full', separated by a comma
        keywords_full_text = ', '.join(str(element) for element in keywords_full)

        key_term_frequencies_in_title = key_term_frequency_per_taxonomy_layers(reference_key_terms,
                                                                               title,
                                                                               analysis_method)
        key_term_frequencies_in_abstract = key_term_frequency_per_taxonomy_layers(reference_key_terms,
                                                                                  abstract,
                                                                                  analysis_method)
        key_term_frequencies_in_keywords = key_term_frequency_per_taxonomy_layers(reference_key_terms,
                                                                                  keywords_full_text,
                                                                                  analysis_method)

        title_hits_stats = {}
        title_hits_words = {}
        abstract_hits_stats = {}
        abstract_hits_words = {}
        keyword_hits_stats = {}
        keyword_hits_words = {}
        # Layers are the same for all of the following dictionaries. We iterate over layers separately for each, in
        # case in the future the layers applicable to every dictionary differs.
        for layer in key_term_frequencies_in_title:
            title_freq_dict = key_term_frequencies_in_title[layer]
            title_hits_stats[layer] = sum(title_freq_dict.values())
            tmp_string = ""
            for key in title_freq_dict.keys():
                tmp_string += f"{key}: {title_freq_dict[key]};" if title_freq_dict[key] > 0 else ""
            title_hits_words[layer] = tmp_string

        for layer in key_term_frequencies_in_abstract:
            abstract_freq_dict = key_term_frequencies_in_abstract[layer]
            abstract_hits_stats[layer] = sum(abstract_freq_dict.values())
            tmp_string = ""
            for key in abstract_freq_dict.keys():
                tmp_string += f"{key}: {abstract_freq_dict[key]};" if abstract_freq_dict[key] > 0 else ""
            abstract_hits_words[layer] = tmp_string

        for layer in key_term_frequencies_in_keywords:
            keyword_freq_dict = key_term_frequencies_in_keywords[layer]
            keyword_hits_stats[layer] = sum(keyword_freq_dict.values())
            tmp_string = ""
            for key in keyword_freq_dict.keys():
                tmp_string += f"{key}: {keyword_freq_dict[key]};" if keyword_freq_dict[key] > 0 else ""
            keyword_hits_words[layer] = tmp_string

        title_hits_sum = sum(title_hits_stats.values())
        abstract_hits_sum = sum(abstract_hits_stats.values())
        keyword_hits_sum = sum(keyword_hits_stats.values())
        total_hits = title_hits_sum + abstract_hits_sum + keyword_hits_sum

        stats_row_to_add = [row['Paper ID'], row['Source ID'], row['Query'],
                            title_hits_stats['layer_1'], title_hits_stats['layer_2'],
                            title_hits_stats['layer_3'], title_hits_stats['layer_4'],
                            abstract_hits_stats['layer_1'], abstract_hits_stats['layer_2'],
                            abstract_hits_stats['layer_3'], abstract_hits_stats['layer_4'],
                            keyword_hits_stats['layer_1'], keyword_hits_stats['layer_2'],
                            keyword_hits_stats['layer_3'], keyword_hits_stats['layer_4'],
                            title_hits_sum, abstract_hits_sum, keyword_hits_sum, total_hits]
        stats_rows_list.append(stats_row_to_add)

        words_row_to_add = [row['Paper ID'], row['Source ID'], row['Query'],
                            title_hits_words['layer_1'], title_hits_words['layer_2'],
                            title_hits_words['layer_3'], title_hits_words['layer_4'],
                            abstract_hits_words['layer_1'], abstract_hits_words['layer_2'],
                            abstract_hits_words['layer_3'], abstract_hits_words['layer_4'],
                            keyword_hits_words['layer_1'], keyword_hits_words['layer_2'],
                            keyword_hits_words['layer_3'], keyword_hits_words['layer_4']]
        words_rows_list.append(words_row_to_add)

    analysis_result_stats_df = pd.DataFrame(stats_rows_list, columns=analysis_columns_stats)
    analysis_result_words_df = pd.DataFrame(words_rows_list, columns=analysis_columns_words)

    return analysis_result_stats_df, analysis_result_words_df


def metadata_analysis_glossary_key_terms(uber_df, glossary_key_terms):
    """
    This function specifically considers our automatically composed key terms from the glossary.
    :param uber_df:
    :param glossary_key_terms:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    analysis_columns_stats = ['Paper ID', 'Source ID', 'Query',
                              'Title Hits', 'Abstract Hits', 'Keywords Hits', 'Total Hits', 'Rank Score']
    analysis_columns_words = ['Paper ID', 'Source ID', 'Query',
                              'Title Hits', 'Abstract Hits', 'Keywords Hits', 'Rank Score']
    # ==================================================
    # To test if the NLP library gives similar results to string-wise comparison
    # Options: 'nlp' and 'counter'
    # ==================================================
    analysis_method = 'counter'
    # analysis_method = 'nlp'
    # ==================================================
    # ==================================================

    analysis_result_stats_per_query_dfs = []
    analysis_result_words_per_query_dfs = []
    queries = glossary_key_terms['Query'].unique()
    for query in queries:
        stats_rows_list = []
        words_rows_list = []
        tmp_uber_df = uber_df[uber_df['Query'] == query]
        tmp_glossary_df = glossary_key_terms[glossary_key_terms['Query'] == query]
        for index, row in tmp_uber_df.iterrows():
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
            keywords_full = [*keywords_author, *keywords_ieee,
                             *keywords_inspec_controlled, *keywords_inspec_non_controlled]
            # Make a string out of all keywords in 'keywords_full', separated by a comma
            keywords_full_text = ', '.join(str(element) for element in keywords_full)

            key_term_frequencies_in_title = key_term_frequency_per_glossary(tmp_glossary_df,
                                                                            title,
                                                                            analysis_method)
            key_term_frequencies_in_abstract = key_term_frequency_per_glossary(tmp_glossary_df,
                                                                               abstract,
                                                                               analysis_method)
            key_term_frequencies_in_keywords = key_term_frequency_per_glossary(tmp_glossary_df,
                                                                               keywords_full_text,
                                                                               analysis_method)

            title_freq_dict = key_term_frequencies_in_title
            title_hits_stats = sum(title_freq_dict.values())
            tmp_string = ""
            for key in title_freq_dict.keys():
                tmp_string += f"{key}: {title_freq_dict[key]};" if title_freq_dict[key] > 0 else ""
            title_hits_words = tmp_string

            abstract_freq_dict = key_term_frequencies_in_abstract
            abstract_hits_stats = sum(abstract_freq_dict.values())
            tmp_string = ""
            for key in abstract_freq_dict.keys():
                tmp_string += f"{key}: {abstract_freq_dict[key]};" if abstract_freq_dict[key] > 0 else ""
            abstract_hits_words = tmp_string

            keyword_freq_dict = key_term_frequencies_in_keywords
            keyword_hits_stats = sum(keyword_freq_dict.values())
            tmp_string = ""
            for key in keyword_freq_dict.keys():
                tmp_string += f"{key}: {keyword_freq_dict[key]};" if keyword_freq_dict[key] > 0 else ""
            keyword_hits_words = tmp_string

            total_hits = title_hits_stats + abstract_hits_stats + keyword_hits_stats
            # Calculate a rank score to sort the papers based on it. This score is the measure for paper relevancy. We
            # consider a hit in the title more valuable as hits in the abstract and in the collection of keywords, as
            # in the case of a key term hit within a title, the paper is very much relevant to our search.
            rank_score = (title_hits_stats * 2) + abstract_hits_stats + keyword_hits_stats

            stats_row_to_add = [row['Paper ID'], row['Source ID'], row['Query'],
                                title_hits_stats, abstract_hits_stats, keyword_hits_stats, total_hits, rank_score]
            stats_rows_list.append(stats_row_to_add)

            words_row_to_add = [row['Paper ID'], row['Source ID'], row['Query'],
                                title_hits_words, abstract_hits_words, keyword_hits_words, rank_score]
            words_rows_list.append(words_row_to_add)

        analysis_result_stats_per_query_df = pd.DataFrame(stats_rows_list, columns=analysis_columns_stats)
        analysis_result_stats_per_query_df.sort_values(by=['Rank Score'], ascending=False, inplace=True)
        analysis_result_words_per_query_df = pd.DataFrame(words_rows_list, columns=analysis_columns_words)
        analysis_result_words_per_query_df.sort_values(by=['Rank Score'], ascending=False, inplace=True)
        analysis_result_stats_per_query_dfs.append(analysis_result_stats_per_query_df)
        analysis_result_words_per_query_dfs.append(analysis_result_words_per_query_df)

    analysis_result_stats_df = pd.concat(analysis_result_stats_per_query_dfs, ignore_index=True)
    analysis_result_words_df = pd.concat(analysis_result_words_per_query_dfs, ignore_index=True)

    return analysis_result_stats_df, analysis_result_words_df


def key_term_frequency_per_taxonomy_layers(key_terms_taxonomy_to_search, text, method):
    """
    Description
    :param key_terms_taxonomy_to_search:
    :param text:
    :param method:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    # This dictionary holds different 'per_layer_frequency_dict' structures, with layers as keys.
    layers_frequency_dict = {}
    # This loop iterates over layers, which are keys for the 'key_terms_taxonomy_to_search' dictionary
    for key in key_terms_taxonomy_to_search:
        layer = key
        # This dictionary holds key term frequencies for each key term within a layer.
        per_layer_frequency_dict = key_term_frequency_in_text(key_terms_taxonomy_to_search[layer], text, method)
        layers_frequency_dict[key] = per_layer_frequency_dict

    return layers_frequency_dict


def key_term_frequency_per_glossary(glossary_df, text, method):
    """
    Description
    :param glossary_df:
    :param text:
    :param method:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    # From this point onwards, the pruned keywords collected in glossaries are considered as our key terms. That is the
    # reason behind the variable name 'key_term'.
    frequency_dict = key_term_frequency_in_text(glossary_df['Keyword'], text, method)

    return frequency_dict


def key_term_frequency_in_text(key_terms_to_search, text, method, case=False):
    """
    Extracts key term frequencies in a given text for a given list of key terms.
    :param key_terms_to_search:
    :param text:
    :param method:
    :param case:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    # This dictionary holds frequency results, with key terms as keys.
    frequency_dict = {}
    if method == 'nlp':
        blob_to_analyse = TextBlob(text)
        # If called from 'key_term_frequency_per_taxonomy_layers':
        # This loop iterates over the list of key terms corresponding to a layer in 'key_terms_taxonomy_to_search'
        # (when being called from the function 'key_term_frequency_per_taxonomy_layers') ...
        # OR
        # This loop iterates over the list of key terms in the given glossary, which is coming from the 'keyword'
        # column in the 'glossary_df', as 'key_terms_to_search'.
        for key_term in key_terms_to_search:
            frequency = blob_to_analyse.words.count(key_term, case_sensitive=case)
            frequency_dict[key_term] = frequency

    elif method == 'counter':
        # Initialise 'frequency_dict' with 0 frequencies
        frequency_dict = dict.fromkeys(key_terms_to_search)
        for key_term in key_terms_to_search:
            frequency = 0
            # Check for presence of the 'key_term', which can also be a phrase. That is why functions from packages, or
            # using 'split' might be misleading.
            while text.find(key_term) >= 0:
                frequency += 1
                # Replace the occurrence, but do it only once, in case there are more occurrences.
                text = text.replace(key_term, '', 1)
            frequency_dict[key_term] = frequency

    return frequency_dict


def multiple_key_terms_in_text(key_terms_to_search, text, case=False):
    """
    TODO
    Looks for presence of two or more key terms in a piece of text at the same time. This function addresses scattered
    key terms and does not require them to occur as a sequence.
    E.g., "artificial intelligence" as a reference:
    "Artificial sweeteners ..." -> Nay!
    "Machine intelligence is an example of artificial cognitive capabilities ..." -> Yay!
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    present = False

    return present


def key_term_sequence_in_text(key_terms_to_search, text, case=False):
    """
    TODO
    Looks for presence of two or more keywords in a piece of text at the same time. This function is specifically
    intended for keywords occurring as a sequence.
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    present = False
    ngram_count = len(key_terms_to_search)
    blob_to_analyse = TextBlob(text)
    blob_to_analyse.ngrams(n=ngram_count)

    return present


def collect_high_value_papers(uber_df, key_term_hit_stats_df, head_rows):
    """
    Description
    :param uber_df:
    :param key_term_hit_stats_df:
    :param head_rows:
    :return:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    high_value_dfs_list = []
    queries = uber_df['Query'].unique()
    for query in queries:
        tmp_uber_df = uber_df[uber_df['Query'] == query].copy()
        insert_column(tmp_uber_df, 'Rank Score', 0, reference=1)
        tmp_key_hit_stats_df = key_term_hit_stats_df[key_term_hit_stats_df['Query'] == query]
        for index, row in tmp_key_hit_stats_df.head(head_rows).iterrows():
            tmp_uber_df.loc[tmp_uber_df['Paper ID'] == row['Paper ID'], 'Rank Score'] = row['Rank Score']
        tmp_uber_df.sort_values(by=['Rank Score'], ascending=False, inplace=True)
        tmp_uber_df = tmp_uber_df.head(head_rows)
        high_value_dfs_list.append(tmp_uber_df)

    high_value_df = pd.concat(high_value_dfs_list, ignore_index=True)

    return high_value_df


def convert_to_american_en(text_list):
    """
    Converts all British English word dictations to US English to have more accurate statistics. The choice of
    conversion to US English, and not the other way around, is due to its popularity in academic literature.
    :param text_list:
    :return:
    """
    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    gb_words = ref.gb_to_us_dict.keys()
    us_text_list = []
    for text in text_list:
        text_words = text.split()
        for text_word in text_words:
            if text_word in gb_words:
                us_text_word = ref.gb_to_us_dict[text_word]
                text = text.replace(text_word, us_text_word)
        us_text_list.append(text)

    return us_text_list


def save_analysis_result_report(statistics_dict, statistics_path):
    """
    TODO
    Generates and saves a human readable report for the whole text analysis process and results in text format.
    :param statistics_dict:
    :param statistics_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    pass


def save_analysis_result_csv(statistics_df, file_name, save_path):
    """
    Saves the whole text analysis results as a CSV to have a human-readable alternative.
    :param statistics_df:
    :param file_name:
    :param save_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    tb.log_message(EXECUTION_TIMESTAMP, DEBUG_PATH, 3, "Saving analysis results csv ...")

    queries = statistics_df['Query'].unique()
    for query in queries:
        query_df = statistics_df[statistics_df['Query'] == query]
        save_file_path = save_path / f'{file_name}_{query}.csv'
        query_df.to_csv(save_file_path, index=None, header=True)


def save_high_value_papers_csv(papers_df, file_name, save_path):
    """
    Description ...
    :param papers_df:
    :param file_name:
    :param save_path:
    """

    global EXECUTION_TIMESTAMP
    global DEBUG_PATH

    queries = papers_df['Query'].unique()
    for query in queries:
        query_df = papers_df[papers_df['Query'] == query]
        save_file_path = save_path / f'{file_name}_{query}.csv'
        query_df.to_csv(save_file_path, index=None, header=True)


if __name__ == "__main__":
    main()

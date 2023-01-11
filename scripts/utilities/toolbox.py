"""Description:  """


import time
import sys
import csv
import pickle
import math
import re
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from datetime import datetime

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


# =============================================================================
# =============================================================================
def log_message(execution_timestamp, debug_path, action_id, message, terminal_output=True, file_output=True):
    """
    This is a logging function that depending on the received action id code, prints a log message, while allocating the
    right category to it. Logging is done both in terminal and to a debug file path. Only one of the cases, 'action_id'
    13, results in the application to quit.
    :param execution_timestamp:
    :param debug_path:
    :param action_id:
    :param message:
    :param terminal_output:
    :param file_output:
    """

    global EXECUTION_TIMESTAMP

    debug_file = debug_path / f'debug_{execution_timestamp}.txt'
    with open(debug_file, mode='a') as file:
        # Prepend space to message
        message = "    " + message
        # If the message has multiple lines, prepend space to each new line
        if "\n" in message:
            message = message.replace("\n", "\n    ")

        # Determine the 'action_id' and format the log
        if action_id == 1:
            writable = f"Done: [{get_current_timestamp()}]\n" \
                       f"{message}\n"
            if terminal_output:
                print(writable)
            if file_output:
                file.write(writable)
        elif action_id == 2:
            writable = f"Debug: [{get_current_timestamp()}]\n" \
                       f"{message}\n"
            if terminal_output:
                print(writable)
            if file_output:
                file.write(writable)
        elif action_id == 3:
            writable = f"In progress: [{get_current_timestamp()}]\n" \
                       f"{message}\n"
            if terminal_output:
                print(writable)
            if file_output:
                file.write(writable)
        elif action_id == 4:
            writable = f"Finished: [{get_current_timestamp()}]\n" \
                       f"{message}\n"
            if terminal_output:
                print(writable)
            if file_output:
                file.write(writable)
        elif action_id == 5:
            writable = f"Caution: [{get_current_timestamp()}]\n" \
                       f"{message}\n"
            if terminal_output:
                print(writable)
            if file_output:
                file.write(writable)
        elif action_id == 13:
            writable = f"Major error: [{get_current_timestamp()}]\n" \
                       f"{message}\n"
            if terminal_output:
                print(writable)
            if file_output:
                file.write(writable)
            sys.exit(1)


def save_text(execution_timestamp, save_path, filename, message, terminal_output=False):
    """
    This is a logging function, which is generic for saving text output to a path with a desired filename. The timestamp
    is added to the filename. It includes the option for printing in the Terminal as well.
    :param execution_timestamp:
    :param save_path:
    :param filename:
    :param message:
    :param terminal_output:
    """

    global EXECUTION_TIMESTAMP

    text_file = save_path / f'{filename}_{execution_timestamp}.txt'
    with open(text_file, mode='a') as file:
        writable = f"{message}\n"
        if terminal_output:
            print(message)
        file.write(writable)


def save_csv(execution_timestamp, rows_list, filename, save_path):
    """
    This is a logging function, which generates/saves records in a CSV file. Rows have to be given in a list and each
    row should be dictionaries with headers as their keys.
    :param execution_timestamp:
    :param save_path:
    :param filename:
    :param rows_list:
    """

    global EXECUTION_TIMESTAMP

    csv_file = save_path / f'{filename}_{execution_timestamp}.csv'
    if csv_file.exists():
        file_exists = True
    else:
        file_exists = False

    headers = list(rows_list[0].keys())
    with open(csv_file, 'a') as file:
        csv_writer = csv.DictWriter(file, delimiter=',', quoting=csv.QUOTE_MINIMAL, fieldnames=headers)
        if not file_exists:
            csv_writer.writeheader()
        csv_writer.writerows(rows_list)


def get_current_timestamp():
    """
    Returns the timestamp at the moment of call.
    :return:
    """

    global EXECUTION_TIMESTAMP

    date_time = datetime.now()

    return date_time.time()


def save_object(filename, object_to_save):
    """
    Saves an object using pickle.
    :param filename:
    :param object_to_save:
    """

    global EXECUTION_TIMESTAMP

    with open(filename, 'wb') as file:
        pickle.dump(object_to_save, file)


def load_object(filename):
    """
    Loads a saved pickle object.
    :param filename:
    :return:
    """

    global EXECUTION_TIMESTAMP

    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)

    return loaded_object


def truncate(number, digits) -> float:
    """
    Truncates decimal points of a number, leaving the desired amount of decimal points.
    :param number:
    :param digits:
    :return:
    """

    global EXECUTION_TIMESTAMP

    stepper = 10.0 ** digits

    return math.trunc(stepper * number) / stepper


def get_trailing_number(source_string):
    """
    Detect and returns the trailing number in a string, which can have multiple digits. This is useful for detecting
    numbers in a filename.
    :param source_string:
    :return:
    """

    global EXECUTION_TIMESTAMP

    number_match = re.search(r'\d+$', source_string)

    return int(number_match.group()) if number_match else None


def get_number_after_phrase(source_string, phrase):
    """
    Detects and returns a number occurring after a specific phrase, which can have multiple digits. This is useful for
    detecting numbers in a filename.
    :param source_string:
    :param phrase:
    :return:
    """

    global EXECUTION_TIMESTAMP

    number_match = re.findall(r'{}(\d+)'.format(phrase), source_string)

    return int(number_match[0]) if number_match else None


def get_project_root():
    """
    Returns project root folder.
    :return:
    """

    global EXECUTION_TIMESTAMP

    return Path(__file__).parent.parent


def load_directory_map_from_config_file(config_file):
    global EXECUTION_TIMESTAMP


def convert_str_to_list(list_as_string):
    """
    Description ...
    :param list_as_string:
    :return:
    """

    global EXECUTION_TIMESTAMP

    list_as_string = list_as_string.replace('[', '')
    list_as_string = list_as_string.replace(']', '')
    numeric_list = list(map(int, list_as_string.split(', ')))

    return numeric_list


def convert_str_to_bool(boolean_as_string):
    """
    Description ...
    :param boolean_as_string:
    :return:
    """

    global EXECUTION_TIMESTAMP

    boolean_value = None
    if boolean_as_string == 'True':
        boolean_value = True
    elif boolean_as_string == 'False':
        boolean_value = False

    return boolean_value


def save_classifier_confusion_matrix(true_labels, predicted_labels, label_set, label_names,
                                     classifier_name, save_path):
    """
    Description ...
    :param true_labels:
    :param predicted_labels:
    :param label_set:
    :param label_names:
    :param classifier_name:
    :param save_path:
    """

    global EXECUTION_TIMESTAMP

    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=label_set, normalize='true')
    conf_matrix_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_names)
    conf_matrix_plot = conf_matrix_display.plot(include_values=True, cmap=plt.get_cmap('Blues'), ax=None,
                                                xticks_rotation='horizontal', values_format=None)
    conf_matrix_plot.ax_.set_title(f"Normalised {classifier_name} confusion matrix")
    plot_file = save_path / 'confusion_matrix_best_accuracy.pdf'
    plt.tight_layout()
    plt.savefig(plot_file, format='pdf', dpi=1200)


def file_downloader(url, filename, file_type, save_path):
    """
    Downloads a file from a given URL and saves it on the disk.
    Currently only handles PDF files. To be extended if needed ...
    :param url:
    :param filename:
    :param file_type:
    :param save_path:
    """

    global EXECUTION_TIMESTAMP

    if file_type == 'PDF':
        response = requests.get(url, allow_redirects=True)
        file_path = save_path / filename
        with open(file_path, 'wb') as file:
            file.write(response.content)

        return True

    else:
        return False

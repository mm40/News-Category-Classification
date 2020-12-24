import numpy as np
import math
import re
from pandas import read_json


def attach_token_count_column(df, text_column_name, cnt_column_name):
    """Takes a dataframe, and for each row, splits textColumn by space (" ")
    into tokens, stores the split length into column cnt_column_name.
    NOTE : changes df in-place!

    Args:
        df (DataFrame): dataframe to which to add cnt column
        text_column_name (str): name of the column to split and count parts
        cnt_column_name (str): name of the column to be added, containing counts
    """
    df[cnt_column_name] = df[text_column_name].str.split().str.len()
    return df


def attach_proportional_labels(df, labels, shuffle=True, label_col='LABEL',
                               dropna=True):
    """Attaches column with labels to dataset, with respect to set proportions.

    Args:
     df: dataframe to modify
     labels: dicitonary of {label : proportion,... }. Proportions must sum to 1
             Example : {"train": 0.70, "test": 0.20, "val": 0.10}
     shuffle: whether labels should be shuffled
     label_col: name of the column to be added, to store labels
    Returns:
     Returns the dataframe with additional column label_col, which will contain
     labels with given proportions."""

    if dropna:
        df = df.dropna()
    num_rows = len(df.index)  # fastest way to calculate number of rows
    total_proportions = 0  # must sum to 1
    result_df = df.sample(frac=1) if shuffle else df
    result_df = result_df.assign(**{label_col: np.nan})
    last_index = 0
    last_label = None
    for label in labels.keys():
        current_proportion = labels[label]
        total_proportions += current_proportion

        if total_proportions > 1 and not math.isclose(total_proportions, 1):
            raise ValueError('Proportions for labels should sum to 1. '
                             'Sum is currently : {}'.format(total_proportions))

        new_index = int(last_index + current_proportion * num_rows)
        # Because .iloc receives only integers for both rows and columns,
        # columns.get_loc has to be used to determine column index
        result_df.iloc[last_index:new_index,
                       result_df.columns.get_loc(label_col)] = label
        last_index = new_index
        last_label = label

    if total_proportions < 1 and not math.isclose(total_proportions, 1):
        raise ValueError('Proportions for labels should sum to 1. '
                         'Sum is currently : {}'.format(total_proportions))

    if last_index < num_rows:
        result_df.iloc[last_index:num_rows,
                       result_df.columns.get_loc(label_col)] = last_label
    return result_df


def apply_regex_list_to_text(regex_list, text):
    """Applies regex replacements from regex_list to text, and returns result.

    Args:
        regex_list (list) : list of pairs of strings. First member of pair will
            be what to search for, and second member what to replace it with.
            For example : (("([.,!])", " \1 "), ...)
        text (str): input text to apply regex_list to.

    Returns:
        Text with applied regex (str)
    """
    for pattern, replacement in regex_list:
        text = re.sub(pattern, replacement, text)
    return text


def convert_json_file_to_csv(path_json_in, path_csv_out, only_columns=None):
    """Converts json file loaded from in path to csv file saved to out path

    Args:
        path_json_in (str): filepath to input json file
        path_csv_out (str): filepath to output csv file
        only_columns (list): columns to export. All if none (default None)
    """
    raw_input = read_json(path_json_in, lines=True, orient='columns')
    # Note : further processing of the input file may be needed. See :
    # https://www.kaggle.com/tboyle10/working-with-json-files
    raw_input.to_csv(path_csv_out, index=False, columns=only_columns)

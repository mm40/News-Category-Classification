import numpy as np
import math
import re
import string


def attach_proportional_labels(df, labels, shuffle=True, labelCol='LABEL'):
    """Attaches column with labels to dataset, with respect to set proportions.
    Arguments:
     df: dataframe to modify
     labels: dicitonary of {label : proportion,... }. Proportions must sum to 1
             Example : {"train": 0.70, "test": 0.20, "val": 0.10}
     shuffle: whether labels should be shuffled
     labelCol: name of the column to be added, to store labels
    Output:
     Outputs the dataframe with additional column labelCol, which will contain
     labels with given proportions."""

    num_rows = len(df.index)  # fastest way to calculate number of rows
    total_proportions = 0  # must sum to 1
    result_df = df.sample(frac=1) if shuffle else df
    result_df = result_df.assign(**{labelCol: np.nan})
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
                       result_df.columns.get_loc(labelCol)] = label
        last_index = new_index
        last_label = label

    if total_proportions < 1 and not math.isclose(total_proportions, 1):
        raise ValueError('Proportions for labels should sum to 1. '
                         'Sum is currently : {}'.format(total_proportions))

    if last_index < num_rows:
        result_df.iloc[last_index:num_rows,
                       result_df.columns.get_loc(labelCol)] = last_label
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

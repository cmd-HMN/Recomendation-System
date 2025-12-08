import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast

def split_the_data(data: pd.DataFrame, test_size: int =0.2, threshold: int= 0.5):
    """
    data: pandas DataFrame and should have the `user_index` in it
    test_size: only matter when remaining_data survives the threshold
    threshold: to check if the remaining data is further spliting


    `returns`: (train dataset, test dataset)
    """
    if not ('user_index' in data.columns):
        raise ValueError('The data should have user_index in it')

    atleast_one = data.sample(frac=1).drop_duplicates('user_index')
    remaining_data = data.drop(atleast_one.index)

    if len(remaining_data)/len(atleast_one) <= threshold:
        # train , test
        return atleast_one, remaining_data

    train, test = train_test_split(remaining_data, test_size=test_size, random_state=67)

    return pd.concat([train, atleast_one]), test


def convert_to_list(data):
    """
    Single list or simply for row
    """
    return ast.literal_eval(data)

def convert_series_to_list(data: pd, warn=False, add_nan=True):
    """
    Here data is the pd.Series
    """
    return [skill for sublist in data for skill in sublist]

def get_unique_skills(course, courses):
    l1 = convert_series_to_list(course['skills'])
    l2 = convert_series_to_list(courses['skills'])
    l = l1 + l2
    return set(l)
import numpy as np
import pandas as pd
import math
from math import floor


def sigmoid_rld(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0


def fix_flux(row):
    if 20 >= row['mjd'] >= 7 and row['flux'] <= 0:
        row['flux'] = sigmoid_rld(row['flux'])
    return row


def data_manipulating(training_set, metadata):
    norm_dister = {0: np.random.normal(350, 15, 1)[0],
                   1: np.random.normal(500, 33, 1)[0],
                   2: np.random.normal(600, 33, 1)[0],
                   3: np.random.normal(750, 33, 1)[0],
                   4: np.random.normal(875, 25, 1)[0],
                   5: np.random.normal(1000, 15, 1)[0]}
    training_set['passband'] = training_set['passband'].map(norm_dister)
    training_set['mjd'] = training_set['mjd'].apply(lambda a: floor((((a - floor(a)) * 1440) / 60)))
    training_set = training_set.apply(fix_flux, axis=1)
    training_set.drop(columns=["detected"], inplace=True)
    metadata['hostgal_specz'] = metadata['hostgal_specz'].apply(lambda x: 0 if np.isnan(x) else x)
    metadata['distmod'] = metadata['distmod'].apply(lambda x: 0 if np.isnan(x) else x)
    training_set = training_set.groupby('object_id').mean()
    training_set['object_id'] = training_set.index
    return training_set, metadata

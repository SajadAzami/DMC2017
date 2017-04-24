import pandas as pd
import numpy as np
from pathlib import Path
from preprocessing.data_preparation import read_data

#merge items and train data to a dataFrame
def merge_data(input_path, items_path, output_path):
    tdf = read_data(input_path)
    idf = read_data(items_path)
    output = Path(output_path)
    if not output.is_file():
        mrg = pd.merge(tdf, idf)
        pd.to_pickle(mrg, output_path)
        return mrg
    else:
        return pd.read_pickle(output_path)

# example of using merge_data function for train dataset
mrg = merge_data('../data/train.csv', '../data/items.csv', '../data/train_merged.pkl')

# uppercase all pharmForm values
mrg['pharmForm'] = mrg['pharmForm'].str.upper()

# add count feature (revenue/price)
mrg['count'] = mrg.revenue/mrg.price

# extract pharmForm values as binary feature and adding them to dataset
mrg = pd.concat([mrg, pd.get_dummies(mrg['pharmForm'])], axis=1)
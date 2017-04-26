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

def extract_numbers_from_content(input):
    x_index = input.find('X')
    if x_index == -1:
        return 1, input
    else:
        return input[0 : x_index], input[x_index+1 : len(input)]

def prepare_dataset():
    # example of using merge_data function for train dataset
    mrg = merge_data('../data/train.csv', '../data/items.csv', '../data/train_merged.pkl')

    # uppercase all pharmForm values
    mrg['pharmForm'] = mrg['pharmForm'].str.upper()

    # add count feature (revenue/price)
    mrg['count'] = mrg.revenue/mrg.price

    # extract pharmForm values as binary feature and adding them to dataset
    mrg = pd.concat([mrg, pd.get_dummies(mrg['pharmForm'])], axis=1)

    # split count of packs and amount of each to separate columns
    extracted_numbers = mrg['content'].apply(extract_numbers_from_content)
    extracted_numbers = pd.DataFrame(extracted_numbers.tolist(), columns=['content_count', 'content_pack_amount'], index=extracted_numbers.index)
    mrg = pd.concat([mrg, extracted_numbers], axis=1)

    # fill campaignIndex with 4 and then replace other characters with numbers
    mrg['campaignIndex'].fillna(4, inplace=True)
    mapping = {'A': 1, 'B': 2, 'C': 3}
    mrg = mrg.replace({'campaignIndex': mapping})
    return mrg

prepare_dataset()
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_squared_error, make_scorer

from preprocessing.data_preparation import read_data
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold


# merge items and train data to a dataFrame
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
    if input == 'L   125':
        return 1, 1, 125
    if x_index == -1:
        if input == 'PAK':
            return 1, 1, 1
        return 1, 1, input
    second_part = input[x_index + 1: len(input)]
    x_second_index = second_part.find('X')
    if x_second_index == -1:
        return 1, input[0: x_index], second_part
    return input[0: x_index], second_part[0: x_second_index], second_part[x_second_index + 1: len(second_part)]


def prepare_dataset():
    # example of using merge_data function for train dataset
    mrg = merge_data('../data/train.csv', '../data/items.csv', '../data/train_merged.pkl')
    print('completed')

    # add count feature (revenue/price)
    mrg['count'] = mrg.revenue / mrg.price

    # uppercase all pharmForm values
    mrg['pharmForm'] = mrg['pharmForm'].str.upper()
    # extract pharmForm values as binary feature and adding them to dataset
    mrg = pd.concat([mrg, pd.get_dummies(mrg['pharmForm'])], axis=1)
    mrg = mrg.drop('pharmForm', 1)

    # split count of packs and amount of each to separate columns
    extracted_numbers = mrg['content'].apply(extract_numbers_from_content)
    extracted_numbers = pd.DataFrame(extracted_numbers.tolist(), columns=['content_1', 'content_2', 'content_3'],
                                     index=extracted_numbers.index)
    extracted_numbers['content_1'] = pd.to_numeric(extracted_numbers['content_1'])
    extracted_numbers['content_2'] = pd.to_numeric(extracted_numbers['content_2'])
    extracted_numbers['content_3'] = pd.to_numeric(extracted_numbers['content_3'])
    mrg = pd.concat([mrg, extracted_numbers], axis=1)
    mrg = mrg.drop('content', 1)

    # fill campaignIndex with 4 and then replace other characters with numbers
    mrg['campaignIndex'].fillna('D', inplace=True)
    mrg = pd.concat([mrg, pd.get_dummies(mrg['campaignIndex'])], axis=1)
    mrg = mrg.drop('campaignIndex', 1)

    mrg = mrg.drop('category', 1)

    # mrg = pd.concat([mrg, pd.get_dummies(mrg['group'])], axis=1)
    mrg = mrg.drop('group', 1)

    mrg = pd.concat([mrg, pd.get_dummies(mrg['unit'])], axis=1)
    mrg = mrg.drop('unit', 1)
    return mrg


def predict_competitor(all_data):
    train = all_data[pd.notnull(all_data['competitorPrice'])]
    kf = KFold(n_splits=10)
    estimator = XGBRegressor()
    x = train.drop('competitorPrice', 1)
    y = train['competitorPrice']
    scores = cross_val_score(estimator,
                             x,
                             y,
                             cv=kf,
                             scoring=make_scorer(mean_squared_error))
    print(scores)

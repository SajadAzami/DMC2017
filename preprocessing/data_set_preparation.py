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
    print('data read successfully!')
    output = Path(output_path)
    if not output.is_file():
        train_merged = tdf.copy()
        train_merged = train_merged.merge(tdf.merge(idf, how='left', on='pid', sort=False))
        pd.to_pickle(train_merged, output_path)
        return train_merged
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


unit_map = {
    'KG': 1000,
    'ST': 6350,
    'P': 454,
    'M': 100,
    'L': 1000,
    'G': 1,
    'CM': 1,
    'ML': 1,
}


def unit_converter(row):
    return row['content_3'] * unit_map[row['unit']]


def prepare_dataset():
    output = Path('../data/unit_fixed.pkl')
    if not output.is_file():
        # example of using merge_data function for train dataset
        mrg = merge_data('../data/train.csv', '../data/items.csv', '../data/train_merged.pkl')

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

        mrg['content_3'] = mrg.apply(unit_converter, axis=1)
        mapping = {'KG': 'G', 'ST': 'G', 'P': 'G', 'L': 'ML', 'M': 'CM'}
        mrg = mrg.replace({'unit': mapping})
        pd.to_pickle(mrg, '../data/unit_fixed.pkl')
        print('units converted')
    else:
        mrg = pd.read_pickle('../data/unit_fixed.pkl')

    # fills missing values for campaignIndex
    campaign_missing = data[pd.isnull(data['campaignIndex'])]['lineID']
    adFlag_missing = data[data['adFlag'] == 0]['lineID']
    intersections = pd.Series(list(set(campaign_missing).intersection(set(adFlag_missing))))

    # These are lineIDs with missing campaignIndex and adFlag=0
    ind = data.lineID.isin(intersections.tolist())

    # To be filled with D
    data['campaignIndex'].fillna(data[ind]['campaignIndex'].fillna('D'), inplace=True)

    mrg = pd.concat([mrg, pd.get_dummies(mrg['campaignIndex'])], axis=1)
    mrg = mrg.drop('campaignIndex', 1)

    # mrg = pd.concat([mrg, pd.get_dummies(mrg['group'])], axis=1)
    mrg = mrg.drop('group', 1)
    return mrg


def predict_competitor(all_data):
    train = all_data[pd.notnull(all_data['competitorPrice'])]
    kf = KFold(n_splits=10)
    estimator = XGBRegressor()
    x = train.drop('competitorPrice', 1)
    y = train['competitorPri1ce']
    scores = cross_val_score(estimator,
                             x,
                             y,
                             cv=kf,
                             scoring=make_scorer(mean_squared_error))
    print(scores)


data = prepare_dataset()

# from scipy.stats import pearsonr
#
# print(data['category'].fillna(0))
# print(pearsonr(data['category'].fillna(0), data['count']))


# TODO handle features: category(linear model), group(feature extraction), competitor(linear model)
# TODO Random Forrest on server

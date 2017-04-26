import pandas as pd
import numpy as np
from pathlib import Path
from preprocessing.data_preparation import read_data


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


# example of using merge_data function for train dataset
# mrg = merge_data('../data/train.csv', '../data/items.csv', '../data/train_merged.pkl')


mrg = merge_data('../data/train.csv', '../data/items.csv', '../data/merged.pkl')
print('merge finished!')

# uppercase all pharmForm values
mrg['pharmForm'] = mrg['pharmForm'].str.upper()

# add count feature (revenue/price)
mrg['count'] = mrg.revenue / mrg.price

# extract pharmForm values as binary feature and adding them to dataset
# mrg = pd.concat([mrg, pd.get_dummies(mrg['pharmForm'])], axis=1)

mrg.to_csv('../data/train_merged.csv')

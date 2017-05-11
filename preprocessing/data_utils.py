import pandas as pd
from pathlib import Path

DATA_MERGED_PICKLE = 'data_merged_pickle.pkl'
DATA_FINAL_PICKLE = 'data_final_pickle.pkl'


def load_data(path, label_name=None, drop_cols=None, mode='csv'):
    if mode == 'csv':
        data = pd.read_csv(path)
    else:
        data = pd.read_pickle(path)

    if drop_cols:
        data = data.drop(drop_cols, axis=1)

    if label_name is None:
        return data
    else:
        label = data[label_name]
        data = data.drop(label_name, axis=1)
        return data, label


def check_if_file_exists(path):
    file = Path(path)
    if file.is_file():
        return file
    else:
        return None


# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing(data):
    missing = data.columns[data.isnull().any()].tolist()
    return data[missing].isnull().sum()


# Get missing value counts of a dataframe
def get_missing_count(df):
    return df.isnull().values.ravel().sum()


def merge_data(train_df, items_df):
    train_merged = train_df.copy()
    train_merged = train_merged.merge(train_df.merge(items_df, how='left', on='pid', sort=False))
    return train_merged


def merge_data2(input_path, items_path=None, output_path=None, items_df=None):
    tdf = load_data(input_path)
    if isinstance(items_df, pd.DataFrame):
        idf = items_df
    else:
        idf = load_data(items_path)
    print('data read successfully!')
    output = Path(output_path)
    if not output.is_file():
        train_merged = tdf.copy()
        train_merged = train_merged.merge(tdf.merge(idf, how='left', on='pid', sort=False))
        pd.to_pickle(train_merged, output_path)
        return train_merged
    else:
        return pd.read_pickle(output_path)

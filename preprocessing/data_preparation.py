import pandas as pd


# Reads train data from csv, returns pandas DF
def read_data(path):
    return pd.read_csv(path)


# Reads train data from csv, drops target label and returns pandas DF
def read_data_label(path, label_name):
    data = pd.read_csv(path)
    label = data[label_name]
    data = data.drop(label_name, axis=1)
    return data, label


# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing(data):
    missing = data.columns[data.isnull().any()].tolist()
    return data[missing].isnull().sum()

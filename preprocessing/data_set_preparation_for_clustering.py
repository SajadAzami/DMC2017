import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, make_scorer
# from preprocessing.data_preparation import read_data
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB

def read_data(path):
    return pd.read_csv(path)

# merge items and train data to a dataFrame
def merge_data(input_path, items_path=None, output_path=None, items_df=None):
    tdf = read_data(input_path)
    if isinstance(items_df, pd.DataFrame):
        idf = items_df
    else:
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

def fill_competitor_missings(data):
    df = data[['lineID', 'day', 'weekDay', 'rrp', 'price', 'competitorPrice']]
    train = df[pd.notnull(df['competitorPrice'])]
    train = train[train['competitorPrice'] != 0]

    x = train[['day', 'weekDay', 'rrp', 'price']]
    y = train['competitorPrice']
    T_train_xgb = xgb.DMatrix(x, y)

    params = {"objective": "reg:linear", "booster": "gblinear"}
    gbm = xgb.train(dtrain=T_train_xgb, params=params)

    competitor_missing_ids = set(df['lineID']) - set(train['lineID'])
    na_rows = df[['day', 'weekDay', 'rrp', 'price']][df['lineID'].isin(competitor_missing_ids)]
    y_pred = gbm.predict(xgb.DMatrix(na_rows))
    data.ix[data['lineID'].isin(competitor_missing_ids), 'competitorPrice'] = y_pred
    return data

def prepare_items():
    items = pd.read_csv('../data/items.csv')

    # uppercase all pharmForm values
    items['pharmForm'] = items['pharmForm'].str.upper()

    # split count of packs and amount of each to separate columns
    extracted_numbers = items['content'].apply(extract_numbers_from_content)
    extracted_numbers = pd.DataFrame(extracted_numbers.tolist(), columns=['content_1', 'content_2', 'content_3'],
                                     index=extracted_numbers.index)
    extracted_numbers['content_1'] = pd.to_numeric(extracted_numbers['content_1'])
    extracted_numbers['content_2'] = pd.to_numeric(extracted_numbers['content_2'])
    extracted_numbers['content_3'] = pd.to_numeric(extracted_numbers['content_3'])
    items = pd.concat([items, extracted_numbers], axis=1)
    items = items.drop('content', 1)

    items['content_3'] = items.apply(unit_converter, axis=1)
    mapping = {'KG': 'G', 'ST': 'G', 'P': 'G', 'L': 'ML', 'M': 'CM'}
    items = items.replace({'unit': mapping})
    items = pd.concat([items, pd.get_dummies(items['unit'])], axis=1)
    items = items.drop('unit', 1)
    x_train = items[pd.notnull(items['category'])]
    y_train = x_train['category']
    pids = set(items['pid']) - set(x_train['pid'])
    x_train = x_train[["manufacturer", "content_1", "content_2", "content_3", "G", "ML", "CM", "genericProduct", "salesIndex", "rrp"]]
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=8, weights='distance', n_jobs=3)
    classifier.fit(x_train, y_train)
    x_test = items[items['pid'].isin(pids)]
    x_test = x_test[["manufacturer", "content_1", "content_2", "content_3", "G", "ML", "CM", "genericProduct", "salesIndex", "rrp"]]
    y_pred = classifier.predict(x_test)
    items.ix[items['pid'].isin(pids), 'category'] = y_pred

    # extract pharmForm values as binary feature and adding them to dataset
    # items = pd.concat([items, pd.get_dummies(items['pharmForm'])], axis=1)
    # items = items.drop('pharmForm', 1)
    return items

def prepare_dataset():
    output = Path('../data/train_v2.pkl')
    if not output.is_file():
        output = Path('../data/unit_fixed_v2.pkl')
        if not output.is_file():
            items = prepare_items()
            data = merge_data(input_path='../data/train.csv', items_df=items, output_path='../data/train_merged_v2.pkl')
            # add count feature (revenue/price)
            data['count'] = data.revenue / data.price

            pd.to_pickle(data, '../data/unit_fixed_v2.pkl')
            print('units converted')
        else:
            data = pd.read_pickle('../data/unit_fixed_v2.pkl')

        data['weekDay'] = data['day'] % 7

        data = fill_competitor_missings(data)
        # fills missing values for campaignIndex
        campaign_missing = data[pd.isnull(data['campaignIndex'])]['lineID']
        adFlag_missing = data[data['adFlag'] == 0]['lineID']
        intersections = pd.Series(list(set(campaign_missing).intersection(set(adFlag_missing))))

        # These are lineIDs with missing campaignIndex and adFlag=0
        ind = data.lineID.isin(intersections.tolist())

        # To be filled with D
        data['campaignIndex'].fillna(data[ind]['campaignIndex'].fillna('D'), inplace=True)

        # Filling the rest using naive bayes
        train_data = data[pd.notnull(data['campaignIndex'])]
        test_data = data[pd.isnull(data['campaignIndex'])]

        naive_bayes_clf = GaussianNB()
        naive_bayes_clf.fit(train_data[['pid', 'manufacturer', 'rrp']],
                            train_data['campaignIndex'])
        predictions = naive_bayes_clf.predict(
            test_data[['pid', 'manufacturer', 'rrp']])

        data.ix[data['lineID'].isin(test_data['lineID']),
                'campaignIndex'] = predictions
        # campaignIndex filled completely

        # data = pd.concat([data, pd.get_dummies(data['campaignIndex'])], axis=1)
        # data = data.drop('campaignIndex', 1)

        # data = pd.concat([data, pd.get_dummies(data['group'])], axis=1)
        # data = data.drop('group', 1)
        pd.to_pickle(data, '../data/train_v2.pkl')
    else:
        data = pd.read_pickle('../data/train_v2.pkl')
    return data

''' unused function for model selection '''
def predict_competitor(all_data):
    train = all_data[pd.notnull(all_data['competitorPrice'])]
    kf = KFold(n_splits=10)
    estimator = xgb.XGBRegressor()
    x = train.drop('competitorPrice', 1)
    y = train['competitorPrice']
    scores = cross_val_score(estimator,
                             x,
                             y,
                             cv=kf,
                             scoring=make_scorer(mean_squared_error))
    print(scores)

prepare_dataset()

'''unused function finds best number of neighbors for knn of category feature'''
def find_best_number_of_neighbors_knn():
    items = pd.read_csv('../data/items.csv')
    items['pharmForm'] = items['pharmForm'].str.upper()
    items = pd.concat([items, pd.get_dummies(items['pharmForm'])], axis=1)
    items = items.drop('pharmForm', 1)

    extracted_numbers = items['content'].apply(extract_numbers_from_content)
    extracted_numbers = pd.DataFrame(extracted_numbers.tolist(), columns=['content_1', 'content_2', 'content_3'],
                                     index=extracted_numbers.index)
    extracted_numbers['content_1'] = pd.to_numeric(extracted_numbers['content_1'])
    extracted_numbers['content_2'] = pd.to_numeric(extracted_numbers['content_2'])
    extracted_numbers['content_3'] = pd.to_numeric(extracted_numbers['content_3'])
    items = pd.concat([items, extracted_numbers], axis=1)
    items = items.drop('content', 1)

    items['content_3'] = items.apply(unit_converter, axis=1)
    mapping = {'KG': 'G', 'ST': 'G', 'P': 'G', 'L': 'ML', 'M': 'CM'}
    items = items.replace({'unit': mapping})
    items = pd.concat([items, pd.get_dummies(items['unit'])], axis=1)
    items = items.drop('unit', 1)
    items = items[["manufacturer", "group", "content_1", "content_2", "content_3", "G", "ML", "CM", "genericProduct", "salesIndex", "rrp", 'category']]
    items = items[pd.notnull(items['category'])]
    x = items[["manufacturer", "content_1", "content_2", "content_3", "G", "ML", "CM", "genericProduct", "salesIndex", "rrp"]]
    y = items['category']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV

    param_grid = {"n_neighbors": np.linspace(start=3, stop=99, dtype=np.int32),
        "weights": ['uniform', 'distance']}

    model = KNeighborsClassifier()

    from sklearn.metrics import accuracy_score, make_scorer
    accuracy_scorer = make_scorer(accuracy_score)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=accuracy_scorer, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(grid.best_score_)
    print(grid.best_params_)
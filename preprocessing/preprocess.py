from .preprocess_items import ItemsPreprocessor
from .preprocess_train import TrainProcessor
from .data_utils import *


def preprocess():
    items_processor = ItemsPreprocessor()
    items_processor.prepare()

    if check_if_file_exists('../data/{filename}'.format(filename=DATA_FINAL_PICKLE)):
        data_df = load_data('../data/{filename}'.format(filename=DATA_FINAL_PICKLE))
    else:
        train_processor = TrainProcessor(items_processor.items_df)
        train_processor.prepare()

        data_df = train_processor.data_df

    return data_df

from keras.models import Sequential
from keras.layers import Dense, Dropout

from preprocessing.data_utils import load_data, DATA_FINAL_PICKLE


def simple_nn():
    data_df, labels = load_data('../data/{filename}'.format(filename=DATA_FINAL_PICKLE),
                                label_name='order',
                                drop_cols=['count', 'group', 'lineID', 'click', 'basket'],
                                mode='pickle')
    data_df = data_df.as_matrix()
    labels = labels.as_matrix()

    # model = Sequential([
    #     Dense(units=8192, activation='relu', input_dim=215),
    #     Dropout(0.8),
    #     Dense(units=4096, activation='relu', input_dim=215),
    #     Dropout(0.7),
    #     Dense(units=2048, activation='relu', input_dim=100),
    #     Dropout(0.5),
    #     Dense(units=1024, activation='relu', input_dim=100),
    #     Dropout(0.3),
    #     Dense(units=256, activation='relu', input_dim=100),
    #     Dropout(0.),
    # ])
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=data_df.shape[1]))
    model.add(Dense(128, activation='relu', input_dim=100))
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(data_df, labels, epochs=10, batch_size=32)


if __name__ == '__main__':
    simple_nn()

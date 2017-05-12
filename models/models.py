from keras.models import Sequential
from keras.layers import Dense, Dropout

from preprocessing.data_utils import load_data, split_train_val_test, data_target, DATA_FINAL_PICKLE


class NeuralNets:
    def __init__(self):
        self.load_data()

    def load_data(self):
        data_df = load_data('../data/{filename}'.format(filename=DATA_FINAL_PICKLE),
                            drop_cols=['count', 'group', 'click', 'basket'],
                            mode='pkl')
        train_df, val_df, test_df = split_train_val_test(data_df)

        train_df, self.train_target = data_target(train_df, 'order')
        val_df, self.val_target = data_target(val_df, 'order')
        test_df, self.test_target = data_target(test_df, 'order')
        self.train_df = train_df.as_matrix()
        self.val_df = val_df.as_matrix()
        self.test_df = test_df.as_matrix()

    def simple_nn(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=self.train_df.shape[1]))
        model.add(Dense(128, activation='relu', input_dim=100))
        model.add(Dense(32, activation='relu', input_dim=100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.train_df, self.train_target,
                  validation_data=(self.val_df, self.val_target),
                  epochs=10,
                  batch_size=32)
        model.save('simple_nn.h5')

    def complex_nn(self):
        model = Sequential([
            Dense(2048, activation='relu', input_dim=self.train_df.shape[1]),
            Dropout(0.5),
            Dense(512, activation='relu', input_dim=150),
            Dropout(0.5),
            Dense(128, activation='relu', input_dim=100),
            Dropout(0.5),
            Dense(32, activation='relu', input_dim=50),
            Dropout(0.5),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.train_df, self.train_target,
                  validation_data=(self.val_df, self.val_target),
                  epochs=10,
                  batch_size=32)


if __name__ == '__main__':
    nn = NeuralNets()
    nn.simple_nn()

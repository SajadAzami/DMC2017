from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import preprocessing
from sklearn.metrics import classification_report, precision_score, recall_score, auc, roc_curve
import numpy as np
from sklearn.utils import class_weight

from preprocessing.data_utils import load_data, split_train_val_test, data_target, DATA_FINAL_PICKLE

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


class NeuralNets:
    def __init__(self):
        self.load_data()

    def load_data(self):
        data_df = load_data('../data/{filename}'.format(filename=DATA_FINAL_PICKLE),
                            drop_cols=['count', 'click', 'basket', 'revenue'],
                            mode='pkl')
        train_df, val_df, test_df = split_train_val_test(data_df)

        train_df, self.train_target = data_target(train_df, 'order')
        val_df, self.val_target = data_target(val_df, 'order')
        test_df, self.test_target = data_target(test_df, 'order')

        self.train_df = preprocessing.normalize(train_df, axis=1)
        self.val_df = preprocessing.normalize(val_df, axis=1)
        self.test_df = preprocessing.normalize(test_df, axis=1)

        # self.train_df = train_df.as_matrix()
        # self.val_df = val_df.as_matrix()
        # self.test_df = test_df.as_matrix()

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
                  validation_data=(self.val_df, self.val_target))
        model.save('simple_nn_v1.h5')
        return model

    def complex_nn(self):
        print('complex nn')
        model = Sequential([
            Dense(256, activation='relu', input_dim=self.train_df.shape[1]),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        class_weights = class_weight.compute_class_weight('auto', np.unique(self.train_target), self.train_target)
        print('class weights')
        print(class_weights)
        model.fit(self.train_df, self.train_target,
                  validation_data=(self.val_df, self.val_target),
                  epochs=1,
                  class_weight=class_weights)
        model.save('complex_nn_v1.h5')
        return model

    def evaluate_model(self, model):
        print()
        print('target preds')
        target_pred = model.predict_classes(self.test_df)
        report = classification_report(self.test_target, target_pred)
        print()
        print('classification report')
        print(report)
        precision = precision_score(self.test_target, target_pred)
        print('precision score')
        print(precision)
        recall = recall_score(self.test_target, target_pred)
        print('recall score')
        print(recall)
        fpr, tpr, thresholds = roc_curve(self.test_target, target_pred, pos_label=2)
        auc_metric = auc(fpr, tpr)
        print('auc')
        print(auc_metric)
        print('target prediction probs')
        y_pred_prob = model.predict_proba(self.test_df)
        print(y_pred_prob)


if __name__ == '__main__':
    nn = NeuralNets()
    # model = nn.simple_nn()
    model = nn.complex_nn()
    nn.evaluate_model(model)

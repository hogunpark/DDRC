'''
Author: Hogun park
Date: 10.13.2016
Goal: To provide a way to build, train, eval a neural netowrk model

'''

import os
import numpy as np
from keras.models import Sequential, Graph
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.core import TimeDistributedDense, Masking, Reshape, Flatten, Activation, RepeatVector, Permute, Highway, Dropout, Merge, Lambda, TimeDistributedDense
from keras.layers import Input, Dense, Dropout, MaxPooling1D, MaxPooling2D
from keras.regularizers import WeightRegularizer, l1
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.utils import np_utils

class nn_model:

    def __init__(self, maxseq, n_users, n_attr, n_classes, b_parallel, neighbSum, perfectNeighbors, rInit, patience):
        self.d_id_pred = {}
        self.model = Sequential()
        self.maxseq = maxseq
        self.n_users = n_users
        self.n_attr = n_attr
        self.n_classes = n_classes
        self.b_parallel = b_parallel
        self.neighbSum = neighbSum
        self.perfectNeighbors = perfectNeighbors
        self.rInit = rInit
        self.patience = patience


    def argmax_ndarray(self, a):
        l_argmax = []
        for i in range(a.shape[0]):
            if np.isnan(np.max(a[i])):
                l_argmax.append(0)
            else:
                l_argmax.append(np.nonzero(a[i].max() == a[i])[0][0])
        return np.array(l_argmax)

    def rand_data(self, x_train, attr_train, y_train, id_train, seed):
        np.random.seed(seed)
        np.random.shuffle(x_train)
        np.random.seed(seed)
        np.random.shuffle(attr_train)
        np.random.seed(seed)
        np.random.shuffle(y_train)
        np.random.seed(seed)
        np.random.shuffle(id_train)
        return x_train, attr_train, y_train, id_train

    def assign_pred_dic(self, l_pred, n_ids, n_classes):
        d_id_pred = {}
        for i in range(n_ids.shape[0]):
            d_id_pred[n_ids[i]] = np_utils.to_categorical([l_pred[i]], n_classes)
        return d_id_pred

    # generate train batches in paralell
    def generate_arrays_train(self, X, attrX, Y, batch_size, maxseq, n_users, n_attr, dntype_option):
        while 1:
            if dntype_option == 3 or dntype_option == 4:
                n_feats = n_users
            else:
                n_feats = n_users + n_attr

            size = Y.shape[0]
            nb_batch = int(np.ceil(size / float(batch_size)))
            for z in range(0, nb_batch):
                start = int(z * batch_size)
                end = int(min(size, (z + 1) * batch_size))
                # print start
                # print end
                x_data = [a.toarray() for a in X[start:end]]
                attr_data = [a.toarray() for a in attrX[start:end]]
                y_data = Y[start:end, :]
                if dntype_option == 5: #dnn
                    x_data = np.reshape(x_data, ((end - start), n_feats))
                else:
                    x_data = np.reshape(x_data, ((end-start), maxseq, n_feats))
                attr_data = np.reshape(attr_data , ((end-start), n_attr))

                if dntype_option == 3:
                    yield ([x_data, attr_data], y_data)
                elif dntype_option == 4:
                    yield ([x_data, attr_data, attr_data], y_data)
                else:
                    yield (x_data, y_data)

    # add class_dist of neighbors
    def add_class_to_seq(self, X, d_id_pred, n_users, n_classes):
        np_dist = np.zeros((n_classes))
        for i in range(X.shape[0]):
            f_user = X[i, :n_users]
            l_nonzero = f_user.nonzero()[0]
            for idx in l_nonzero:
                if idx in d_id_pred:
                    np_dist = np_dist + d_id_pred[idx]
        if np.sum(np_dist) != 0:
            np_dist = np_dist / float(np.sum(np_dist))
        np_class = np.empty((0, n_classes))
        for i in range(X.shape[0]):
            np_class = np.vstack((np_class, np_dist))
        X = np.hstack((X, np_class))
        return X

    # generate train batches in paralell
    def generate_arrays_train_withneighbors(self, X, Y, batch_size, maxseq, n_users, n_attr, n_classes, d_id_pred):
        while 1:

            for i in range(0, len(X) - 1, batch_size):
                size = min(len(X), i + batch_size) - i
                x = self.add_class_to_seq(X[i].toarray(), d_id_pred, n_users, n_classes)
                for j in range(size - 1):
                    x2 = self.add_class_to_seq(X[i + j + 1].toarray(), d_id_pred, n_users, n_classes)
                    x = np.vstack((x, x2))
                x = np.reshape(x, (size, maxseq, n_users + n_attr + n_classes))
                y = Y[i:i + size, :]
                y = np.reshape(y, (size, Y.shape[1]))
                # print "\t#### train ", i, "th", x.shape, y.shape, "size-", size
                yield (x, y)


    # generate test batches in paralell
    def generate_arrays_test(self, X, attrX, Y, maxseq, n_users, n_attr, dntype_option):
        while 1:
            if dntype_option == 3 or dntype_option == 4:
                n_feats = n_users
            else:
                n_feats = n_users + n_attr

            for i in range(len(X)):
                x = X[i].toarray()
                attrx = attrX[i].toarray()
                if dntype_option == 5:
                    x = np.reshape(x, (1, n_feats))
                else:
                    x = np.reshape(x, (1, maxseq, n_feats))
                attrx = np.reshape(attrx, (1, n_attr))
                y = Y[i, :]
                y = np.reshape(y, (1, y.shape[0]))
                if dntype_option == 3:
                    yield ([x, attrx], y)
                elif dntype_option == 4:
                    yield ([x, attrx, attrx], y)
                else:
                    yield (x, y)

    def generate_arrays_test_for_prediction(self, X, attrX, maxseq, n_users, n_attr, dntype_option):
        while 1:
            if dntype_option == 3 or dntype_option == 4:
                n_feats = n_users
            else:
                n_feats = n_users + n_attr
            for i in range(len(X)):
                x = X[i].toarray()
                attrx = attrX[i].toarray()
                if dntype_option == 5:
                    x = np.reshape(x, (1, n_feats))
                else:
                    x = np.reshape(x, (1, maxseq, n_feats))
                attrx = np.reshape(attrx, (1, n_attr))
                if dntype_option == 3:
                    yield ([x, attrx])
                elif dntype_option == 4:
                    yield ([x, attrx, attrx])
                else:
                    yield (x)

    def generate_arrays_test_withneighbors(self, X, Y, maxseq, n_users, n_attr, n_classes, d_id_pred):
        while 1:
            for i in range(len(X)):
                x = self.add_class_to_seq(X[i].toarray(), d_id_pred, n_users, n_classes)
                x = np.reshape(x, (1, maxseq, n_users + n_attr + n_classes))
                y = Y[i, :]
                y = np.reshape(y, (1, y.shape[0]))
                yield (x, y)

    def generate_arrays_test_withneighbors_for_prediction(self, X, maxseq, n_users, n_attr, n_classes, d_id_pred):
        while 1:
            for i in range(len(X)):
                x = self.add_class_to_seq(X[i].toarray(), d_id_pred, n_users, n_classes)
                x = np.reshape(x, (1, maxseq, n_users + n_attr + n_classes))
                yield (x)

    # generate validation batches in paralell
    def generate_arrays_eval(self, X, attrX, Y, maxseq, n_users, n_attr, dntype_option):
        while 1:
            if dntype_option == 3 or dntype_option == 4:
                n_feats = n_users
            else:
                n_feats = n_users + n_attr

            for i in range(len(X)):
                x = X[i].toarray()
                attrx = attrX[i].toarray()
                if dntype_option == 5:
                    x = np.reshape(x, (1, n_feats))
                else:
                    x = np.reshape(x, (1, maxseq, n_feats))
                attrx = np.reshape(attrx, (1, n_attr))
                y = Y[i, :]
                y = np.reshape(y, (1, y.shape[0]))
                # print "\t#### test ", i, "th", x.shape, y.shape
                if dntype_option == 3:
                    yield ([x, attrx], y)
                elif dntype_option == 4:
                    yield ([x, attrx, attrx], y)

                else:
                    yield (x, y)

    def generate_arrays_eval_withneighbors(self, X, Y, maxseq, n_users, n_attr, n_classes, d_id_pred):
        while 1:
            for i in range(len(X)):
                x = self.add_class_to_seq(X[i].toarray(), d_id_pred, n_users, n_classes)
                x = np.reshape(x, (1, maxseq, n_users + n_attr + n_classes))
                y = Y[i, :]
                y = np.reshape(y, (1, y.shape[0]))
                # print "\t#### test ", i, "th", x.shape, y.shape
                yield (x, y)

    def evaluate_model(self, x_test, y_test, attr_test, model_file, batch_size):
        # Load the best weights
        if os.path.isfile(model_file):
            self.model.load_weights(model_file)

        if self.b_parallel:
            if self.dntype_option == 2:
                if self.neighbSum:
                    score, acc = self.model.evaluate_generator(self.generate_arrays_test_withneighbors(
                        x_test, y_test, self.maxseq, self.n_users, self.n_attr, self.n_classes, self.d_id_pred),
                        val_samples=y_test.shape[0])
            else:
                score, acc = self.model.evaluate_generator(self.generate_arrays_test(
                    x_test, attr_test, y_test, self.maxseq, self.n_users, self.n_attr, self.dntype_option),
                    val_samples=y_test.shape[0])

        else:
            if self.dntype_option == 3:
                score, acc = self.model.evaluate([x_test, attr_test], y_test, batch_size=batch_size)
            elif self.dntype_option == 4:
                score, acc = self.model.evaluate([x_test, attr_test, attr_test], y_test, batch_size=batch_size)
            else:
                score, acc = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        return score, acc

    def predict(self, x_test, attr_test, model_file, batch_size):
        # Load the best weights
        if os.path.isfile(model_file):
            self.model.load_weights(model_file)

        if self.b_parallel:
            if self.dntype_option == 2:
                if self.neighbSum:
                    l_pred = self.argmax_ndarray(self.model.predict_generator(self.generate_arrays_test_withneighbors_for_prediction(
                        x_test, self.maxseq, self.n_users, self.n_attr, self.n_classes, self.d_id_pred), val_samples = x_test.shape[0]))
            else:
                l_pred = self.argmax_ndarray(self.model.predict_generator(self.generate_arrays_test_for_prediction(
                    x_test, attr_test, self.maxseq, self.n_users, self.n_attr, self.dntype_option), val_samples = x_test.shape[0]))

        else:
            if self.dntype_option == 3:
                l_pred = self.model.predict_classes([x_test, attr_test], batch_size=batch_size)
            elif self.dntype_option == 4:
                l_pred = self.model.predict_classes([x_test, attr_test, attr_test], batch_size=batch_size)
            else:
                l_pred = self.model.predict_classes(x_test, batch_size=batch_size)

        return l_pred

    def predict_prob (self, x_test, attr_test, model_file, batch_size):

        # Load the best weights
        if os.path.isfile(model_file):
            self.model.load_weights(model_file)

        if self.b_parallel:
            if self.dntype_option == 2:
                if self.neighbSum:
                    l_pred = self.model.predict_generator(self.generate_arrays_test_withneighbors_for_prediction(
                            x_test, self.maxseq, self.n_users, self.n_attr, self.n_classes, self.d_id_pred),
                            val_samples=x_test.shape[0])
            else:
                l_pred = self.model.predict_generator(self.generate_arrays_test_for_prediction(
                    x_test, attr_test, self.maxseq, self.n_users, self.n_attr, self.dntype_option), val_samples=x_test.shape[0])

        else:
            if self.dntype_option == 3:
                l_pred = self.model.predict_proba([x_test, attr_test], batch_size=batch_size)
            elif self.dntype_option == 4:
                l_pred = self.model.predict_proba([x_test, attr_test, attr_test], batch_size=batch_size)
            else:
                l_pred = self.model.predict_proba(x_test, batch_size=batch_size)

        return l_pred

    def build_model(self, dntype_option = 0, hidden_units = 32, timedist_units = 8, maxpooling = 5, conv_nums = 16, conv_size = 4, b_pooling = True, b_attention_sum = False, b_attention_withattr = False):

        self.dntype_option = dntype_option
        # rnn model with attr
        if dntype_option == 0:

            if b_pooling:

                self.model.add(
                    LSTM(hidden_units, input_shape=(self.maxseq, self.n_users + self.n_attr), return_sequences=True))
                self.model.add(LSTM(hidden_units / 2, return_sequences=True))
                self.model.add(TimeDistributed(Dense(timedist_units)))

                if b_pooling:
                    self.model.add(MaxPooling1D(pool_length=(maxpooling)))

                self.model.add(Flatten())
                self.model.add(Dropout(0.2))

                self.model.add(Dense(32, activation='relu'))
                self.model.add(Dropout(0.2))
                self.model.add(Dense(self.n_classes, activation='softmax'))

            else:
                self.model.add(Masking(mask_value=-1.0, input_shape=(self.maxseq, self.n_users + self.n_attr)))
                self.model.add(
                    LSTM(hidden_units))

                self.model.add(Dropout(0.2))
                self.model.add(Dense(self.n_classes, activation='softmax'))



        # cnn model
        elif dntype_option == 1:
            print conv_nums, conv_size, self.maxseq, self.n_users, self.n_attr
            self.model.add(Convolution1D(nb_filter=conv_nums,
                                    filter_length=conv_size,
                                    border_mode='valid',
                                    activation='relu',
                                    subsample_length=1,
                                    input_shape=(self.maxseq, self.n_users + self.n_attr)))
            # we use max pooling:
            if b_pooling:
                self.model.add(MaxPooling1D(pool_length=maxpooling))


            # We flatten the output of the conv layer,
            # so that we can add a vanilla dense layer:
            self.model.add(Flatten())
            # We add a vanilla hidden layer:
            # self.model.add(Dense(32))
            self.model.add(Dropout(0.2))
            # self.model.add(Activation('relu'))
            # self.model.add(Dropout(0.2))
            self.model.add(Dense(self.n_classes, activation='softmax'))

        # rnn model with attr and class
        elif dntype_option == 2:
            # self.model.add(LSTM(hidden_units, input_shape=(maxseq, n_users + n_attr), return_sequences=True))
            self.model.add(LSTM(hidden_units, input_shape=(self.maxseq, self.n_users + self.n_attr + self.n_classes), return_sequences=True))
            self.model.add(LSTM(hidden_units / 2, return_sequences=True))
            self.model.add(TimeDistributed(Dense(timedist_units)))

            if b_pooling:
                self.model.add(MaxPooling1D(pool_length=(maxpooling)))
            self.model.add(Flatten())
            self.model.add(Dropout(0.2))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.n_classes, activation='softmax'))

        # attention model with lstm
        elif dntype_option == 3:

            model1 = Sequential()
            model1.add(LSTM(hidden_units, input_shape=(self.maxseq, self.n_users), dropout_U=0.2, return_sequences=True))
            model1.add(LSTM(hidden_units / 2, dropout_U=0.2, return_sequences=True))
            model1.add(TimeDistributed(Dense(timedist_units)))
            model1.add(Dropout(0.2))
            # The weight model  - actual output shape  = (batch, step)
            # after reshape : output_shape = (batch, step,  hidden)
            model2 = Sequential()
            model2.add(Dense(input_dim=self.n_attr, output_dim=self.maxseq))
            model2.add(Activation('softmax'))  # Learn a probability distribution over each  step.
            # Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
            model2.add(RepeatVector(timedist_units))
            model2.add(Permute((2, 1)))

            # The final model which gives the weighted sum:
            # model = Sequential()

            time_distributed_merge_layer = Lambda(function=lambda x: K.mean(x, axis=1),
                                                  output_shape=lambda shape: (shape[0],) + shape[2:])

            self.model.add(
                Merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
            if b_attention_sum:
                self.model.add(time_distributed_merge_layer)  # Sum the weighted elements.
            else:
                if b_pooling:
                    self.model.add(MaxPooling1D(pool_length=(maxpooling)))
                # self.model.add(Highway)
                self.model.add(Flatten())
            self.model.add(Dropout(0.2))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.n_classes, activation='softmax'))

        elif dntype_option == 4:

            model1 = Sequential()
            model1.add(
                LSTM(hidden_units, input_shape=(self.maxseq, self.n_users), dropout_U=0.2, return_sequences=True))
            model1.add(LSTM(hidden_units / 2, dropout_U=0.2, return_sequences=True))
            model1.add(TimeDistributed(Dense(timedist_units)))
            model1.add(Dropout(0.2))
            # The weight model  - actual output shape  = (batch, step)
            # after reshape : output_shape = (batch, step,  hidden)
            model2 = Sequential()
            model2.add(Dense(input_dim=self.n_attr, output_dim=self.maxseq))
            model2.add(Activation('softmax'))  # Learn a probability distribution over each  step.
            # Reshape to match LSTM's output shape, so that we can do element-wise multiplication.

            model2.add(RepeatVector(timedist_units))
            model2.add(Permute((2, 1)))

            # The final model which gives the weighted sum:
            # model = Sequential()
            time_distributed_merge_layer = Lambda(function=lambda x: K.mean(x, axis=1),
                                                  output_shape=lambda shape: (shape[0],) + shape[2:])

            model3 = Sequential()
            model3.add(
                Merge([model1, model2], 'mul'))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
            # self.model.add(time_distributed_merge_layer)  # Sum the weighted elements.
            if b_pooling:
                model3.add(MaxPooling1D(pool_length=(maxpooling)))
            # self.model.add(Highway)
            model3.add(Flatten())

            model4 = Sequential()
            model4.add(Dense(input_dim=self.n_attr, output_dim=self.n_attr / 2))
            model4.add(Activation('relu'))  # Learn a probability distribution over each  step.
            # Reshape to match LSTM's output shape, so that we can do element-wise multiplication.

            self.model.add(
                Merge([model3, model4], 'concat'))

            # self.model.add(Dropout(0.2))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.n_classes, activation='softmax'))

        # dnn model
        elif dntype_option == 5:

            self.model.add(Dense(1024, input_dim=(self.n_users + self.n_attr), init='uniform', activation='relu'))

            self.model.add(Dense(512))
            self.model.add(Dropout(0.2))
            self.model.add(Activation('relu'))

            self.model.add(Dense(64))
            self.model.add(Dropout(0.2))
            self.model.add(Activation('relu'))

            self.model.add(Dropout(0.2))
            self.model.add(Dense(self.n_classes, activation='softmax'))

        # model compile
        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        #
        print self.model.summary()

    # train neural network model
    def train_model (self, model_file, x_train, y_train, attr_train, id_train, x_valid, y_valid, attr_valid, x_unlabeled, y_unlabeled, attr_unlabeled, id_unlabeled, y_all, id_all, callbacks, batch_size = 5, epochs = 20):

        if self.dntype_option == 2:

            # init class dist of neighbors

            if self.perfectNeighbors:
                for i in range(len(id_all)):
                    self.d_id_pred[id_all[i]] = y_all[i]
            else:
                if self.rInit:
                    self.d_id_pred = self.assign_pred_dic(np.random.randint(self.n_classes, size=len(id_all)), id_all, self.n_classes)
                    for i in range(len(id_train)):
                        self.d_id_pred[id_train[i]] = y_train[i]

        best_acc = 0.
        best_score = 1000.0
        i_patience = 0

        if self.b_parallel:
            if self.dntype_option == 2:
                for e in range(epochs):
                    print "-----------", e, " th /", epochs, " iteration-------------"
                    # x_train, attr_train, y_train, id_train = self.rand_data(x_train, attr_train, y_train, id_train, e)
                    if self.neighbSum:
                        train_generator = self.generate_arrays_train_withneighbors(x_train, y_train, batch_size, self.maxseq, self.n_users,
                                                                                   self.n_attr, self.n_classes, self.d_id_pred)
                    for sample in range(len(x_train) / batch_size):
                        if sample % 100 == 0:
                            print "--", sample, " / ", str(len(x_train) / batch_size)
                        (x, y) = next(train_generator)
                        self.model.train_on_batch(x, y)

                    # assign lables for unlabed nodes
                    if not self.perfectNeighbors:
                        if self.neighbSum:
                            l_pred = self.argmax_ndarray(self.model.predict_generator(self.generate_arrays_test_withneighbors(
                                x_unlabeled, y_unlabeled, self.maxseq, self.n_users, self.n_attr, self.n_classes, self.d_id_pred),
                                val_samples=y_unlabeled.shape[0]))
                            self.d_id_pred = self.assign_pred_dic(l_pred, id_unlabeled, self.n_classes)

                    if self.neighbSum:
                        score, acc = self.model.evaluate_generator(self.generate_arrays_test_withneighbors(
                            x_valid, y_valid, self.maxseq, self.n_users, self.n_attr, self.n_classes, self.d_id_pred),
                            val_samples=y_valid.shape[0])
                    print "\tval loss: ", score, ", val accuracy: ", acc
                    if acc >= best_acc and score < best_score:
                        best_acc = acc
                        best_score = score
                        i_patience = 0
                        self.model.save_weights(model_file)

                    else:
                        i_patience += 1
                        if i_patience == 15:
                            print "# early stopping ... "
                            break

            elif self.dntype_option == 3 or self.dntype_option == 4:
                for e in range(epochs):
                    print "-----------", e, " th /", epochs, " iteration-------------"
                    x_train, attr_train, y_train, id_train = self.rand_data(x_train, attr_train, y_train, id_train, e)
                    train_generator = self.generate_arrays_train(x_train, attr_train, y_train, batch_size, self.maxseq,
                                                                 self.n_users, self.n_attr, self.dntype_option)

                    size = y_train.shape[0]

                    for sample in range(int(np.ceil(size / float(batch_size)))):
                        if sample % 100 == 0:
                            print "--", sample, " / ", str(int(np.ceil(size / float(batch_size))))
                        if self.dntype_option == 3:
                            ([x, attrx], y) = next(train_generator)
                            self.model.train_on_batch([x, attrx], y)
                        elif self.dntype_option == 4:
                            ([x, attrx, attrx2], y) = next(train_generator)
                            self.model.train_on_batch([x, attrx, attrx2], y)
                        else:
                            (x, y) = next(train_generator)
                            self.model.train_on_batch(x, y)

                    score, acc = self.model.evaluate_generator(self.generate_arrays_eval(
                        x_valid, attr_valid, y_valid, self.maxseq, self.n_users, self.n_attr, self.dntype_option),
                        val_samples=y_valid.shape[0])
                    print y_valid.shape
                    print "\tval loss: ", score, ", val accuracy: ", acc

                    if acc > best_acc:
                        best_acc = acc
                        i_patience = 0
                        self.model.save_weights(model_file)
                    else:
                        i_patience += 1
                        if i_patience == self.patience:
                            print "# early stopping ... "
                            break

            elif self.dntype_option == 5:

                for e in range(epochs):
                    print "-----------", e, " th /", epochs, " iteration-------------"
                    x_train, attr_train, y_train, id_train = self.rand_data(x_train, attr_train, y_train, id_train, e)
                    train_generator = self.generate_arrays_train(x_train, attr_train, y_train, batch_size, self.maxseq, self.n_users, self.n_attr, self.dntype_option)
                    for sample in range(len(x_train) / batch_size):
                        if sample % 100 == 0:
                            print "--", sample, " / ", str(len(x_train) / batch_size)
                        (x, y) = next(train_generator)
                        self.model.train_on_batch(x, y)

                    score, acc = self.model.evaluate_generator(self.generate_arrays_eval(
                        x_valid, attr_valid, y_valid, self.maxseq, self.n_users, self.n_attr, self.dntype_option),
                        val_samples=y_valid.shape[0])
                    print "\tval loss: ", score, ", val accuracy: ", acc

                    if acc > best_acc:
                        best_acc = acc
                        i_patience = 0
                        self.model.save_weights(model_file)
                    else:
                        i_patience += 1
                        if i_patience == self.patience:
                            print "# early stopping ... "
                            break


            else:
                for e in range(epochs):
                    print "-----------", e, " th /", epochs, " iteration-------------"
                    x_train, attr_train, y_train, id_train = self.rand_data(x_train, attr_train, y_train, id_train, e)
                    train_generator = self.generate_arrays_train(x_train, attr_train, y_train, batch_size, self.maxseq, self.n_users, self.n_attr, self.dntype_option)
                    for sample in range(len(x_train) / batch_size):
                        if sample % 100 == 0:
                            print "--", sample, " / ", str(len(x_train) / batch_size)
                        (x, y) = next(train_generator)
                        self.model.train_on_batch(x, y)

                    score, acc = self.model.evaluate_generator(self.generate_arrays_eval(
                        x_valid, attr_valid, y_valid, self.maxseq, self.n_users, self.n_attr, self.dntype_option),
                        val_samples=y_valid.shape[0])
                    print "\tval loss: ", score, ", val accuracy: ", acc

                    if acc > best_acc:
                        best_acc = acc
                        i_patience = 0
                        self.model.save_weights(model_file)
                    else:
                        i_patience += 1
                        if i_patience == self.patience:
                            print "# early stopping ... "
                            break

        else:
            if self.dntype_option == 3:
                self.model.fit([x_train, attr_train], y_train, batch_size=batch_size, nb_epoch=epochs,
                               callbacks=callbacks,
                               validation_data=([x_valid, attr_valid], y_valid))  # starts training
            elif self.dntype_option == 4:
                self.model.fit([x_train, attr_train, attr_train], y_train, batch_size=batch_size, nb_epoch=epochs,
                               callbacks=callbacks,
                               validation_data=([x_valid, attr_valid, attr_valid], y_valid))  # starts training
            elif self.dntype_option == 2:
                print "# please use parallel option, Thanks"
                exit()
            else:
                self.model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, callbacks=callbacks,
                      validation_data=(x_valid, y_valid))  # starts training d


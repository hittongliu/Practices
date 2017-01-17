#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
from dataprocess import *
from gensim.models.word2vec import Word2Vec
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import jieba
from sklearn.externals import joblib
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
import h5py

def CNN_LSTM(lstm_true):
    X_train = np.load('./data/rnn_small_xtrain_vec.npy')
    X_test = np.load('./data/rnn_small_xtest_vec.npy')
    y_train = np.load('./data/rnn_y_small_train.npy')
    y_test = np.load('./data/rnn_y_small_test.npy')

    pading_size = 100
    # set parameters:
    batch_size = 32
    n_filter = 16
    filter_length = 4
    nb_epoch = 5
    n_pool = 2

    # neg = [word for word in y_train if word == 1]
    # pos = [word for word in y_train if word == 0]
    # print len(neg)
    # print len(pos)
    # print y_train.shape
    # 新建一个sequential的模型
    model = Sequential()
    model.add(Convolution2D(16,4,4,border_mode='same',input_shape=(1,pading_size,128)))
    model.add(Activation('relu'))
    model.add(Convolution2D(n_filter,filter_length,filter_length,border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(n_pool, n_pool),border_mode='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    print model.summary()
    # 后面接上一个ANN
    if lstm_true == 1:
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('softmax'))
    elif lstm_true == 0:
        model.add(Reshape((50,16)))
        model.add(LSTM(128, dropout_W = 0.2, dropout_U = 0.2))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation = 'sigmoid'))
        model.add(Dense(1))
        model.add(Activation('softmax'))
    # compile模型
    model.compile(loss='mse',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch,
              verbose = 1)
    score = model.evaluate(X_test, y_test, verbose = 1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    if lstm_true == 1:
        model.save_weights('my_model_weights.h5')
    else:
        model.save_weights('my_model_lstm_weights.h5')
CNN_LSTM(0)
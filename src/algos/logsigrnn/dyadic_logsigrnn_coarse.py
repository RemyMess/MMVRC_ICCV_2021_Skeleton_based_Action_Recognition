#!/usr/bin/env python
# coding: utf-8


import os
import pickle

from esig import tosig
import iisignature
from joblib import Parallel, delayed
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    Lambda,
    Reshape,
    concatenate,
)
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.utils import to_categorical

from src.algos.logsigrnn.sigutils import *
from src.algos.logsigrnn.dyadic_sigutils import *


# constants

N_JOINTS = 17
N_AXES = 2
N_PERSONS = 2
N_TIMESTEPS = 305


PATH_DATA = r"_input/train_process_data.npy"
PATH_LABELS = r"_input/train_process_label.pkl"
PATH_LABELS_DF = r"_input/train_process_label.csv"
PATH_LEARNING_CURVE = r"_output/learning.csv"
PATH_MODEL = r"_output/logsigrnn.hdf5"


# hyperparameters

# number of classes [0-155]; pick number smaller than 155 to learn less actions
PERMITTED = np.arange(155)
SIGNATURE_DEGREE = 2
N_SEGMENTS = 32
BATCH_SIZE = 256
FILTER_SIZE_1 = 5
FILTER_SIZE_2 = 40
N_EPOCHS = 50
N_HIDDEN_NEURONS = 64
DROP_RATE_2 = 0.8
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15


# split

def load_data(data):

    labels = pd.read_csv(PATH_LABELS_DF)
    idx = labels.loc[labels['label'].astype(int).isin(PERMITTED)].index

    X = data[idx].transpose((0, 2, 3, 1, 4))
    X = X.reshape(X.shape[:3] + (N_AXES * N_PERSONS,))
    y = labels.loc[idx, 'label']

    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(y))

    return X, y


# log-sigrnn

def build_model(n_segments_range=(4, 8, 16, 32, 64), drop_rate_2=DROP_RATE_2, filter_size_2=FILTER_SIZE_2, n_hidden_neurons=N_HIDDEN_NEURONS):
    
    input_shape = (N_TIMESTEPS, N_JOINTS, N_AXES * N_PERSONS)
    logsiglen = iisignature.logsiglength(filter_size_2, SIGNATURE_DEGREE)

    input_layer = Input(input_shape)

    lin_projection_layer = Conv2D(32, (1, 1), strides=(1, 1), data_format='channels_last')(input_layer)
    lin_projection_layer = Conv2D(16, (FILTER_SIZE_1, 1), strides=(1, 1), data_format='channels_last')(lin_projection_layer)

    reshape = Reshape((input_shape[0] - FILTER_SIZE_1 + 1, 16 * N_JOINTS))(lin_projection_layer)
    lin_projection_layer = Conv1D(filter_size_2, 1)(reshape)
    
    mid_output = Lambda(SP, arguments=dict(no_of_segments=max(n_segments_range)), output_shape=(max(n_segments_range), filter_size_2))(lin_projection_layer)

    hidden_layer = Lambda(dyadic_CLF,
                          arguments=dict(n_segments_range=n_segments_range, deg_of_logsig=SIGNATURE_DEGREE, logsiglen=logsiglen),
                          output_shape=(max(n_segments_range), logsiglen))(lin_projection_layer)

    hidden_layer = Reshape((max(n_segments_range), len(n_segments_range) * logsiglen))(hidden_layer)

    BN_layer = BatchNormalization()(hidden_layer)

    # samples from the signal + log signatures
    mid_input = concatenate([mid_output, BN_layer])

    # LSTM
    lstm_layer = LSTM(units=n_hidden_neurons, return_sequences=True)(mid_input)
    drop_layer = Dropout(drop_rate_2)(lstm_layer)
    output_layer = Flatten()(drop_layer)
    output_layer = Dense(len(PERMITTED), activation='softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    adam = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    
    y_train_categorical = to_categorical(y_train)

    early_stopping_monitor = EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, verbose=1, factor=0.8, min_lr=0.000001)
    mcp_save = ModelCheckpoint(PATH_MODEL, save_best_only=True, monitor='acc', mode='auto')

    callbacks = [early_stopping_monitor, reduce_lr, mcp_save]
    history = model.fit(X_train, y_train_categorical, epochs=n_epochs, batch_size=batch_size, validation_split=VALIDATION_SPLIT, callbacks=callbacks)
    
    return history


if __name__ == '__main__': 
    
    print("Loading data (takes less than a minute)")

    # load numpy array
    data = np.load(PATH_DATA).astype(np.float64)

    # train test split
    X, y = load_data(data) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)

    # build and train model
    model = build_model()
    history = train_model(model, X_train, y_train)
    
    df = pd.DataFrame(history.history)
    df.to_csv(PATH_LEARNING_CURVE)
    
    # model should be saved at checkpoints, in case it's not make sure last version is saved
    if not os.path.exists(PATH_MODEL):
        model.save(PATH_MODEL)


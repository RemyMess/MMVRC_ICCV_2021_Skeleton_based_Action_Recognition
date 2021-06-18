#!/usr/bin/env python
# coding: utf-8


from esig import tosig
import iisignature
import numpy as np
from keras import Model
from keras.layers import concatenate, Dense, Dropout, LSTM, Input, InputLayer, Embedding, Flatten, Conv1D, Conv2D, MaxPooling2D, Reshape, Lambda, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.pipeline import Pipeline
from keras.regularizers import l1, l2, l1_l2
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.sigutils import *
import os
import pickle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


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
BATCH_SIZE = 256
FILTER_SIZE_1 = 5
FILTER_SIZE_2 = 40
N_EPOCHS = 20
N_HIDDEN_NEURONS = 64
DROP_RATE_2 = 0.8
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2



# split

def load_data():

    labels = pd.read_csv(PATH_LABELS_DF)
    idx = labels.loc[labels['label'].astype(int).isin(PERMITTED)].index

    # one-hot encoding
    encoder = OneHotEncoder()
    y = encoder.fit_transform(labels.loc[idx, 'label'].to_numpy().reshape((-1, 1))).toarray()
    
    # train test split
    X = data[idx].transpose((0, 2, 3, 1, 4))
    X = X.reshape(X.shape[:3] + (N_AXES * N_PERSONS,))

    assert not np.any(np.isnan(X))
    assert not np.any(np.isnan(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42)

    return X_train, X_test, y_train, y_test


# log-sigrnn

def build_model(drop_rate_2=DROP_RATE_2, filter_size_2=FILTER_SIZE_2, n_hidden_neurons=N_HIDDEN_NEURONS):
    
    input_shape = (N_TIMESTEPS, N_JOINTS, N_AXES * N_PERSONS)
    logsiglen = iisignature.logsiglength(filter_size_2, SIGNATURE_DEGREE)

    input_layer = Input(input_shape)

    lin_projection_layer = Conv2D(32, (1, 1), strides=(1, 1), data_format='channels_last')(input_layer)
    lin_projection_layer = Conv2D(16, (FILTER_SIZE_1, 1), strides=(1, 1), data_format='channels_last')(lin_projection_layer)

    reshape = Reshape((input_shape[0] - FILTER_SIZE_1 + 1, 16 * N_JOINTS))(lin_projection_layer)
    lin_projection_layer = Conv1D(filter_size_2, 1)(reshape)
    
    layers = []

    for i, n_segments in enumerate((4, 8, 16, 32, 64)):

        mid_output = Lambda(SP, arguments=dict(no_of_segments=n_segments), output_shape=(n_segments, filter_size_2), name=f"start_layer_{i}")(lin_projection_layer)

        hidden_layer = Lambda(CLF,
                              arguments=dict(number_of_segment=n_segments, deg_of_logsig=SIGNATURE_DEGREE, logsiglen=logsiglen),
                              output_shape=(n_segments, logsiglen))(lin_projection_layer)

        hidden_layer = Reshape((n_segments, logsiglen))(hidden_layer)

        BN_layer = BatchNormalization()(hidden_layer)

        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer])

         # LSTM
        lstm_layer = LSTM(units=n_hidden_neurons, return_sequences=True)(mid_input)

        drop_layer = Dropout(drop_rate_2)(lstm_layer)
        inter_layer = Flatten()(drop_layer)
        layers += [inter_layer]

    upper_mid_input = concatenate(layers)

    # Dropout
    output_layer = Dense(len(PERMITTED), activation='softmax')(upper_mid_input)

    model = Model(inputs=input_layer, outputs=output_layer)

    adam = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])

    return model


def train(X_train, X_test, y_train, y_test):
    
    early_stopping_monitor = EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, verbose=1, factor=0.8, min_lr=0.000001)
    mcp_save = ModelCheckpoint(PATH_MODEL, save_best_only=True, monitor='acc', mode='auto')

    model = build_model()
    callbacks = [early_stopping_monitor, reduce_lr, mcp_save]
    history = model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks=callbacks)
    
    return model, history


if __name__ == '__main__':    
    
    print("Loading data (takes less than a minute)")

    # load numpy array
    data = np.load(PATH_DATA).astype(np.float64)

    # train test split
    X_train, X_test, y_train, y_test = load_data()

    model, history = train(X_train, X_test, y_train, y_test)
    
    df = pd.DataFrame(history.history)
    df.to_csv(PATH_LEARNING_CURVE)
    
    # model should be saved at checkpoints, in case it's not make sure last version is saved
    if not os.path.exists(PATH_MODEL):
        model.save(PATH_MODEL)
   

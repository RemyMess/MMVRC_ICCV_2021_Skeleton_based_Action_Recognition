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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from src.algos.logsigrnn.utils import *
from src.algos.logsigrnn.sigutils import *
from src.algos.logsigrnn.dyadic_sigutils import *
from src.algos.linconv import LinConv


# constants

N_JOINTS = 17
N_AXES = 3
N_PERSONS = 2
N_TIMESTEPS = 305


PATH_DATA = r"_input/train_new_data.npy"
PATH_LABELS = r"_input/train_new_label.pkl"
PATH_LABELS_DF = r"_input/train_new_label.csv"
PATH_LEARNING_CURVE = r"_output/learning.csv"
PATH_MODEL = r"_output/logsigrnn.hdf5"


# hyperparameters

# number of classes [0-155]; pick number smaller than 155 to learn less actions
PERMITTED = np.arange(155)
# PERMITTED = 57, 106, 92, 119, 101, 120, 112, 11, 7, 107
# PERMITTED = 87, 41, 75, 80, 76, 61, 38, 151, 121, 5
SIGNATURE_DEGREE = 2
N_SEGMENTS = 32,
BATCH_SIZE = 256
FILTER_SIZE_1 = 5
FILTER_SIZE_2 = 40
N_EPOCHS = 50
N_HIDDEN_NEURONS = 64
DROP_RATE_2 = 0.8
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15


# dyadic log-sigrnn

def build_model(input_shape, mode,
                n_segments=N_SEGMENTS,
                drop_rate_2=DROP_RATE_2,
                filter_size_2=FILTER_SIZE_2,
                n_hidden_neurons=N_HIDDEN_NEURONS):
   
    logsiglen = iisignature.logsiglength(filter_size_2, SIGNATURE_DEGREE)

    input_layer = Input(input_shape)

    lin_projection_layer = Conv2D(32, (1, 1), strides=(1, 1), data_format='channels_last')(input_layer)
    lin_projection_layer = Conv2D(16, (FILTER_SIZE_1, 1), strides=(1, 1), data_format='channels_last')(lin_projection_layer)

    reshape = Reshape((input_shape[0] - FILTER_SIZE_1 + 1, 16 * N_JOINTS))(lin_projection_layer)
    lin_projection_layer = Conv1D(filter_size_2, 1)(reshape)
    
    if mode == 0:
        
        assert len(n_segments) == 1, "Non-dyadic log-signature, n_segments should be of size 1"

        mid_output = Lambda(SP, arguments=dict(no_of_segments=n_segments[0]), output_shape=(n_segments[0], filter_size_2))(lin_projection_layer)

        hidden_layer = Lambda(CLF,
                              arguments=dict(number_of_segment=n_segments[0], deg_of_logsig=SIGNATURE_DEGREE, logsiglen=logsiglen),
                              output_shape=(n_segments[0], logsiglen))(lin_projection_layer)

        hidden_layer = Reshape((n_segments[0], logsiglen))(hidden_layer)

        BN_layer = BatchNormalization()(hidden_layer)

        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer])

        # LSTM
        lstm_layer = LSTM(units=n_hidden_neurons, return_sequences=True)(mid_input)
        drop_layer = Dropout(drop_rate_2)(lstm_layer)
        upper_mid_input = Flatten()(drop_layer)

    elif mode == 1:

        assert len(n_segments) > 1, "Dyadic log-signature, n_segments should be of size > 1"

        mid_output = Lambda(SP, arguments=dict(no_of_segments=max(n_segments)), output_shape=(max(n_segments), filter_size_2))(lin_projection_layer)

        hidden_layer = Lambda(dyadic_CLF,
                              arguments=dict(n_segments_range=n_segments, deg_of_logsig=SIGNATURE_DEGREE, logsiglen=logsiglen),
                              output_shape=(max(n_segments), logsiglen))(lin_projection_layer)

        hidden_layer = Reshape((max(n_segments), len(n_segments) * logsiglen))(hidden_layer)
        
        BN_layer = BatchNormalization()(hidden_layer)

        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer])

        # LSTM
        lstm_layer = LSTM(units=n_hidden_neurons, return_sequences=True)(mid_input)
        drop_layer = Dropout(drop_rate_2)(lstm_layer)
        upper_mid_input = Flatten()(drop_layer)

    elif mode == 2:

        assert len(n_segments) > 1, "Dyadic log-signature, n_segments should be of size > 1"

        layers = []

        for i, n_segment in enumerate(n_segments):

            mid_output = Lambda(SP, arguments=dict(no_of_segments=n_segment), output_shape=(n_segment, filter_size_2), name=f"start_layer_{i}")(lin_projection_layer)

            hidden_layer = Lambda(CLF,
                                  arguments=dict(number_of_segment=n_segment, deg_of_logsig=SIGNATURE_DEGREE, logsiglen=logsiglen),
                                  output_shape=(n_segment, logsiglen))(lin_projection_layer)

            hidden_layer = Reshape((n_segment, logsiglen))(hidden_layer)

            BN_layer = BatchNormalization()(hidden_layer)

            # samples from the signal + log signatures
            mid_input = concatenate([mid_output, BN_layer])

             # LSTM
            lstm_layer = LSTM(units=n_hidden_neurons, return_sequences=True)(mid_input)
            drop_layer = Dropout(drop_rate_2)(lstm_layer)
            inter_layer = Flatten()(drop_layer)
            layers += [inter_layer]

        upper_mid_input = concatenate(layers)

    else:

        raise ValueError("Mode not supported.")

    output_layer = Dense(len(PERMITTED), activation='softmax')(upper_mid_input)
    model = Model(inputs=input_layer, outputs=output_layer)

    adam = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])

    return model


def train(model, train_index, test_index, **kwargs):
   
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    early_stopping_monitor = EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, verbose=1, factor=0.8, min_lr=0.000001)
    mcp_save = ModelCheckpoint(PATH_MODEL, save_best_only=True, monitor='val_accuracy', mode='auto', save_weights_only=True)
    
    callbacks = [early_stopping_monitor, reduce_lr, mcp_save]
    history = model.fit(X_train, y_train, callbacks=callbacks, **kwargs)

    return history


if __name__ == '__main__': 
    
    print("Loading data (takes less than a minute)")

    data = np.load(PATH_DATA).astype(np.float64)
    labels = pd.read_csv(PATH_LABELS_DF)
    idx = labels.loc[labels['label'].astype(int).isin(PERMITTED)].index
    labels = labels.loc[idx]

    X = data[idx].copy()
    # X = interpolate_frames(X, labels)
    # X = append_confidence_score(X, labels)

    X = processed_data.transpose((0, 2, 3, 1, 4))
    X = X.reshape(X.shape[:3] + (-1,)).astype(np.float32)

    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(labels.loc[idx, 'label'].to_numpy().reshape(-1, 1))

    # train test split (we work indices to ensure pickling for parallelization
    train_index, test_index = train_test_split(np.arange(len(X)), test_size=TEST_SPLIT, random_state=42) 

    
    # build and train model

    model = build_model(input_shape=X.shape[1:], mode=0, n_segments=(32,))
    history = train(model, train_index, test_index, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)
    
    # record learning curve
    df = pd.DataFrame(history.history)
    df.to_csv(PATH_LEARNING_CURVE)
    
    # model should be saved at checkpoints, in case it's not make sure last version is saved
    if not os.path.exists(PATH_MODEL):
        model.save(PATH_MODEL)


    # cross-validate

    # skf = StratifiedKFold(n_splits=5)
    # out = Parallel(n_jobs=5, verbose=100)(delayed(train)(train_index, test_index, epochs=N_EPOCHS, batch_size=BATCH_SIZE) for train_index, test_index in list(skf.split(X, y)))


    # grid search

    # model = KerasClassifier(build_fn=build_model, verbose=1)
    # grid_result = grid_search(model, X_train, y_train)

    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # params = grid_result.cv_results_['params']
    # mean_test_score = grid_result.cv_results_['mean_test_score']
    # std_test_score = grid_result.cv_results_['std_test_score']

    # for params, mean, std in zip(params, mean_test_score, std_test_score):
    #     print("%f (%f) with: %r" % (mean, std, params))

    

    # random search

    # model = KerasClassifier(build_fn=build_model, verbose=1)
    # random_search = randomized_search(model, X_train, y_train)
    
    # print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
    # means = random_search.cv_results_['mean_test_score']
    # stds = random_search.cv_results_['std_test_score']
    # params = random_search.cv_results_['params']
    
    # for mean, stdev, param in sorted(zip(means, stds, params), key=lambda x: x[0])[::-1]:
    #     print("%f (%f) with: %r" % (mean, stdev, param))



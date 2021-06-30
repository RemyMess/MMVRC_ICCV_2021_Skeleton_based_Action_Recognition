 #!/usr/bin/env python
# coding: utf-8


import os
import pickle

from joblib import Parallel, delayed
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tqdm import tqdm

from src.algos.logsigrnn.utils import *
from src.algos.logsigrnn.sigutils import *
from src.algos.logsigrnn.dyadic_sigutils import *
from src.algos.linconv import LinConv
from src.algos.logsigrnn.vae import *
from src.algos.logsigrnn.logsigrnn import *


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
PERMITTED = np.arange(10)
TUPLE_SIZE = 2
SIGNATURE_DEGREE = 2
N_SEGMENTS = 32,
BATCH_SIZE = 256
FILTER_SIZE_1 = 5
N_PROJECTION_NEURONS = 40
N_EPOCHS = 50
N_HIDDEN_NEURONS = 64
DROP_RATE_2 = 0.8
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15


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


    # %% load data

    print("Loading data (takes less than a minute)")

    data = np.load(PATH_DATA).astype(np.float64)
    labels = pd.read_csv(PATH_LABELS_DF)
    idx = labels.loc[labels['label'].astype(int).isin(PERMITTED)].index
    labels = labels.loc[idx]

    one_hot_encoder = OneHotEncoder(sparse=False)
    y = one_hot_encoder.fit_transform(labels.loc[idx, 'label'].to_numpy().reshape(-1, 1))



    # %% create X, y for vanilla model

    X = data[idx].copy()
    # X = interpolate_frames(X, labels)
    # X = append_confidence_score(X, labels)

    X = X.transpose((0, 2, 3, 1, 4))
    X = X.reshape(X.shape[:3] + (-1,)).astype(np.float32)

    one_hot_encoder = OneHotEncoder(sparse=False)
    y = one_hot_encoder.fit_transform(labels.loc[idx, 'label'].to_numpy().reshape(-1, 1))

    # train test split (we work indices to ensure pickling for parallelization
    train_index, test_index = train_test_split(np.arange(len(X)), test_size=TEST_SPLIT, random_state=42) 

    
    # %% build and train model

    model = build_logsigrnn_model(input_shape=X.shape[1:],
                                  n_segments=(32,),
                                  drop_rate_2=DROP_RATE_2,
                                  signature_degree=SIGNATURE_DEGREE,
                                  filter_size_1=FILTER_SIZE_1,
                                  n_projection_neurons=N_PROJECTION_NEURONS,
                                  n_joints=N_JOINTS,
                                  n_classes=len(PERMITTED),
                                  learning_rate=LEARNING_RATE,
                                  n_hidden_neurons=N_HIDDEN_NEURONS)

    history = train(model, train_index, test_index, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)


    # %% create X, y for pairs model

    X_sigs = sigs.reshape((sigs.shape[0] * sigs.shape[1], -1))
    X = np.zeros((X_sigs.shape[0], 102))

    for i in tqdm(range(X.shape[0]//10000+1)):
        s = slice(i*10000, (i+1)*10000)
        X[s] = encoder(X_sigs[s])

    X = X.reshape((len(idx), data.shape[2], 102))

    # train test split (we work indices to ensure pickling for parallelization
    train_index, test_index = train_test_split(np.arange(len(X)), test_size=TEST_SPLIT, random_state=42) 

    
    # %% build and train model

    model = build_logsigrnn_model(input_shape=X.shape[1:],
                                  n_segments=(32,),
                                  drop_rate_2=DROP_RATE_2,
                                  signature_degree=SIGNATURE_DEGREE,
                                  filter_size_1=FILTER_SIZE_1,
                                  n_projection_neurons=102,
                                  n_joints=N_JOINTS,
                                  n_classes=len(PERMITTED),
                                  learning_rate=LEARNING_RATE,
                                  n_hidden_neurons=N_HIDDEN_NEURONS)

    history = train(model, train_index, test_index, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)


    # %% create X, y for vanilla model

    X = data[idx].copy()
    # X = interpolate_frames(X, labels)
    # X = append_confidence_score(X, labels)

    X = X.transpose((0, 2, 3, 1, 4))
    X = X.reshape(X.shape[:3] + (-1,)).astype(np.float32)
    
    # train test split (we work indices to ensure pickling for parallelization
    train_index, test_index = train_test_split(np.arange(len(X)), test_size=TEST_SPLIT, random_state=42) 

    
    # %% build and train model

    model = build_logsigrnn_model(input_shape=X.shape[1:],
                                  n_segments=(32,),
                                  drop_rate_2=DROP_RATE_2,
                                  signature_degree=SIGNATURE_DEGREE,
                                  filter_size_1=FILTER_SIZE_1,
                                  n_projection_neurons=N_PROJECTION_NEURONS,
                                  n_joints=N_JOINTS,
                                  n_classes=len(PERMITTED),
                                  learning_rate=LEARNING_RATE,
                                  n_hidden_neurons=N_HIDDEN_NEURONS)

    history = train(model, train_index, test_index, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)


    # %% record learning curve

    df = pd.DataFrame(history.history)
    df.to_csv(PATH_LEARNING_CURVE)
    
    # model should be saved at checkpoints, in case it's not make sure last version is saved
    if not os.path.exists(PATH_MODEL):
        model.save(PATH_MODEL)


    # %% data vae
    
    X = data.transpose(0, 2, 4, 3, 1).reshape(-1, 102)
    X = X[::100]

    X_train, X_test = train_test_split(X, test_size=0.15)

    encoder, decoder, vae = build_vae_model(102, 48, 4)
    vae.fit(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_test, X_test))


    # %% calc sigs

    x = data[idx, :, :, :, :1].transpose(0, 2, 1, 3, 4)
    x = x.reshape((*x.shape[:3], -1)).swapaxes(2, 3)
    with mp.Pool(mp.cpu_count()-2) as p:
        sigs = np.array(list(tqdm(p.imap(_foo, range(x.shape[0])), total=x.shape[0])))

    
    # %% sigs vae

    X = sigs.reshape((sigs.shape[0] * sigs.shape[1], -1))
    # X = X[::100]

    X_train, X_test = train_test_split(X, test_size=0.15)

    encoder, decoder, vae = build_vae_model(6732, 1024, 48)
    vae.fit(X_train, X_train, epochs=2, batch_size=32, validation_data=(X_test, X_test))


    # %% cross-validate

    # skf = StratifiedKFold(n_splits=5)
    # out = Parallel(n_jobs=5, verbose=100)(delayed(train)(train_index, test_index, epochs=N_EPOCHS, batch_size=BATCH_SIZE) for train_index, test_index in list(skf.split(X, y)))


    # %% grid search

    # model = KerasClassifier(build_fn=build_model, verbose=1)
    # grid_result = grid_search(model, X_train, y_train)

    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # params = grid_result.cv_results_['params']
    # mean_test_score = grid_result.cv_results_['mean_test_score']
    # std_test_score = grid_result.cv_results_['std_test_score']

    # for params, mean, std in zip(params, mean_test_score, std_test_score):
    #     print("%f (%f) with: %r" % (mean, std, params))

    

    # %% random search

    # model = KerasClassifier(build_fn=build_model, verbose=1)
    # random_search = randomized_search(model, X_train, y_train)
    
    # print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
    # means = random_search.cv_results_['mean_test_score']
    # stds = random_search.cv_results_['std_test_score']
    # params = random_search.cv_results_['params']
    
    # for mean, stdev, param in sorted(zip(means, stds, params), key=lambda x: x[0])[::-1]:
    #     print("%f (%f) with: %r" % (mean, stdev, param))



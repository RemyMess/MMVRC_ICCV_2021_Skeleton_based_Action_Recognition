import numpy as np
from joblib import Parallel, delayed
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
import pandas as pd
import os
import tensorflow as tf


def append_confidence_score(data, labels):

    confidence_score = np.zeros(data.shape)
    confidence_score[:, :, 1:, :, :] = (data[:, :, 1:, :, :] - data[:, :, :-1, :, :])**2
    confidence_score = np.cumsum(confidence_score, axis=2)

    return np.concatenate([data, confidence_score], axis=1)


def interpolate_frames(data, labels):

    # interpolate and bring the data to the same length
    print("Interpolating frames")

    for i in tqdm(range(data.shape[0])):
        length = labels.iloc[i]['length']
        data[i] = np.apply_along_axis(lambda x: np.interp(np.linspace(0, length, data.shape[2]), np.arange(length), x[:length]), 1, data[i]) 
   
    return data


def randomized_search(model, param_grid, X, y, **kwargs):

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, factor=0.8, min_lr=0.000001)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, n_jobs=8, cv=3, verbose=10)
    random_search.fit(X, y, callbacks=[reduce_lr], **kwargs)
    return random_search


def grid_search(model, param_grid, X, y, **kwargs):

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, verbose=1, factor=0.8, min_lr=0.000001)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-2, cv=3, verbose=10) 
    return grid.fit(X, y, callbacks=[reduce_lr], verbose=0, **kwargs)


def cross_validate(model, n_epochs, batch_size):

    skf = StratifiedKFold(n_splits=5)
    out = Parallel(n_jobs=5, verbose=100)(delayed(train)(model, train_index, test_index, epochs=n_epochs, batch_size=batch_size) for train_index, test_index in list(skf.split(X, y)))

    return out


def output_grid_search(build_model, param_grid):

    model = KerasClassifier(build_fn=build_model, verbose=1)
    search_result = grid_search(model, param_grid, X, y)

    print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))

    params = search_result.cv_results_['params']
    mean_test_score = search_result.cv_results_['mean_test_score']
    std_test_score = search_result.cv_results_['std_test_score']

    for params, mean, std in zip(params, mean_test_score, std_test_score):
        print("%f (%f) with: %r" % (mean, std, params))

    return search_result


def output_random_search(build_model, param_grid):

    model = KerasClassifier(build_fn=build_model, verbose=1)
    search_result = randomized_search(model, param_grid, X, y)

    print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
    means = search_result.cv_results_['mean_test_score']
    stds = search_result.cv_results_['std_test_score']
    params = search_result.cv_results_['params']

    for mean, stdev, param in sorted(zip(means, stds, params), key=lambda x: x[0])[::-1]:
        print("%f (%f) with: %r" % (mean, stdev, param))

    return search_result


def load_data(labels, path_data, path_labels_df):

    print("Loading data (takes less than a minute)")

    labels_df = pd.read_csv(path_labels_df)
    idx = labels_df.loc[labels_df['label'].astype(int).isin(labels)].index
    labels_df = labels_df.loc[idx]

    data = np.load(path_data).astype(np.float64)[idx]

    one_hot_encoder = OneHotEncoder(sparse=False)
    y = one_hot_encoder.fit_transform(labels_df.loc[idx, 'label'].to_numpy().reshape(-1, 1))

    return data, y

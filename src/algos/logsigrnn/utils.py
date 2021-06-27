import numpy as np
from tqdm import tqdm
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


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


def randomized_search(model, X_train, y_train, **kwargs):

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, factor=0.8, min_lr=0.000001)

    param_grid = dict(n_segments=[4, 8, 16, 32, 64],
                      n_hidden_neurons=[64, 128, 256],
                      drop_rate_2=[0.5, 0.6, 0.7, 0.8, 0.9],
                      filter_size_2=[20, 40, 60, 80],
                      batch_size=[64, 128, 256, 512])
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, n_jobs=8, cv=3, verbose=10)
    random_search.fit(X_train, y_train, callbacks=[reduce_lr], **kwargs)
    return random_search


def grid_search(model, X_train, y_train, **kwargs):

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, verbose=1, factor=0.8, min_lr=0.000001)

    param_grid = dict(n_segments=[4, 8, 16, 32, 64],
                      n_hidden_neurons=[64, 128, 256],
                      drop_rate_2=[0.5, 0.6, 0.7, 0.8, 0.9],
                      filter_size_2=[20, 40, 60, 80],
                      batch_size=[64, 128, 256, 512])

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-2, cv=3, verbose=10) 
    return grid.fit(X_train, y_train, callbacks=[reduce_lr], verbose=0, **kwargs)



def interpolate_frames(data):

    # interpolate and bring the data to the same length
    print("Interpolating frames")

    idata = np.zeros(data.shape)
    
    for i in tqdm(range(data.shape[0])):
        length = labels.iloc[i]['length']
        idata[i] = np.apply_along_axis(lambda x: np.interp(np.linspace(0, length, N_TIMESTEPS), np.arange(length), x[:length]), 1, data[i]) 
   
    X = idata[idx].transpose((0, 2, 3, 1, 4))
    X = X.reshape(X.shape[:3] + (N_AXES * N_PERSONS,))

    return idata


def randomized_search(model, X_train, y_train):

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, factor=0.8, min_lr=0.000001)

    param_grid = dict(n_segments=[4, 8, 16, 32, 64],
                      n_hidden_neurons=[64, 128, 256],
                      drop_rate_2=[0.5, 0.6, 0.7, 0.8, 0.9],
                      filter_size_2=[20, 40, 60, 80],
                      batch_size=[64, 128, 256, 512])
    
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, n_jobs=8, cv=3, verbose=10)
    random_search.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, callbacks=[reduce_lr])
    return random_search


def grid_search(model, X_train, y_train):

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=50, verbose=1, factor=0.8, min_lr=0.000001)

    param_grid = dict(n_segments=[4, 8, 16, 32, 64],
                      n_hidden_neurons=[64, 128, 256],
                      drop_rate_2=[0.5, 0.6, 0.7, 0.8, 0.9],
                      filter_size_2=[20, 40, 60, 80],
                      batch_size=[64, 128, 256, 512])

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-2, cv=3, verbose=10) 
    return grid.fit(X_train, y_train, epochs=N_EPOCHS, callbacks=[reduce_lr], verbose=0)



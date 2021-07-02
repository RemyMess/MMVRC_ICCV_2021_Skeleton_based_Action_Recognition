from itertools import combinations
import numpy as np
import iisignature
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from tensorflow.keras.models import Model, Sequential
import multiprocessing as mp
from tqdm import tqdm


def calculate_spatial_signatures(data, tuple_size, signature_degree):

    samples = data.transpose(0, 2, 1, 3, 4)
    samples = samples.reshape((*samples.shape[:3], -1)).swapaxes(2, 3)
    samples = samples[::100]

    with mp.Pool(mp.cpu_count() - 2) as p:
        signatures = np.array(list(tqdm(
            p.imap(lambda i: calculate_sample_spatial_signatures(samples[i], tuple_size, signature_degree),
                   range(samples.shape[0])), total=samples.shape[0])))

    return signatures


def calculate_sample_spatial_signatures(sample, tuple_size, signature_degree):

    tups = list(combinations(range(34), tuple_size))
    siglen = iisignature.siglength(3, signature_degree)
    signatures = np.zeros((sample.shape[0], len(list(tups)), siglen))

    for j in range(sample.shape[0]):
        for k, tup in enumerate(list(tups)):
            signatures[j, k] = iisignature.sig(sample[j, list(tup)], signature_degree)

    return signatures


def nll(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """
    Identity transform layer that adds KL divergence to the final model loss.
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


def build_vae_model(original_dim, intermediate_dim, latent_dim, epsilon_std):

    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='sigmoid')
    ])

    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    kl_z_mu, kl_z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5*t))(kl_z_log_var)

    eps = K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0], latent_dim))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([kl_z_mu, z_eps])

    x_pred = decoder(z)

    vae = Model(inputs=x, outputs=x_pred)
    vae.compile(optimizer='rmsprop', loss=nll)

    encoder = Model(x, z_mu)

    return encoder, decoder, vae


class VAETransformer:

    def __init__(self, n_input, n_intermediate, n_latent, epsilon_std) -> None:

        self.n_input = n_input
        self.n_intermediate = n_intermediate
        self.n_latent = n_latent
        self.epsilon_std = epsilon_std

        self.encoder, self.decoder, self.vae = build_vae_model(n_input, n_intermediate, n_latent, epsilon_std)

    def fit(self, X, y, **fit_params):

        train_index, test_index = train_test_split(np.arange(len(X)), test_size=0.15)

        X_train = X[train_index].reshape(-1, self.n_input)
        X_test = X[test_index].reshape(-1, self.n_input)

        self.vae.fit(X_train, X_train, validation_data=(X_test, X_test), **fit_params)

        return self

    def transform(self, X):

        # %% prepare features and labels X, y for spatial signatures vae model

        X_sigs = X.reshape((X.shape[0] * X.shape[1], -1))
        X_encoded = np.zeros((X_sigs.shape[0], self.n_latent))

        for i in tqdm(range(X_encoded.shape[0] // 10000 + 1)):
            s = slice(i * 10000, (i + 1) * 10000)
            X_encoded[s] = self.encoder(X_sigs[s])

        X_encoded = X_encoded.reshape((X.shape[0], X.shape[1], self.n_latent))

        return X_encoded

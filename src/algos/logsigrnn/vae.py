from itertools import combinations
import multiprocessing as mp
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import iisignature
from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from tensorflow.keras.models import Model, Sequential

def _foo(i):

    tups = list(combinations(range(34), TUPLE_SIZE))
    siglen = iisignature.siglength(3, SIGNATURE_DEGREE)
    sigs = np.zeros((x.shape[1], len(list(tups)), siglen))

    for j in range(x.shape[1]):
        for k, tup in enumerate(list(tups)):
            sigs[j, k] = iisignature.sig(x[i, j, list(tup)], SIGNATURE_DEGREE)

    return sigs


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


def build_vae_model():

    original_dim = 561*12
    intermediate_dim = 561 * 4
    latent_dim = 256
    epsilon_std = 1.0

    decoder = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
        Dense(original_dim, activation='sigmoid')
    ])

    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

    eps = K.random_normal(stddev=epsilon_std, shape=(K.shape(x)[0], latent_dim))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])

    x_pred = decoder(z)

    vae = Model(inputs=x, outputs=x_pred)
    vae.compile(optimizer='rmsprop', loss=nll)

    encoder = Model(x, z_mu)

    return encoder, decoder, vae


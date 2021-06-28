from itertools import combinations
import multiprocessing as mp
from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras import Model
from keras.losses import binary_crossentropy
from sklearn.preprocessing import MinMaxScaler


# generate sigs

x = data.transpose(0, 2, 1, 3, 4)
x = x.reshape((*x.shape[:3], -1)).swapaxes(2, 3)


TUPLE_SIZE = 2
SIGNATURE_DEGREE = 2


tups = list(combinations(range(34), TUPLE_SIZE))
siglen = iisignature.siglength(3, SIGNATURE_DEGREE)


def _foo(i):
    sigs = np.zeros((x.shape[1], len(list(tups)), siglen))
    for j in range(x.shape[1]):
        for k, tup in enumerate(list(tups)):
            sigs[j, k] = iisignature.sig(x[i, j, list(tup)], SIGNATURE_DEGREE)
    return sigs


with mp.Pool(mp.cpu_count()-2) as p:
    sigs = np.array(list(tqdm(p.imap(_foo, range(x.shape[0])), total=x.shape[0])))




# create vae model

X_train = sigs.reshape((sigs.shape[0], sigs.shape[1], -1)).reshape((sigs.shape[0] * sigs.shape[1], -1))
X_train = X_train[::1000]

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)

original_dim = 561*12
intermediate_dim = 561
latent_dim = 64

inputs = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling)([z_mean, z_log_sigma])

# Create encoder
encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
decoder = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# vae
reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.15)


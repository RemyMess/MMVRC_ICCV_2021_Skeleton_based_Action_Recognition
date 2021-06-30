from keras import Model, Sequential
from keras.layers import Flatten, BatchNormalization, Reshape, Dropout, Lambda, LSTM, Input, Conv2D, Conv1D, concatenate, Layer
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from src.algos.logsigrnn.sigutils import *
from src.algos.logsigrnn.dyadic_sigutils import *
import tensorflow as tf
from base_notebook.pose_data_tools.graph import Graph
from keras.optimizers import Adam
import iisignature

BatchNormalization._USE_V2_BEHAVIOR = False



class GraphAttention(Layer):
    
    def __init__(self, n_hidden_units, mat, **kwargs):
        super(GraphAttention, self).__init__(**kwargs)
        self.n_hidden_units = n_hidden_units
        self.mat = mat
        self.bn = tf.keras.layers.BatchNormalization()
    
    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', (input_shape[3], input_shape[4], self.n_hidden_units), trainable=True)
        self.attention_mask = self.add_weight('attention_mask', (17, 17, self.n_hidden_units), trainable=True)

    @tf.function
    def call(self, inputs, training=False):

        att = tf.einsum('ij,ijn->ijn', self.mat, self.attention_mask)
        att = tf.nn.softmax(att)
        # @todo fix this einsum equation
        x = tf.einsum('mln,ijklo,lon->ikln', att, inputs, self.kernel)

        x = self.bn(x, training=training)
        return tf.nn.relu(x)


X = data[:, :, :, :, :]
y = labels['label']

N_SEGMENTS = 32
N_HIDDEN_UNITS = 50

logsiglen = iisignature.logsiglength(FILTER_SIZE_2, 2)

graph = Graph()
mat = graph.get_adjacency_matrix().astype(np.float32)

input_layer = Input(shape=X.shape[1:])
gcn_layer = GraphAttention(N_HIDDEN_UNITS, mat)(input_layer)

lin_projection_layer = Conv2D(16, (FILTER_SIZE_1, 1), strides=(1, 1), data_format='channels_last')(gcn_layer)
reshape = Reshape((N_TIMESTEPS - FILTER_SIZE_1 + 1, 16 * N_JOINTS))(lin_projection_layer)
lin_projection_layer = Conv1D(FILTER_SIZE_2, 1)(reshape)

mid_output = Lambda(SP, arguments=dict(no_of_segments=N_SEGMENTS), output_shape=(N_SEGMENTS, N_HIDDEN_UNITS), name='mid_output')(lin_projection_layer)
hidden_layer = Lambda(CLF, arguments=dict(number_of_segment=N_SEGMENTS, deg_of_logsig=2, logsiglen=logsiglen), output_shape=(N_SEGMENTS, logsiglen), name='logsig')(lin_projection_layer)
hidden_layer = Reshape((N_SEGMENTS, logsiglen))(hidden_layer)
BN_layer = BatchNormalization()(hidden_layer)

mid_input = concatenate([mid_output, BN_layer])
lstm_layer = LSTM(units=64, return_sequences=True)(mid_input)
drop_layer = Dropout(0.8)(lstm_layer)
upper_mid_input = Flatten()(drop_layer)
output_layer = Dense(155, activation='softmax')(upper_mid_input)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, to_categorical(y), epochs=50, verbose=1, validation_split=0.15, batch_size=256)




# !/usr/bin/env python
# coding: utf-8


from keras import Model
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

from src.algos.linconv import LinConv
from src.algos.logsigrnn.dyadic_sigutils import *
from src.algos.logsigrnn.graph_attention import GraphAttention
from src.algos.logsigrnn.sigutils import *
from src.algos.logsigrnn.vae import *


class LogsigLSTM(Layer):

    def __init__(self, clf, n_segments, n_projection_neurons, n_hidden_neurons, drop_rate_2, signature_degree,
                 **kwargs):
        super().__init__(**kwargs)

        self.clf = clf
        self.n_segments = n_segments
        self.n_projection_neurons = n_projection_neurons
        self.n_hidden_neurons = n_hidden_neurons
        self.drop_rate_2 = drop_rate_2
        self.signature_degree = signature_degree

        logsiglen = iisignature.logsiglength(self.n_projection_neurons, self.signature_degree)

        self.sp = Lambda(SP,
                         arguments=dict(no_of_segments=max(self.n_segments)),
                         output_shape=(max(self.n_segments), self.n_projection_neurons))

        self.logsiglayer = Lambda(self.clf,
                                  arguments=dict(number_of_segment=max(self.n_segments),
                                                 deg_of_logsig=self.signature_degree, logsiglen=logsiglen),
                                  output_shape=(max(self.n_segments), logsiglen))

        self.bn = BatchNormalization()
        self.lstm = LSTM(units=self.n_hidden_neurons, return_sequences=True)
        self.dropout = Dropout(self.drop_rate_2)
        self.flatten = Flatten()

    def call(self, inputs, **kwargs):
        x = self.sp(inputs)
        y = self.logsiglayer(inputs)
        y = self.bn(y)

        # samples from the signal + log signatures
        z = concatenate([x, y])

        # LSTM
        z = self.lstm(z)
        z = self.dropout(z)
        z = self.flatten(z)

        return z


def build_logsigrnn_model(input_shape,
                          projection_mode,
                          lstm_mode,
                          n_segments,
                          drop_rate_2,
                          signature_degree,
                          filter_size_1,
                          n_projection_neurons,
                          n_joints,
                          n_classes,
                          learning_rate,
                          n_hidden_neurons):

    input_layer = Input(input_shape)

    projection_layer = get_projection_layer(input_layer, projection_mode, input_shape, n_joints, n_projection_neurons, filter_size_1)
    upper_mid_input = get_lstm_layer(projection_layer, lstm_mode, n_hidden_neurons, n_projection_neurons, n_segments, drop_rate_2, signature_degree)

    output_layer = Dense(n_classes, activation='softmax')(upper_mid_input)
    model = Model(inputs=input_layer, outputs=output_layer)

    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def get_lstm_layer(projection_layer,
                   lstm_mode,
                   n_hidden_neurons,
                   n_projection_neurons,
                   n_segments,
                   drop_rate_2,
                   signature_degree):

    if lstm_mode == 0:
        layer = LogsigLSTM(CLF, n_segments, n_projection_neurons, n_hidden_neurons, drop_rate_2, signature_degree)
        upper_mid_input = layer(projection_layer)

    elif lstm_mode == 1:
        layer = LogsigLSTM(dyadic_CLF, n_segments, n_projection_neurons, n_hidden_neurons, drop_rate_2, signature_degree)
        upper_mid_input = layer(projection_layer)

    elif lstm_mode == 2:
        upper_mid_input = concatenate([LogsigLSTM(CLF, n_segment, n_projection_neurons, n_hidden_neurons, drop_rate_2,
                                                  signature_degree)(projection_layer) for n_segment in n_segments])

    else:
        raise ValueError("lstm_mode not supported.")

    return upper_mid_input


def get_projection_layer(input_layer,
                         projection_mode,
                         input_shape,
                         n_joints,
                         n_projection_neurons,
                         filter_size_1):

    if projection_mode == 0:
        projection_layer = input_layer

    elif projection_mode == 1:
        projection_layer = GraphAttention(50, tf.ones((17, 17)))(input_layer)

    elif projection_mode == 2:
        projection_layer = LinConv(5, 1, 1)(input_layer)

    elif projection_mode == 3:
        projection_layer = Conv2D(32, (1, 1), strides=(1, 1), data_format='channels_last')(input_layer)
        projection_layer = Conv2D(16, (filter_size_1, 1), strides=(1, 1), data_format='channels_last')(projection_layer)
        reshape = Reshape((input_shape[0] - filter_size_1 + 1, 16 * n_joints))(projection_layer)
        projection_layer = Conv1D(n_projection_neurons, 1)(reshape)

    else:
        raise ValueError("projection_mode not supported.")

    return projection_layer

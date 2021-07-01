import tensorflow as tf
from esig import tosig
import iisignature
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling1D, concatenate, Dense, Dropout, LSTM, Input, InputLayer, Embedding, Flatten, Conv1D, Conv2D, MaxPooling2D, Reshape, Lambda, Permute ,BatchNormalization, GaussianNoise
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.pipeline import Pipeline
from tensorflow.keras.regularizers import l1, l2, l1_l2
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.algos.utils.sigutils import * #SP,CLF
from src.algos.linconv import LinConv
import os
import sys
import pickle

class SigRNNLinConv:
    # constants
    N_JOINTS = 17
    N_AXES = 2
    N_PERSONS = 2
    N_TIMESTEPS = 305

    #model parameters
    SIGNATURE_DEGREE = 2
    N_SEGMENTS = 32
    FILTER_SIZE_1 = N_JOINTS*N_PERSONS
    FILTER_SIZE_2 =  64
    N_HIDDEN_NEURONS = 256
    DROP_RATE_2 = 0.8
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.2

    PATH_DATA = r"/cache/gale/train_process_data.npy"
    PATH_LABELS = r"/cache/gale/train_process_label.pkl"
    MODEL_NAME = os.path.dirname(__file__) + "/linrnnmodels/"+"base32.h5"

    def __init__(self,sig_data_wrapper,batch_size=256,lr=0.001,epochs=50):
        self.sig_data_wrapper=sig_data_wrapper
        self.batch_size=batch_size
        self.lr=lr
        self.epochs=epochs

        self.X_train, self.y_train, self.X_test, self.y_test = sig_data_wrapper.train_data, sig_data_wrapper.train_label, sig_data_wrapper.val_data, sig_data_wrapper.val_label
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)

        self.PERMITTED = np.arange(155)

        self.model=self.build_model(signature_deg=self.SIGNATURE_DEGREE)


    def build_model(self,signature_deg):
        logsiglen = iisignature.logsiglength(self.FILTER_SIZE_2, signature_deg)

        input_shape=(SigRNNLinConv.N_TIMESTEPS, SigRNNLinConv.N_AXES+1, SigRNNLinConv.N_JOINTS* SigRNNLinConv.N_PERSONS)
        input_layer = Input(input_shape)
        noise_layer=GaussianNoise(0.01)(input_layer)

        #Learns optimal linear configurations of the joint vectors
        joint_config_layer=LinConv(self.FILTER_SIZE_1,1,3,activation='relu',use_bias=False)(noise_layer)

        reshape_layer = Reshape((input_shape[0], self.FILTER_SIZE_1*(SigRNNLinConv.N_AXES+1)))(joint_config_layer)

        temp_config_layer = Conv1D(self.FILTER_SIZE_2, 5, padding='same')(reshape_layer)
        noise_layer=GaussianNoise(0.01)(temp_config_layer)

        mid_output = Lambda(lambda x: SP(x, self.N_SEGMENTS), output_shape=(self.N_SEGMENTS, self.FILTER_SIZE_2), name='start_position')(noise_layer)

        hidden_layer = Lambda(lambda x: CLF(x, self.N_SEGMENTS, signature_deg, logsiglen), output_shape=(self.N_SEGMENTS, logsiglen), name='logsig')(noise_layer)

        hidden_layer = Reshape((self.N_SEGMENTS, logsiglen))(hidden_layer)

        BN_layer = BatchNormalization()(hidden_layer)

        # samples from the signal + log signatures
        mid_input = concatenate([mid_output, BN_layer], axis=-1)

        # LSTM
        lstm_layer = LSTM(units=SigRNNLinConv.N_HIDDEN_NEURONS, return_sequences=True)(mid_input)

        # Dropout
        drop_layer = Dropout(SigRNNLinConv.DROP_RATE_2)(lstm_layer)
        output_layer = Flatten()(drop_layer)
        output_layer = Dense(len(self.PERMITTED), activation='softmax')(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        adam = Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics = ['accuracy'])

        print(model.layers[1].get_config())

        return model

    def run(self,render_plot):
        history=self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=SigRNNLinConv.VALIDATION_SPLIT)


        print("Evaluate on test data")
        results = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        print("test loss, test acc:", results)


        self.model.save_weights(self.MODEL_NAME)
        print('model saved at',self.MODEL_NAME)
        # Plot training accuracy
        if render_plot:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

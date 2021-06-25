#Layer performs convolution to optimize for linear combinations of positional data in coordinate form (e.g. landmarks of an image) for further processing
#Accepts input of shape (frames,coordinates,landmarks)
#It uses the same kernel for all depth layers (coordinates) in contrast to the convention for the traditional convolution.
#Suitable e.g. for video analysis with landmark data, to optimize the spatial data for temporal analysis
#Curently just for 2-dim vectors with no bias: to be expanded.

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

class LinConv(Layer):
    """Performs convolution to detect features in the form of linear combinations of positional data (landmarks) presented in coordinate form (e.g. landmarks of an image).

    Convolves with the same kernel in the depth direction (coordinates) in contrast to the convention for the traditional convolution.

    Accepts input of shape (frames,coordinates,landmarks).

    """
    def __init__(self,filters,kernel_size,dim,activation=None,use_bias=False,**kwargs):
        """Arguments
        filters=number of filters
        kernel_size=temporal size of kernel (int)
        dim=dimesion of coordinate space
        activation=activation function (ReLu or None) (string)
        use_bias=not implemented
        """
        self.filters = filters
        #kernels of shape(n,1)
        self.kernel_size = (kernel_size,1)
        self.activation=activation
        self.use_bias=use_bias
        self.dim=dim
        super(LinConv, self).__init__(**kwargs)
        # self.initializer=keras.initializers.get('glorot_uniform')

    def build(self, input_shape):
        shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(name='kernel', shape=shape,
                                      initializer='glorot_uniform')
        dim=input_shape[1]
        super(LinConv, self).build(input_shape)

    def call(self, x):

        # z=tf.zeros(self.kernel[:,0].shape)
        # dup_cols = K.stack([self.kernel[:,0],z],axis=1)
        # z=K.conv2d(x, dup_cols,padding='same')

        # print(x.shape)
        # print(x[:,:,1].shape)
        # print(tf.expand_dims(x[:,:,1],axis=2).shape)
        # print(K.conv2d(tf.expand_dims(x[:,:,1],axis=2),tf.expand_dims(self.kernel[:,0],axis=1))[:,:,0,:])
        conv_coord=[]
        for i in range(self.dim):
            conv_coord.append(K.conv2d(tf.expand_dims(x[:,:,i],axis=2),tf.expand_dims(self.kernel[:,0],axis=1))[:,:,0,:])

        z = K.stack(conv_coord,axis=2)
        # print(z.shape)

        if self.activation=='relu':
            return K.relu(z)
        elif self.activation is None:
            return z
        else:
            raise ValueError('Unsupported activation' + str(self.activation))


    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        base_config=super(LinConv,self).get_config()
        # config={"initializer":keras.initializers.serialize(self.initializer)}
        # return dict(list(base_config.items())+list(config.items()))
        return base_config

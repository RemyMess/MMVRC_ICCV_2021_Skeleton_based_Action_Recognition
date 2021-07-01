from tensorflow.keras.layers import BatchNormalization, Layer
import tensorflow as tf

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


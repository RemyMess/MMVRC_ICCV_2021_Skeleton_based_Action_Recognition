import tensorflow as tf
from tensorflow import ones,expand_dims
from tensorflow.sparse import SparseTensor
from tensorflow.sparse import sparse_dense_matmul as spMul
from tensorflow.keras.layers import Layer

def TFfactorial(n):
    n_float=tf.constant(float(n))
    return tf.math.exp(tf.math.lgamma(n_float+1))

def SignatureTensor(path_batch,step=1,degree=3):
    """
    Input:
    path,step,max_degree
    recieves a batch of paths - an array with dimensions batch_size x Time x (space dims) and returns
    signature stand-in with sample at "step" intervals
    """
    length=path_batch.shape[1]
    depth=path_batch.shape[2]
    batch_size=-1
    """
    generate vector of differences at "step" intervals
    """
    # print('*****************path_batch SHAPE****************: ',path_batch.shape)
    # diff=tf.reshape(path_batch,[batch_size,length,depth]) # reshape path into batch_sizextimeXtotal_space_dim
    # print('*****************DIFF SHAPE****************: ',diff.shape)

    diff=path_batch
    ### shift rows over at increment "step" with 0 padding
    paddings=[[0,0],[step,0],[0,0]]
    diff=tf.pad(diff,paddings)
    shifted_diff=diff
    diff=tf.roll(diff,shift=-step,axis=1)
    ### take differences from shifted
    diff=diff-shifted_diff
    diff=diff[:,step:,:] # cut off excess padding
    """
    generate signatures
    """
    factorial_denoms=tf.stack([1/TFfactorial(i) for i in range(degree+1)],axis=0)
    # compute tensor powers of vectors

    diff_powers=[tf.reshape(diff,(batch_size,length*depth))]
    diff_multiplier=tf.reshape(diff,(batch_size,1,-1))
    for i in range(1,degree):
        diff_powers.append(tf.expand_dims(diff_powers[i-1],axis=-1)@diff_multiplier)
    # add up relevant powers to get truncated signature
    signature=[]
    for i in range(1,degree+1):
        signature_i=diff_powers[i-1]
        signature_i=tf.reshape(signature_i,[batch_size,length]+[depth for j in range(i)])
        signature_i=tf.transpose(signature_i,[0]+[j+1 for j in range(1,i+1)]+[1])
        signature_i=signature_i@tf.ones((signature_i.shape[-1],1)) #this isn't correct
         #when have repeated index need to divide by n!
        signature_i=tf.transpose(signature_i,[0,i+1]+[j+1 for j in range(i)])
        signature.append(tf.reshape(signature_i,(batch_size,length*(depth**i))))
    return tf.concat(signature,axis=1)

class CLF_Adam_Layer(Layer):
    def __init__(self,n_segments,signature_deg,**kwargs):
        super(CLF_Adam_Layer, self).__init__(**kwargs)
        self.n_segments=n_segments
        self.signature_deg=signature_deg

    def call(self,x):
        length=x.shape[1]
        new_length=length-(length%self.n_segments)
        segments=tf.split(x[:,:new_length,:],self.n_segments,axis=1)
        signatures=[]
        for s in segments:
            signatures.append(SignatureTensor(s,step=1,degree=self.signature_deg))
        return tf.stack(signatures,axis=1)

    # def compute_output_shape(self):
    #     return (self.n_segments,int(((64)**(self.signature_deg+1)-1)/(64-1))-1)

    def get_config(self):
        base_config=super(CLF_Adam_Layer,self).get_config()
        # config={"initializer":keras.initializers.serialize(self.initializer)}
        # return dict(list(base_config.items())+list(config.items()))
        return base_config

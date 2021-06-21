from functools import partial

import iisignature
import numpy as np
import tensorflow as tf
from keras import backend as K
from functools import partial


def SP(path, no_of_segments):
    start_vec = np.linspace(1, K.int_shape(path)[1], no_of_segments + 1)
    start_vec = [int(round(x)) - 1 for x in start_vec]
    fn = partial(K.gather, indices=start_vec[:no_of_segments]) 
    return tf.map_fn(fn, path)


def compute_logsig_features(path, number_of_segment, deg_of_logsig):
    """
    The implementation of computing the log-signature of segments of path.
    path: dimension (sample_size,n, d)
    number_of_segment: the number of segments
    deg_of_logsig: the degree of the log-signature
    """
    nT = int(np.shape(path)[1])
    dim_path = int(np.shape(path)[-1])
    t_vec = np.linspace(1, nT, number_of_segment + 1)
    t_vec = [int(round(x)) for x in t_vec]
    s = iisignature.prepare(dim_path, deg_of_logsig)
    MultiLevelLogSig = []
    for k in range(int(np.shape(path)[0])):
        tmpMultiLevelLogSig = np.zeros((number_of_segment, iisignature.logsiglength(dim_path, deg_of_logsig)))
        for i in range(number_of_segment):
            temp_path = path[k][t_vec[i] - 1:t_vec[i + 1], :]
            temp_start = temp_path[0]
            try:
                tmpMultiLevelLogSig[i, :] = iisignature.logsig(temp_path, s)
            except SystemError:
                pass
        MultiLevelLogSig.append(tmpMultiLevelLogSig)
    return np.float32(np.array(MultiLevelLogSig))


def CLF_grad(op, grad, deg_of_logsig, number_of_segment):
    """
    The backward operation of computing the derivatives w.r.t. the elements of path. The output is of dimension (sample_size, n, d),
    the same as the dimension of input path.

    op: op stores input parameters
    grad: the flown gradient from backpropagation
    number_of_segment: the number of segments
    deg_of_logsig: the degree of the log-signature
    """
    CLF_g_Imp = partial(CLF_grad_Imp, deg_of_logsig=deg_of_logsig, number_of_segment=number_of_segment)
    return tf.compat.v1.py_func(CLF_g_Imp, [grad] + list(op.inputs), [tf.float32])


def CLF(path, number_of_segment, deg_of_logsig, logsiglen):
    """
    Tensorflow forward operation of computing the log-signature of segments of path, where the path is of
    dimension (sample_size, n, d). The operation splits the path to N segments, and compute the degree M logsig of each
    segment then concatenate which results in a new path of dimension (sample_size, N, iisignature.logsiglength(d, M)).

    path: dimension (sample_size,n, d)
    number_of_segment: the number of segments
    deg_of_logsig: the degree of the log-signature
    """
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    CLF_g = partial(CLF_grad, deg_of_logsig=deg_of_logsig, number_of_segment=number_of_segment)
    tf.RegisterGradient(rnd_name)(CLF_g)
    g = tf.compat.v1.get_default_graph()
    _CLF = partial(compute_logsig_features, deg_of_logsig=deg_of_logsig, number_of_segment=number_of_segment)
    with g.gradient_override_map({"PyFunc": rnd_name}):
        f = tf.compat.v1.py_func(_CLF, [path], tf.float32)
        f.set_shape((None, number_of_segment, logsiglen))
        return f


def CLF_grad_Imp(g, path, deg_of_logsig, number_of_segment):
    """
    The implementation of computing the derivatives of gradient from backpropagation.
    g: the flown gradient from backpropagation
    path: dimension (sample_size,n, d)
    number_of_segment: the number of segments
    deg_of_logsig: the degree of the log-signature
    """
    nT = int(np.shape(path)[1])

    dim_path = int(np.shape(path)[-1])
    t_vec = np.linspace(1, nT, number_of_segment + 1)
    t_vec = [int(round(x)) for x in t_vec]
    s = iisignature.prepare(dim_path, deg_of_logsig)
    MultiLevelBP = []
    for k in range(int(np.shape(path)[0])):
        tmpMultiLevelBP = np.zeros([1, np.shape(path)[-1]])
        for i in range(number_of_segment):
            temp_path = path[k][t_vec[i] - 1:t_vec[i + 1], :]
            tempBP = iisignature.logsigbackprop(g[k][i], temp_path, s, None)
            tmpMultiLevelBP[-1] += tempBP[0]
            tempBP = np.delete(tempBP, 0, axis=0)
            tmpMultiLevelBP = np.concatenate((tmpMultiLevelBP, tempBP), axis=0)
        MultiLevelBP.append(tmpMultiLevelBP)

    return np.float32(np.array(MultiLevelBP))


import iisignature
import numpy as np
import tensorflow as tf
from functools import partial


def dyadic_logsig_features(path, n_segments_range, deg_of_logsig):

    dim_path = path.shape[2]
    logsigdim = iisignature.logsiglength(dim_path, deg_of_logsig)

    logsigs = np.zeros((path.shape[0], max(n_segments_range), logsigdim * len(n_segments_range)))
    s = iisignature.prepare(dim_path, deg_of_logsig)

    for i, n_segments in enumerate(n_segments_range[::-1]):
        factor = max(n_segments_range) / n_segments
        segments = np.array_split(path, n_segments, axis=1)
        logsigs[:, :, i*logsigdim:(i+1)*logsigdim] = \
            np.array([[iisignature.logsig(sample_segment, s) for sample_segment in segment] for segment in segments]) \
                .repeat(factor, axis=0) \
                .transpose((1, 0, 2)) / factor

    return np.float32(logsigs)


def dyadic_logsig_grad(g, path, n_segments_range, deg_of_logsig):
    
    dim_path = path.shape[2]
    logsigdim = iisignature.logsiglength(dim_path, deg_of_logsig)

    ret = np.zeros(path.shape)
    s = iisignature.prepare(dim_path, deg_of_logsig) 

    for i, n_segments in enumerate(n_segments_range[::-1]):
        p_segments = np.array_split(path, n_segments, axis=1)
        r_segments = np.array_split(ret, n_segments, axis=1)
        for j, (p_segment, r_segment) in enumerate(zip(p_segments, r_segments)):
            r_segment += np.array([iisignature.logsigbackprop(g[k, j, i*logsigdim:(i+1)*logsigdim], sample_segment, s) for k, sample_segment in enumerate(p_segment)]) * 2**i

    return np.float32(ret)


def dyadic_CLF(path, n_segments_range, deg_of_logsig, logsiglen):

    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1e8))
    CLF_g = partial(dyadic_logsig_grad, deg_of_logsig=deg_of_logsig, n_segments_range=n_segments_range)
    
    tf.RegisterGradient(rnd_name)(CLF_g)
    g = tf.compat.v1.get_default_graph()
    _CLF = partial(dyadic_logsig_features, deg_of_logsig=deg_of_logsig, n_segments_range=n_segments_range)
    
    with g.gradient_override_map({"PyFunc": rnd_name}):
        f = tf.compat.v1.py_func(_CLF, [path], tf.float32)
        f.set_shape((None, max(n_segments_range), len(n_segments_range) * logsiglen))
        return f
    

def dyadic_CLF_grad(op, grad, deg_of_logsig, n_segments_range):
    
    CLF_g_Imp = partial(dyadic_logsig_grad, deg_of_logsig=deg_of_logsig, n_segments_range=n_segments_range)
    return tf.compat.v1.py_func(CLF_g_Imp, [grad] + list(op.inputs), [tf.float32])


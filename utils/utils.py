import errno

import numpy as np
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def softmax(z):
    zz = z - np.max(z)
    sm = (np.exp(zz).T / np.sum(np.exp(zz), axis=0)).T
    return sm


def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

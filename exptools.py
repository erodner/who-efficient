import numpy as np

""" computing the relative error between two matrices """
def relativeError ( A, B, ignore_zeros=False ):
    D = A - B
    if ignore_zeros:
        D[A==0.0] = 0.0
    err = np.sum ( np.abs(D), axis=None ) / ( np.sum(np.abs(B),axis=None)+1e-5)
    return err



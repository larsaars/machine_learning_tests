"""
convolution and pooling with numpy
"""

import numpy as np


# Sobel filter for detecting vertical edges
F_ver = np.array([
    [1, 0, -1], 
    [2, 0, -2],
    [1, 0, -1]
])

# Sobel filter for detecting horizontal edges
F_hor = np.array([
    [1, 2, 1], 
    [0, 0, 0],
    [-1, -2, -1]
])

def out_size(m_in, k, p, s):
    '''
    calculate the outsize of pooling / convolution
    '''
    m_out = (m_in - k + 2*p) / s + 1
    m_floor = np.floor(m_out)

    if not np.isclose(m_out, m_floor):
        # raise ValueError(f'Expected integer as output: {m_out}')
        pass

    return int(m_floor)


def pad2(X, p=0):
    '''
    2d padding with zeros
    '''

    h, w = X.shape

    Z = np.zeros((2*p + h, 2*p + w))
    Z[p:h+p, p:w+p] = X

    return Z



def conv2(X, F=[[1]], p=0, s=[1, 1]):
    '''
    X is a grayscale image of shape (h, w)
    F is a filter of shape (kh, kw)
    p is the padding size
    s is the stride

    The padding size is identical for the height and width. The stride s is a list
    consisting of two elements. The first element s[0] is the stride along the
    height and the second element s[1] is the stride along the width.

    This function throws an error if the filter size, padding, and stride
    are chosen in such a way that the output height or width is not an integer.
    '''

    X = np.array(X)
    F = np.array(F)

    h, w = X.shape
    kh, kw = F.shape
    sh, sw  = s

    h_out = out_size(h, kh, p, sh)
    w_out = out_size(w, kw, p, sw)

    Z = pad2(X, p)

    O = np.zeros((h_out, w_out))

    for r in range(h_out):
        for c in range(w_out):
            O[r, c] = (Z[sh*r:sh*r+kh, sw*c:sw*c+kw] * F).sum()

    return O

def pool2(X, k=[2, 2], s=[1, 1], action=np.min):
    '''
    X is a grayscale image of shape (h, w)
    k is the filter size
    s is the stride

    The filter size k and stride s are lists consisting of two elements.
    The first sizes k[0], s[0] refer to the height and the second sizes
    k[1], s[1] refer to the width.

    This function throws an error if the filter size and stride are chosen
    in such a way that the output height or width is not an integer.
    '''

    X = np.array(X)

    h, w = X.shape
    kh, kw = k
    sh, sw  = s

    h_out = out_size(h, kh, 0, sh)
    w_out = out_size(w, kw, 0, sw)

    O = np.zeros((h_out, w_out))

    for r in range(h_out):
        for c in range(w_out):
            O[r, c] = action(X[sh*r:sh*r+kh, sw*c:sw*c+kw])

    return O


def pad1(X, p=0):
    '''
    1d padding with zeros
    '''

    w = X.shape[0]

    Z = np.zeros(2*p + w)
    Z[p:w+p] = X

    return Z



def conv1(X, F=[1], p=0, s=1):
    '''
    1d convolution
    '''
    
    X = np.array(X)
    F = np.array(F)
    
    w = X.shape[0]
    k = F.shape[0]
    
    out = out_size(w, k, p, s)
    
    Z = pad1(X, p)
    
    O = np.zeros(out)
    
    for i in range(out):
        si = s * i
        O[i] = (Z[si:si+k] * F).sum()
            
    return O

    
def pool1(X, k=2, s=1, action=np.max):
    '''
    1 dim pooling
    '''
    
    X = np.array(X)
    w = X.shape[0]
    out = out_size(w, k, 0, s)
    O = np.zeros(out)
    
    for i in range(out):
        si = s * i
        O[i] = action(X[si:si+k])
            
    return O


#!/usr/bin/env python3

"""
Implements a function for flattening an ndarray with hilbert curve
and an module (layer) for pytorch.
"""

import time

import numpy as np
from hilbert import decode, encode

from torch import nn
import torch

import sys
sys.path.append('../conv_pool')

from conv import pad1


def hilbert_output_size(L, dim=2):
    '''
    L array length
    dim dimension the output hilbert curve

    returns the size of the next fitting 1d array length to fit the
    given dimensions of the hilbert curve

    If the sizes do not match, the 1d array can be augmented using
    repetition or padding.
    
    hilbert curves have the size (2**dim)**order
    the sizes are always in square, cubic, etc. form
    '''

    # base belonging to dimension
    dim_base = 2**dim

    # get the nearest order (the exponent to the base)
    # to be an correct order, the order must be an integer
    # if it is not, augment array to the next order (the size)
    order = np.log(L) / np.log(dim_base)
    order_floor = np.floor(order)

    if np.isclose(order, order_floor):
        return L
    else:
        # augment to next order
        return int(dim_base**(order_floor + 1))




def hilbert_flatten(array: np.ndarray, byte_size=8) -> np.ndarray:
    '''
    takes ndarray and does not change overall length, but array will be flattened with an hilbert curve
    '''

    D = array.ndim
    S = np.arange(np.array(array.shape).prod())
    L = decode(S, D, byte_size)

    return array[tuple(L.T)]


def hilbert_expand(array: np.ndarray, dim=2, byte_size=8) -> np.ndarray:
    '''
    array n-dimensional array to expand in order 
    dim number of output dimensions
    byte_size byte size for the hilbert algorithm

    takes 1-darray and expands it in n dims
    '''

    array = array.flatten()
    a_len = array.shape[0]

    S = np.arange(a_len)
    L = decode(S, dim, byte_size)

    O = np.zeros(dim * tuple([int(a_len**(1 / dim))]))
    O[tuple(L.T)] = array

    return O


def hilbert_remap(array: np.ndarray, out_dim=1, byte_size=8, p=0):
    ''' 
    array n-dimensional array
    out_dim dimensions of the output array
    byte_size the byte size for the hilbert algorithm
    p padding that will be applied as soon as the array is in 1d form

    remaps an n-dimensional array to m dimensions

    1. map to 1d
    2. pad
    3. map to n dim
    '''

    F = hilbert_flatten(array, byte_size)
    if p > 0: F = pad1(F, p)
    
    return hilbert_expand(F, out_dim, byte_size)

def hilbert_auto_adjust_size(X: np.ndarray, dim=2):
    '''
    adjust size of 1d array X to match the next hilbert size
    '''

    # length of X
    L = X.shape[0]
    # next fit
    N = hilbert_output_size(L, dim=dim)

    # if already fits, return
    if N == L:
        return X

    # augment to X to next fit
    # first try by repeating
    scalar = L / N
    X = np.repeat(X, scalar, axis=-1)
    L = L * scalar

    # if it now fits, return
    if N == L:
        return X

    # else, augment by padding with N-L zeros
    padding = int(np.floor((N - L) / 2))
    X = pad1(X, padding)
    L += 2 * padding

    # if the remainder is not divisible by 2 (padding is always 2*p), add one zero to right
    if N - L == 1:
        Z = np.zeros(N)
        Z[:-2] = X
        Z[-1] = 0
        return Z
    else:
        return X


class HilbertFlatten(nn.Module):
    def __init__(self):
        super(HilbertFlatten, self).__init__()

    
    def forward(self, x):
        # detach and convert to numpy array
        x = x.detach().numpy()
        x = hilbert_flatten(x)
        return torch.from_numpy(x)


if __name__ == '__main__':
    start = time.time()

    a = np.array([[12, 15, 5, 0],
                  [3, 11, 3, 7],
                  [9, 3, 5, 2],
                  [4, 7, 6, 8]])

    f = hilbert_flatten(a)
    g = hilbert_expand(f, 2)

    print(f'Operations took {(time.time() - start) * 1000}ms.')

    print(f)
    print(g)



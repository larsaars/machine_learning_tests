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


"takes ndarray and does not change overall length, but array will be flattened with an hilbert curve"
def hilbert_flatten(array: np.ndarray, byte_size=8) -> np.ndarray:
    D = array.ndim
    S = np.arange(np.array(array.shape).prod())
    L = decode(S, D, byte_size)
    return array[tuple(L.T)]


"takes 1-darray and expands it in n dims"
def hilbert_expand(array: np.ndarray, dim=2, byte_size=8) -> np.ndarray:
    array = array.flatten()
    a_len = array.shape[0]

    S = np.arange(a_len)
    L = decode(S, dim, byte_size)

    O = np.zeros(dim * tuple([int(a_len**(1 / dim))]))
    O[tuple(L.T)] = array[S]

    return O


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

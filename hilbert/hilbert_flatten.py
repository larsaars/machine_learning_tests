#!/usr/bin/env python3

"""
Implements a function for flattening an ndarray with hilbert curve
and an module (layer) for pytorch.
"""

import time

import numpy as np
from hilbert import decode

from torch import nn
import torch


"takes ndarray and does not change overall length, but array will be flattened with an hilbert curve"
def hilbert_flatten(array: np.ndarray) -> np.ndarray:
    D = array.ndim
    S = np.arange(np.array(array.shape).prod())
    L = decode(S, D, 8).T.tolist()

    return array[tuple(L)]


class HilbertFlatten(nn.Module):
    def __init__(self, ):
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

    print(f'Operation took {time.time() - start}ns.')
    print(f)

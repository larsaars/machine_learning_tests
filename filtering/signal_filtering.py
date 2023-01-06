#!/usr/bin/env python3

import numpy as np
import numba as nb

# VERSION WITHOUT FFTCONVOLVE
@nb.njit(parallel=True, fastmath=True)
def symm_filt(X, kernel_size=3, a=0.6, with_init=False):
    """
    Symmetric linear convolution combined with flatting

    :param X: input 1d array
    :param kernel_size: size of the kernel filter
    :param a: Influence of previous A values (flattening factor)
    :param with_init: return with inited values (same len)
    """

    # create filter
    mu_idx = (kernel_size - 1) // 2  # index of mid value
    b = mu_idx + 1

    F = np.zeros(kernel_size)

    # create filters (added up they sum up to 1, linear symmetic filter)
    bsq = 1 / b**2
    for x in range(b):
        F[x + mu_idx] = (-x + b) * bsq

    # add symmetric values to filter
    for i in range(mu_idx):
        F[i] = F[kernel_size - i - 1]

    # make filters have only the total size of 1 - a
    F *= 1 - a

    # out array
    A = np.zeros(X.shape)

    # shape
    rows, cols = X.shape

    # initial values (depth-first)
    for c in nb.prange(cols):
        for r in range(kernel_size - 1):
            A[r, c] = X[r, c]

    # do depth first filtering
    for c in nb.prange(cols):
        for r in range(kernel_size - 1, rows):
            # apply multiplication with previous filterd value of of filter center index
            # (reduce propelling effect)
            A[r, c] += a * A[r - mu_idx, c]

            # do normal convolution (without FFT)
            for i in nb.prange(len(F)):
                A[r, c] += F[i] * X[r - i, c]

    return A if with_init else A[kernel_size - 1:]
  
 # determine where new cycles start
@nb.njit
def determine_cycles(cycles):
    # indexes where cycles end (+1 idx)
    idx = []

    # is old cycle as long as number augments,
    # is new cycle when is not augmenting anymore
    last = cycles[0]

    for i in range(len(cycles)):
        if cycles[i] < last:
            idx.append(i)
        last = cycles[i]

    # append last index (len of cycles)
    idx.append(len(cycles))

    return idx
  
  

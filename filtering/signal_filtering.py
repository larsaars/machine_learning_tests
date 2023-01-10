#!/usr/bin/env python3

import numpy as np
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def symm_filt(X, kernel_size=3, a=0.6, with_init=False):
    """
    Symmetric linear convolution combined with flatting (could be also done combined with fft convolution)

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

@nb.njit(parallel=True)
def shift_filt(X, shift=0.3):
    """
    Shift filtering, assumes filters out too large positive changes (for example reflections)
    
    :param X: input data
    :param shift: the shift tested against
    """
    
    # copy of filtered values
    A = np.zeros_like(X)
    
    # get shape
    rows, cols = X.shape
    
    # init first row
    A[0] = X[0]
    
    # perform filtering (depth first)
    for c in nb.prange(cols):
        for r in range(1, rows):
            A[r, c] = min(X[r, c], X[r - 1, c], A[r - 1, c] + shift)
    
    
    return A
  
@nb.njit
def determine_cycles(cycles):
    """
    determine where new cycles start
    
    :param cycles: numpy array of counter
    :return: the indexes where new cycles end
    """
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

@nb.njit(parallel=True)
def create_cyclic_features(X, cycles, H=3):
    """
    parse array with historic data with cycles
    from X. When new cycle starts the hist values will be zeros

    ex:

    X = np.array([[1], [2], [3], [4], [5], [6], [7]])
    cycles = np.array([1, 2, 3, 4, 1, 2, 3])
    N = 2

                                                                        here new cycle
                                                                               ↓
    out = [[[1.], [0.]],  [[2.], [1.]],  [[3.], [2.]],  [[4.], [3.]],  [[5.], [0.]],  [[6.], [5.]],  [[7.], [6.]]]


    :param X: input data
    :param cycles: the cycles
    :param H: number of hist data per feature
    :return: transformed array
    """

    # get cycle indexes
    cycle_idx = np.array(determine_cycles(cycles))

    # out array is the same shape as input but with new
    # inner dimension with the historic data
    out = np.zeros((X.shape[0], H, X.shape[1]))

    # paraellel cycle-wise computing
    for c in range(len(cycle_idx)):
        begin = 0 if c == 0 else cycle_idx[c - 1]  # included index
        end = cycle_idx[c]  # excluded index

        # loop through all indexes of cycle
        for i in nb.prange(begin, end):

            # loop through num of historic data
            # break when reached end of cycles
            for h in range(H):
                last = i - h

                if last < begin:
                    break

                out[i, h] = X[last]

    return out
  
  

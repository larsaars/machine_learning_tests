import numpy as np
from hilbert import decode


def hilbert_flatten(array):
    D = array.ndim
    S = np.arange(np.array(array.shape).prod())
    L = decode(S, D, 8).T.tolist()

    return array[tuple(L)]


if __name__ == '__main__':
    a = np.array([[12, 15, 5, 0],
                  [3, 11, 3, 7],
                  [9, 3, 5, 2],
                  [4, 7, 6, 8]])

    print(hilbert_flatten(a))

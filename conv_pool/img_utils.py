"""
some matplotlib etc. image utils
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io


def plt_im(X):
    '''Plots grayscale image X.'''

    plt.figure(figsize=(36, 21))
    plt.imshow(X, cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.show()


def read_im(path):
    '''reads grayscale image'''
    img = io.imread(path)
    img = color.rgb2gray(img)
    return img


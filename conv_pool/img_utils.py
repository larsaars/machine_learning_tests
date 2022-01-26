"""
some matplotlib etc. image utils
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io


def plt_im(X, g=True):
    '''Plots grayscale image X (or colored)'''

    plt.figure(figsize=(36, 21))
    if g:
        plt.imshow(X, cmap='gray')
    else:
        plt.imshow(X)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def read_im(path, g=True):
    '''reads grayscale image (or colored)'''
    img = io.imread(path)
    if g: img = color.rgb2gray(img)
    return img


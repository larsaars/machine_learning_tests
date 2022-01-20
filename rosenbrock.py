#!/usr/bin/env python3

"""
testing out rosenbrock function optimization with
gradient descent

inspiration:
https://www.indusmic.com/post/rosenbrock-function
"""

import numpy as np

eta = .0065
iterations = 100

x1, x2 = .5, .5

f = lambda x1, x2: 100*(x2-x1**2)**2+(x1-1)**2
f_x1 = lambda x1, x2: 2(x1−1)−400x1(x2−x21)
f_x2 = lambda x1, x2: 


for i in range(iterations):
    x1 = eta * f_x1(x1)
    x2 = eta * f_x2(x2)

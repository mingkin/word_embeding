# -*- coding: utf-8 -*-
"""
# Author  : Ming
# File    : {NAME}.py
# Time    : 2018/12/11 0011 下午 5:18
"""


import numpy as np
import matplotlib.pyplot as plt

'x为共现矩阵'
la = np.linalg
words = ["I" , "like" , "enjoy" , "deep" , "learning" , "NLP" , "flying" , "."]
X = np.array([
    [0,2,1,0,0,0,0,0],
    [2,0,0,1,0,1,0,0],
    [1,0,0,0,0,0,1,0],
    [0,1,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,1],
    [0,1,0,0,0,0,0,1],
    [0,0,1,0,0,0,0,1],
    [0,0,0,0,1,1,1,0]
])
U, s, Vh = la.svd(X, full_matrices=False)

print(U, s, Vh)
for i in range(len(words)):
    plt.scatter(U[i, 0], U[i, 1])
    plt.text(U[i, 0], U[i, 1], words[i])

plt.show()








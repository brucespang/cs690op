from __future__ import division
from solve import *
import numpy as np
from common import *
import matplotlib.pyplot as plt

gradient = lambda x: np.matrix([[1], [2]])

def project(x):
    x_p = project_plane(np.matrix([[1], [1]]), 1)(x)
    if x_p[0,0] >= 0 and x_p[1,0] >= 0:
        return x_p
    elif x_p[0,0] < 0:
        return np.matrix([[0], [1]])
    elif x_p[1,0] < 0:
        return np.matrix([[1], [0]])


def plot():
    opt = np.matrix([[1], [0]])
    x0 = np.matrix([[0], [1]])
    xs =  list(linesearch_project(x0, lambda k: 1, gradient, projection=project))

    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("l2 distance from optimal")
    plt.plot([numpy.linalg.norm(x - opt) for x in xs])
    plt.axis(ymin=0,xmin=0)
    plt.savefig("projected-descent.png")

plot()
plt.show()

from __future__ import division
import random
import solve
import numpy
import matplotlib.pyplot as plt
import itertools
import scipy.optimize

def f(x):
    return x**2

def g_f(x):
    return 2*x

def h_f(x):
    return numpy.matrix([[2]])

def plot_convergence(max_iters):
    opt = numpy.array([0])
    x0 = numpy.array([3])
    methods = {
        "newtons": solve.newtons(x0, g_f, h_f),
        "linear": solve.linesearch(x0, lambda k: 0.1, g_f),
    }

    plt.figure()
    plt.title("Convergence of optimization methods on x**2")
    plt.xlabel("Iteration")
    plt.ylabel("l2 distance from optimal")
    legends = []
    for t,xs in methods.items():
        xs = itertools.islice(xs, max_iters)
        plt.plot([numpy.linalg.norm(x - opt) for x in xs])
        legends.append(t)
    plt.legend(legends)
    plt.axis(ymin=0,xmin=0)

plot_convergence(40)
plt.show()

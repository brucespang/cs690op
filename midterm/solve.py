from __future__ import division
import numpy

def linesearch_project(x, stepsize, gradient, projection=lambda x: x):
    g = gradient(x)
    x_prev = None
    while x_prev is None or numpy.linalg.norm(x - x_prev) > 0:
        yield x
        g = gradient(x)
        x_prev = x
        y = x - stepsize(x)*g
        x = projection(y)
        print "y", y
        print "x", x

def linesearch(x, stepsize, gradient):
    g = gradient(x)
    while g.any():
        g = gradient(x)
        x = x - stepsize(x)*g
        yield x

def newtons(x, gradient, hessian):
    g = gradient(x)
    while g.any():
        g = gradient(x)
        p_k = numpy.linalg.inv(hessian(x))*g
        x = x - p_k
        yield x

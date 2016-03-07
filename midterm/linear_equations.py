from __future__ import division
import random
import numpy
import solve

def linesearch(A, b, n_iters=1000):
    """Returns x such that Ax=b"""
    x = numpy.matrix([0.0 for _ in range(A.shape[1])]).transpose()
    for _ in range(n_iters):
        r_i = b - A*x
        if not r_i.any():
            return x
        t_i = (r_i.transpose()*r_i)/(r_i.transpose()*(A*r_i))
        x = x + t_i[0,0]*r_i
    return None

def newtons(A, b):
    """Returns x such that Ax=b"""
    grad = lambda x: b - A*x
    hessian = lambda x: -1*A
    x = numpy.matrix([0.0 for _ in range(A.shape[1])]).transpose()
    return solve.newtons(x, grad, hessian)

for _ in range(10):
    A = numpy.matrix([[1,0,0], [0,1,0], [0,0,1]])
    b = numpy.matrix([random.random() for _ in range(3)]).transpose()
    # x = linesearch(A,b)
    x = newtons(A, b)
    print x, numpy.linalg.solve(A, b)

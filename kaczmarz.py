import random
import numpy

def solve(A, b, epsilon=0.00001):
    """Returns x such that Ax=b"""
    x_prev = [-1 for _ in range(A.shape[1])]
    x = numpy.array([random.random() for _ in range(A.shape[1])])
    k = 0
    while numpy.linalg.norm(x_prev - x) >= epsilon:
        x_prev = x
        i = k % len(A)
        x = x + ((b[i] - numpy.dot(A[i], x))/numpy.linalg.norm(A[i])**2)*A[i]
        k += 1
    return x


for _ in range(10):
    A = numpy.array([[1,0,1], [0,1,1], [1,1,0]])
    b = numpy.array([random.random() for _ in range(3)])
    x = solve(A,b)
    print x, numpy.linalg.solve(A, b)

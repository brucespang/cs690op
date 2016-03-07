import random
import numpy
from common import *
import itertools
import matplotlib.pyplot as plt

def kaczmarz(A, b, next_set=lambda A,k,x: None):
    """Returns x such that Ax=b"""
    x = numpy.matrix([[0] for _ in range(A.shape[1])])
    i = 0
    while (b-A*x).any():
        i = next_set(A, i, x)
        n = A[i].T
        x = project_plane(n, b[i])(x)
        yield x

def solve(A, b, next_set, n_iters=2000):
    xs = itertools.islice(kaczmarz(A, b, next_set), n_iters)
    return list(xs)

def plot_convergence(A, b, max_iters=40):
    opt = numpy.linalg.solve(A, b)
    methods = {
        "cyclic": lambda A,k,x: (k+1) % len(A),
        "random": lambda A,k,x: random.choice(list(set(range(len(A))) - set([k]))),
        "distal": lambda A,k,x: max(range(len(A)),
                                    key=lambda i: plane_distance(A[i].T, b[i], x))
    }

    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("l2 distance from optimal")
    legends = []
    for t,next_set in methods.items():
        xs = itertools.islice(kaczmarz(A,b,next_set=next_set), max_iters)
        plt.plot([numpy.linalg.norm(x - opt) for x in xs])
        legends.append(t)
    plt.legend(legends)
    plt.axis(ymin=0,xmin=0)
    plt.savefig("kaczmarz.png")

A = numpy.matrix([[1,0,1], [0,1,1], [1,1,0]])
b = numpy.matrix([[random.random()] for _ in range(3)])
plot_convergence(A, b)

# plt.show()

# for _ in range(10):
#     A = numpy.matrix([[1,1,0], [0,1,0], [1,1,1]])
#     b = numpy.matrix([[random.random()] for _ in range(3)])
#     x_cyclic = solve(A,b,next_set=lambda A,k,x: k % len(A))
#     x_random = solve(A,b,next_set=lambda A,k,x: random.randint(0,len(A)-1))
#     x_distal = solve(A,b,next_set=lambda A,k,x: max(range(len(A)), key=lambda i: plane_distance(A[i].T, b[i], x)))
#     opt = numpy.linalg.solve(A, b)
#     def num_to_converge(xs):
#         return len([x for x in xs if numpy.linalg.norm(x - opt) > 0.001]), numpy.linalg.norm(xs[-1] - opt)
#     print ""
#     print "A", A
#     print "b", b
#     print "cyclic", num_to_converge(x_cyclic)
#     print "random", num_to_converge(x_random)
#     print "distal", num_to_converge(x_distal)

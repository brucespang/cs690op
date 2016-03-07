from __future__ import division
import random
import solve
import numpy
import matplotlib.pyplot as plt
import itertools
import scipy.optimize

def rosenbrock(x,y):
    return (1-x)**2 + 100*(y-x**2)**2

def g_rosenbrock(x):
    x,y = x[0,0],x[1,0]
    return numpy.matrix([-2*(1-x) - 400*x*(-1*(x**2) + y),
                        200*(-1*(x**2) + y)]).T

def h_rosenbrock(x):
    x,y = x[0,0],x[1,0]
    return numpy.matrix([[2+800*x**2-400*(-1*x**2+y), -400*x],
                         [-400*x, 200]])

def plot_newton():
    opt = numpy.array([1, 1**2])
    x0 = numpy.matrix([random.randint(-100000, 100000), random.randint(-100000, 100000)]).T
    print x0
    xs = list(solve.newtons(x0, g_rosenbrock, h_rosenbrock))

    plt.figure(dpi=300)
    plt.xlabel("Iteration")
    plt.ylabel("l2 distance from optimal")
    plt.plot([numpy.linalg.norm(x - opt) for x in xs])
    plt.axis(ymin=0,xmin=0)
    plt.savefig("rosenbrock-newtons.png")

def plot_linear(max_iters=1000):
    opt = numpy.array([1, 1**2])
    points = [[1,0],[2,0],[2,1]]
    stepsizes = [0.0001, 0.001, 0.0011, 0.0013]

    plt.figure(figsize=(8.5, 7),dpi=300)
    plt.xlabel("Iteration")
    plt.ylabel("l2 distance from optimal")
    for i,start in enumerate(points):
        plt.subplot(len(points), 1, i+1)
        x0 = numpy.matrix(start).T
        legends = []
        for a in stepsizes:
            xs = solve.linesearch(x0, lambda k: a, g_rosenbrock)
            xs = itertools.islice(xs, max_iters)
            plt.plot([numpy.linalg.norm(x - opt) for x in xs])
            legends.append("a=%0.4f"%(a))
            plt.title("Starting at (%d, %d)"%(start[0], start[1]))
        plt.legend(legends)
        plt.axis(ymin=0,xmin=0)
    plt.tight_layout()
    plt.savefig("rosenbrock-linear.png")

plot_newton()
plot_linear()
plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = Axes3D(fig)

# xs = range(-10,10)
# ys = range(-10,10)
# zs = [rosenbrock(x,y) for x in xs for y in ys]
# ax.plot_figure(xs,ys,zs)
# plt.show()

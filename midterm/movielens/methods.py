from __future__ import division
import sys
import numpy as np
from common import *
from math import ceil
import matplotlib.pyplot as plt

def global_mean(T):
    mu = np.sum(T)/np.count_nonzero(T)
    R = np.ma.masked_array(T, mask=T==0).filled(mu)
    return R

def user_mean(T):
    R = np.copy(T)
    for i in range(T.shape[1]):
        user = T[:,i].flatten()
        user = np.ma.masked_array(user, mask=user==0)
        mu = np.sum(user)/np.count_nonzero(user)
        R[:,i] = user.filled(mu)
    return R

def movie_mean(T):
    R = np.copy(T)
    for i in range(T.shape[0]):
        movie = T[i,:].flatten()
        movie = np.ma.masked_array(movie, mask=movie==0)
        mu = np.sum(movie)/np.count_nonzero(movie)
        R[i,:] = movie.filled(mu)
    return R

def mixture_mean(T):
    a = 0.452
    return a*user_mean(T) + (1-a)*movie_mean(T)

def shrink(Y, t):
    # This is slow
    u,s,v = np.linalg.svd(Y, full_matrices=False)
    return np.dot(u, np.dot(np.diag((s-t).clip(min=0)), v))

def svt(T, step_size=1.9, epsilon=0.001):
    n,d = T.shape[0], T.shape[1]
    # XXX: this is made up
    # t = 5*np.sqrt(n*d)
    t = 1
    Y = np.zeros((n,d))
    R = Y
    errors = []
    # while relative_error(R, T) >= epsilon:
    while len(errors) < 2 or errors[-1] > 0.9 or (errors[-2] - errors[-1]) >= epsilon:
        errors.append(relative_error(R, T))
        print >>sys.stderr, errors[-1]
        R = shrink(Y, t)
        Y = Y + step_size*matrix_project_nonzero(T, T - R).filled(0)
    errors.append(relative_error(R, T))
    return R, errors

def slow_factorization(T, a=0.001, epsilon=0.3):
    n,d = T.shape[0], T.shape[1]
    # XXX: this is made up
    k = int(ceil(min(n,d)/2))
    P = np.ones((n,k))
    Q = np.ones((d,k))
    rows,cols = T.nonzero()
    # print np.ravel(rows), np.ravel(cols)
    R = np.dot(P, Q.T)
    errors = []
    while relative_error(R, T) >= epsilon:
        for i,j in zip(np.ravel(rows), np.ravel(cols)):
            g = T[i,j] - np.dot(P[i,:], Q[j,:])
            P_n = P[i,:] + a*g*Q[j,:]
            Q_n = Q[j,:] + a*g*P[i,:]
            P[i,:] = P_n
            Q[j,:] = Q_n
        R = np.dot(P, Q.T)
        errors.append(relative_error(R, T))
        print errors[-1]
    return R, errors

def factorization(V, a=0.0000015, epsilon=0.000001):
    """
    Based on the gradient ascent method from:

      D. D. Lee and H. S. Seung, "Algorithms for non-negative matrix factorization,"
      Advances in Neural Information Processing Systems, vol. 13, 2001.

    Which is similar to the method in the assignment, but __much__ faster
    """
    n,d = V.shape[0], V.shape[1]
    # XXX: this is made up
    k = int(ceil(min(n,d)/2))
    W = np.ones((n,k))
    H = np.ones((k,d))
    R = np.dot(W, H)
    errors = []
    while len(errors) < 2 or (errors[-2] - errors[-1]) >= epsilon:
        H_n = H + a*(np.dot(W.T, (V - np.dot(W, H))))
        W_n = W + a*(np.dot(V - np.dot(W, H), H.T))
        H = H_n
        W = W_n
        R = np.dot(W, H)
        errors.append(relative_error(R, V))
        print >>sys.stderr, errors[-1]
    return R, errors

def run_baselines():
    algorithms = [global_mean, user_mean, movie_mean, mixture_mean]
    datasets = ['u1', 'u2', 'u3', 'u4', 'u5']
    # datasets = ['r1', 'r2', 'r3', 'r4', 'r5']
    dirname = "ml-1m/"
    dirname = "ml-100k/"
    for alg in algorithms:
        for dataset in datasets:
            train,test = read_user_data(dirname + dataset, "\t")
            R = alg(train)
            print alg.__name__, dataset, RMSE(R, test)

if __name__ == "__main__":
    # run_baselines()

    # graph algorithms that converge
    # datasets = ['u1', 'u2', 'u3', 'u4', 'u5']
    datasets = ['r1', 'r2', 'r3', 'r4', 'r5']
    dirname = "ml-1m/"
    algorithms = [svt]
    # algorithms = [svt, factorization]
    for alg in algorithms:
        for dataset in datasets:
            print >>sys.stderr, "starting", dataset
            train,test = read_user_data(dirname + dataset, " ")
            R, errors = alg(train)
            plt.figure()
            plt.xlabel("Iteration")
            plt.ylabel("relative error")
            plt.plot(errors)
            plt.axis(ymin=0,xmin=0)
            plt.savefig("convergence-%s-%s.png"%(alg.__name__, dataset))
            print alg.__name__, dataset, RMSE(R, test)
    plt.show()

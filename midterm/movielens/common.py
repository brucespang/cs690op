from __future__ import division
import numpy as np
import csv

def matrix_project_nonzero(M, X):
    return np.ma.masked_array(X, mask=M==0)

def RMSE(R,S):
    R = matrix_project_nonzero(S, R)
    return np.sqrt(np.sum(np.square(R-S))/np.count_nonzero(S))

def relative_error(R,T):
    return np.linalg.norm(matrix_project_nonzero(T, R-T))/np.linalg.norm(matrix_project_nonzero(T, T))

def read_user_data(path, delimeter="\t"):
    with open(path + ".base", 'r') as f:
        reader = csv.reader(f, delimiter=delimeter)
        train_data = {(int(item_id),int(user_id)):int(rating) for (user_id, item_id, rating, _) in reader}
    with open(path + ".test", 'r') as f:
        reader = csv.reader(f, delimiter=delimeter)
        test_data = {(int(item_id),int(user_id)):int(rating) for (user_id, item_id, rating, _) in reader}

    items = max(i for (i,_) in train_data.keys() + test_data.keys())
    users = max(j for (_,j) in train_data.keys() + test_data.keys())
    train = np.zeros((items+1, users+1))
    test = np.zeros((items+1, users+1))
    for (i,j),r in train_data.items():
        train[i,j] = r
    for (i,j),r in test_data.items():
        test[i,j] = r
    return train,test

if __name__ == "__main__":
    ones = np.matrix([[1]*3]*3)
    twos = np.matrix([[2]*3]*3)
    print matrix_project_nonzero(np.eye(3), ones)
    print matrix_project_nonzero(np.eye(3), twos)
    print RMSE(ones, np.eye(3))
    print RMSE(np.matrix([[2]*3]*3), np.eye(3))
    print read_user_data('ml-100k/u1.base')

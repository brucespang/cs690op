import numpy as np

def plane_orthogonal_component(n, b, x):
    return n*((b - x.T*n)/(np.linalg.norm(n)**2))

def plane_distance(n,b,x):
    return np.linalg.norm(plane_orthogonal_component(n,b,x))

def project_plane(n, b=0):
    def project(x):
        return x + plane_orthogonal_component(n,b,x)
    return project

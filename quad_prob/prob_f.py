import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
dim = 500
width = dim
interval_start = -dim


np.random.seed(1110)

b1 = np.random.rand(dim) * dim * 2 - dim
b2 = np.random.rand(dim) * dim * 2 - dim
b3 = np.random.rand(dim) * dim * 2 - dim
a1 = np.arange(0,dim,1) + 1
a2 = np.arange(0,dim,1) + 1
a3 = np.arange(0,dim,1) + 1

A = np.vstack((np.vstack((a1,a2)),a3))

a1[0]+=1e1
a2[0]+=1e1
np.random.shuffle(a1)
np.random.shuffle(a2)

alpha1 = np.ones(2)
alpha1[0] = np.min(a1)
alpha1[1] = np.min(a2)

alpha2 = np.ones(2)
alpha2[0] = np.max(a1)
alpha2[1] = np.max(a2)

print(b1,b2,a1,a2)
A1 = np.diag(a1)
A2 = np.diag(a2)
A3 = np.diag(a3)
def Val(x):
    f1 = 1 / 2 * np.dot(x,np.dot(A1,x)) + np.dot(b1,x)
    f2 = 1 / 2 * np.dot(x,np.dot(A2,x)) + np.dot(b2,x)
    f3 = 1 / 2 * np.dot(x,np.dot(A3,x)) + np.dot(b3,x)
    return np.array([f1, f2, f3])
def Gra(x):
    n = len(x)
    Gra_1 = np.dot(A1,x) + b1
    Gra_2 = np.dot(A2,x) + b2
    Gra_3 = np.dot(A3,x) + b3
    return np.vstack((np.vstack((Gra_1, Gra_2)),Gra_3))

def g(x):
    n = len(x)
    g1 = 1 / n * np.sum(np.abs(x))
    return np.array([g1, g1])


def N_val(H,g,d):
    g1 = np.dot(g[0],d) + 1/2 * np.dot(d, np.dot(H[0],d))
    g2 = np.dot(g[1], d) + 1 / 2 * np.dot(d, np.dot(H[1], d))
    return np.array([g1,g2])

def N_n_val(H,g,d):
    g1 = np.dot(g[0],d) + 1/2 * np.dot(d, np.dot(H[0],d))
    g2 = np.dot(g[1], d) + 1 / 2 * np.dot(d, np.dot(H[1], d))
    g3 = np.dot(g[0],d)
    g4 = np.dot(g[1], d)
    return np.array([g1,g2,g3,g4])

def N_gra(H,g,d):
    gra1 = g[0] + np.dot(H[0],d)
    gra2 = g[1] + np.dot(H[1],d)
    return np.vstack((gra1, gra2))
def Ifindomain(x):
    for i in range(len(x)):
        if -10000 < x[i] < 10000:
            A = True
        else:
            A = False
            break
    return A

# def PS(n):
#     return np.array([[0,0,0],[2,2,2]])
def PS(n):
    a = np.zeros(n)
    b = a + 2
    A = np.vstack((a,b))
    return A

F = "problem_f"
# x = np.random.rand(5)
# erro = 1e-4
# sigma = 0.1
#
# print(Hes(np.array([1,1,1,2,3,4])))
#
#
#
# list = AL.MGD(x, erro, sigma)
# print(len(list))
# print(list)
# point = []
# [point.append(Val(x)) for x in list]
# point = np.array(point)
# print(point)
#
#
#
# fig = plt.figure()
# plt.plot(point[:,0], point[:,1], c = 'b')
#
# plt.show()
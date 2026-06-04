import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
dim = 2
width = 4
interval_start = -2
def Val(x):
    n = len(x)
    f1 =  x[0] ** 2 + x[1] ** 2
    f2 =   100 * (x[0] - 50) ** 2 + 100 * (x[1]+50)**2
    return np.array([f1, f2])
def Gra(x):
    n = len(x)
    Gra_1 = x * np.array([2,2])
    Gra_2 = (x + np.array([-50,50])) * np.array([200,200])
    return np.vstack((Gra_1, Gra_2))

def g(x):
    n = len(x)
    g1 = 1 / n * np.sum(np.abs(x))
    return np.array([g1, g1])

def Inition():
    return np.array([200* np.random.rand() -100, 200* np.random.rand() -100])

def Hes_1(x):
    n = len(x)
    Hes = 2 * np.eye(n)
    return Hes

def Hes_2(x):
    n = len(x)
    Hes = 2 / n * np.eye(n)
    return Hes
def Hes(x):
    return np.array([Hes_1(x), Hes_2(x)])
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
    bound = []
    bound.extend([[-1000, 1000] for i in range(len(x))])
    for i in range(len(x)):
        if bound[i][0] < x[i] < bound[i][1]:
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

F = "JOS_ill2"
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
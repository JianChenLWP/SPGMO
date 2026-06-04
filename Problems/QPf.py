import numpy as np
from scipy.stats import ortho_group
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

from mpmath import mp, matrix

# 设置计算精度（例如：50 位小数）
# mp.dps = 34
np.random.seed(1110)


dim = 100
width = 2 * dim
interval_start = -dim

kappa = 1000

imb = 1000  #目标不平衡性




Q1 = ortho_group.rvs(dim=dim)
Q2 = ortho_group.rvs(dim=dim)






b1 = np.random.rand(dim) * dim * 2 - dim
b2 = np.random.rand(dim) * dim * 2 - dim

a1 = np.arange(0,dim,1) + 1
a2 = np.arange(0,dim,1) + 1

# a1 = np.random.rand(dim) + 1
# a2 = np.random.rand(dim) + 1

a1[0] = kappa
a1[1] = 1
a2[0] = kappa
a2[1] = 1

a1 = a1 * imb * 0.1

a2 = a2 * 0.1

L = np.array([np.max(a1),np.max(a2)])


np.random.shuffle(a1)
np.random.shuffle(a2)




A1 = np.dot(Q1.T,np.dot(np.diag(a1),Q1))
# A2 = np.dot(Q.T,np.dot(np.diag(a2),Q))

A2 = np.dot(Q2.T,np.dot(np.diag(a2),Q2))
# A2 = np.dot(Q.T,np.dot(np.diag(-a2),Q))


A1_inv = np.dot(Q1.T,np.dot(np.diag(1/a1),Q1))
A2_inv = np.dot(Q2.T,np.dot(np.diag(1/a2),Q2))

# A1_mp = matrix(A1)  # 将 A1 转换为 mpmath 的高精度矩阵
# A2_mp = matrix(A2)
# b1_mp = matrix(b1)
# b2_mp = matrix(b2)
#
# def Val(x):
#     """高精度计算目标函数值"""
#     x_mp = matrix(x)  # 将输入 x 转换为高精度
#     f1 = 0.5 * (x_mp.T * A1_mp * x_mp)[0] + (b1_mp.T * x_mp)[0]  # f1(x)
#     f2 = 0.5 * (x_mp.T * A2_mp * x_mp)[0] + (b2_mp.T * x_mp)[0]  # f2(x)
#     return np.array([f1, f2])
#
# def Gra(x):
#     """高精度计算梯度"""
#     x_mp = matrix(x)  # 转换为高精度
#     Gra_1 = A1_mp * x_mp + b1_mp  # f1 的梯度
#     Gra_2 = A2_mp * x_mp + b2_mp  # f2 的梯度
#     return np.array([list(Gra_1), list(Gra_2)])  # 返回高精度梯度

def Val(x):
    f1 = 1 / 2 * np.dot(x,np.dot(A1,x)) + np.dot(b1,x)
    f2 = 1 / 2 * np.dot(x,np.dot(A2,x)) + np.dot(b2,x)
    return np.array([f1, f2])
def Gra(x):
    n = len(x)
    Gra_1 = np.dot(A1,x) + b1
    Gra_2 = np.dot(A2,x) + b2
    return np.vstack((Gra_1, Gra_2))






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
        if -1e12 < x[i] < 1e12:
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

F = "QPf"
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
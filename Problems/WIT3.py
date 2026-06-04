import numpy as np
import copy
F = "WIT3"
dim = 2
width = 4
interval_start = -2
def Val(x):
    a = 0.9
    f1 = a * ((x[0] - 2) ** 2 + (x[1] - 2) ** 2) + (1-a) *((x[0] - 2) ** 4 + (x[1] - 2) ** 8)
    f2 = (x[0] + 2 * a) ** 2 + (x[1] + 2 * a) ** 2
    return np.array([f1, f2])
def Gra(x):
    a = 0.9
    Gra_1 = np.zeros(2)
    Gra_2 = np.zeros(2)
    Gra_1[0] = 2 * a * (x[0] - 2) + 4 * (1 - a) * (x[0] - 2) ** 3
    Gra_1[1] = 2 * a * (x[1] - 2) + 8 * (1 - a) * (x[1] - 2) ** 7
    Gra_2[0] = 2 * (x[0] + 2 * a)
    Gra_2[1] = 2 * (x[1] + 2 * a)
    return np.vstack((Gra_1, Gra_2))
def g(x):
    n = len(x)
    g1 = 1 / n * np.sum(np.abs(x))
    return np.array([g1, g1])
# print(Gra(0.4 * np.ones(2)))
# a = 0.4 *  np.ones(2)
# a[0] = 0.4 + 1e-7
# b = 0.4 * np.ones(2)
#
# print((Val(a)-Val(b))/1e-7)

def Hes_1(x):
    n = len(x)
    Hes = np.eye(n)
    Hes[0][0] = 1 + 6 * (x[0] -2) ** 2
    Hes[1][1] = 1 + 28 * (x[1] -2) ** 6
    return Hes

def Hes_2(x):
    n = len(x)
    Hes = 2 * np.eye(n)
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
    bound.extend([[-2, 2] for i in range(len(x))])
    for i in range(len(x)):
        if bound[i][0] < x[i] < bound[i][1]:
            A = True
        else:
            A = False
            break
    return A
# if Indomain(x):
#     print("T")
# else:
#     print("F")
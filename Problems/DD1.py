import numpy as np
import copy
F = "DD"
dim = 5
width =  40
interval_start = -20

def Val(x):
    f1 = np.sum(x**2)
    f2 = 3 * x[0] + 2 * x[1] - x[2] / 3 + 0.01 * (x[3] - x[4]) ** 3
    return np.array([f1, f2])
def Gra(x):
    Gra_1 = 2 * x
    Gra_2 = np.array([3,2,- 1/3, 0.03 * (x[3] - x[4]) ** 2, -0.03 * (x[3]-x[4]) ** 2])
    return np.vstack((Gra_1, Gra_2))


def g(x):
    n = len(x)
    g1 = 1 / n * np.sum(np.abs(x))
    return np.array([g1, g1])

def Inition():
    return 40 * np.random.rand(5) - 20
def Hes_1(x):
    n = len(x)
    Hes = 2 * np.eye(n)
    return Hes

def Hes_2(x):
    n = len(x)
    Hes = np.zeros((n, n))
    Hes[3][3] = 0.06 * (x[3] - x[4])
    Hes[4][4] = 0.06 * (x[3] - x[4])
    Hes[3][4] = - 0.06 * (x[3] - x[4])
    Hes[4][3] = - 0.06 * (x[3] - x[4])
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
# print(Gra(0.4 * np.ones(2)))
# a = 0.4 *  np.ones(2)
# a[1] = 0.4 + 1e-7
# b = 0.4 * np.ones(2)
# print((Val(a)-Val(b))/1e-7)

def Ifindomain(x):
    bound = []
    bound.extend([[-200, 200] for i in range(len(x))])
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
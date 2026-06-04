import numpy as np
import copy
dim = 2
width = 6
interval_start = -3
def Val(x):
    a = x[0] ** 2 + x[1] ** 2 + 1
    f1 = 1 / a
    f2 = x[0] ** 2 + 3 * x[1] ** 2 + 1
    return np.array([f1, f2])
def Gra(x):
    a = x[0] ** 2 + x[1] ** 2 + 1
    Gra_1 = np.zeros(2)
    Gra_2 = np.zeros(2)
    Gra_1[0] = - 2 * x[0] / a ** 2
    Gra_1[1] = - 2 * x[1] / a ** 2
    Gra_2[0] = 2 * x[0]
    Gra_2[1] = 6 * x[1]
    return np.vstack((Gra_1, Gra_2))

def g(x):
    n = len(x)
    g1 = 1 / n * np.sum(np.abs(x))
    return np.array([g1, g1])

def Hes_1(x):
    n = len(x)
    Hes = np.eye(n)
    Hes[0][0] = 12 * x[0] ** 2 - 2
    Hes[0][1] = -10
    Hes[1][0] = -10
    Hes[1][1] = 12 * x[1] ** 2 + 2
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
def RPS():
    A = np.array([[0,0],[1,1]])
    return A
def Bound(dim):
    bound = [[-2, 2], [-2, 2]]
    return np.array(bound)

def Backtobd(x, v):
    bound = [[-2, 2], [-2, 2]]
    d = []
    for i in range(len(x)):
        if v[i] > 0:
            d.append((bound[i][1] - x[i]) / v[i])
        elif v[i] < 0:
            d.append((bound[i][0] - x[i]) / v[i])
        else:
            d.append(1e5)
    m = min(d) - 1e-5
    return m

def Indomain(a):
    bound = [[-2, 2], [-2, 2]]
    for i in range(len(a)):
        if bound[i][0] <= a[i] <= bound[i][1]:
            pass
        elif a[i] > bound[i][1]:
            a[i] = bound[i][1]
        else:
            a[i] = bound[i][0]
            #print("n", end = " ")
    return np.array(a)

def Ifindomain(x):
    bound = [[-3, 3], [-3, 3]]
    for i in range(len(x)):
        if bound[i][0] < x[i] < bound[i][1]:
            A = True
        else:
            A = False
            break
    return A
def Inition():
    return np.array([4 * np.random.rand() -2, 4 * np.random.rand() -2])

def PS(n):
    x = []
    x1 = np.linspace(0, 1, 100)
    for i in x1:
        x_ = [i]
        for j in range(1, n):
            a = np.sin(6 * np.pi * i + ((j+1) * np.pi)/n)
            x_.append(a)
        x.append(x_)
    return np.array(x)

def proj(x):
    y = np.minimum(np.maximum(x,-2), 2)
    return y

F = "VU1"
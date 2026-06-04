import numpy as np
import copy
dim = 2
width = 1
interval_start = 0

def Val(x):
    a = np.pi / 180 * (45 + 40 * np.sin(2 * np.pi * x[0]) + 25 * np.sin(2 * np.pi * x[1]))
    b = 1 + 0.5 * np.cos(2 * np.pi * x[0])
    f1 = np.cos(a) * b
    f2 = np.sin(a) * b
    return np.array([f1, f2])
def Gra(x):
    a = np.pi / 180 * (45 + 40 * np.sin(2 * np.pi * x[0]) + 25 * np.sin(2 * np.pi * x[1]))
    b = 1 + 0.5 * np.cos(2 * np.pi * x[0])
    da1 =  4 * np.pi ** 2 / 9 * np.cos(2 * np.pi * x[0])
    da2 =  5 * np.pi ** 2 / 18 * np.cos(2 * np.pi * x[1])
    db1 = - np.pi * np.sin(2 * np.pi * x[0])
    Gra_1 = np.zeros(2)
    Gra_2 = np.zeros(2)
    Gra_1[0] = - np.sin(a) * da1 * b + np.cos(a) * db1
    Gra_1[1] = - np.sin(a) * da2 * b
    Gra_2[0] =  np.cos(a) * da1 * b + np.sin(a) * db1
    Gra_2[1] = np.cos(a) * da2 * b
    return np.vstack((Gra_1, Gra_2))

# x = np.array([0.3,0.1])
# e = 1e-6
# print(Gra(x), (Val(x+e)[1] - Val(x)[1])/e )


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
    bound = [[0, 1], [0, 1]]
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

F = "Hil1"
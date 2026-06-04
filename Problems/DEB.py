import numpy as np
import copy
F = "DEB"
dim = 2
width = 0.9
interval_start = 0.1
def Val(x):
    f1 = x[0]
    f2 = (2 - np.exp( - ((x[1] - 0.2) / 0.004) ** 2) - 0.8 * np.exp( - ((x[1] - 0.6) / 0.4) ** 2)) / (x[0] +1e-10)
    return np.array([f1, f2])
def Gra(x):
    Gra_1 = np.zeros(2)
    Gra_2 = np.zeros(2)
    Gra_1[0] = 1
    Gra_2[0] = - (2 - np.exp( - ((x[1] - 0.2) / 0.004) ** 2) - 0.8 * np.exp( - ((x[1] - 0.6) / 0.4) ** 2)) / (x[0] +1e-10) ** 2
    Gra_2[1] = (2 * (x[1] - 0.2) / 0.004 ** 2 * np.exp( - ((x[1] - 0.2) / 0.004) ** 2) + \
               1.6 * (x[1] - 0.6) / 0.4 ** 2 * np.exp( - ((x[1] - 0.6) / 0.4) ** 2)) / (x[0] +1e-10)
    return np.vstack((Gra_1, Gra_2))


def g(x):
    n = len(x)
    g1 = 1 / n * np.sum(np.abs(x))
    return np.array([g1, g1])

def Hes_1(x):
    n = len(x)
    Hes = np.zeros((2,2))
    return Hes

def Hes_2(x):
    n = len(x)
    Hes = np.eye(n)
    Hes[0][0] = 2 * (2 - np.exp( - ((x[1] - 0.2) / 0.004) ** 2) - 0.8 * np.exp( - ((x[1] - 0.6) / 0.4) ** 2)) / x[0] ** 3
    Hes[0][1] = - (2 * (x[1] - 0.2) / 0.004 ** 2 * np.exp( - ((x[1] - 0.2) / 0.004) ** 2) + \
               1.6 * (x[1] - 0.6) / 0.4 ** 2 * np.exp( - ((x[1] - 0.6) / 0.4) ** 2)) / x[0] ** 2
    Hes[1][0] = - (2 * (x[1] - 0.2) / 0.004 ** 2 * np.exp( - ((x[1] - 0.2) / 0.004) ** 2) + \
               1.6 * (x[1] - 0.6) / 0.4 ** 2 * np.exp( - ((x[1] - 0.6) / 0.4) ** 2)) / x[0] ** 2
    Hes[1][1] = (2 / 0.004 ** 2 *(1 - 2 / 0.004 ** 2 * (x[1] - 0.2) ** 2) * np.exp( - ((x[1] - 0.2) / 0.004) ** 2 +
               1.6 / 0.4 ** 2 * (1 - 2 / 0.4 ** 2 * (x[1] - 0.6) ** 2) * np.exp( - ((x[1] - 0.6) / 0.4) ** 2))) / x[0]
    return Hes
def Hes(x):
    return [Hes_1(x), Hes_2(x)]
# print(Hes(np.array([1,1])))
# print(Gra(0.4 * np.ones(2)))
# a = 0.4 *  np.ones(2)
# a[1] = 0.4 + 1e-7
# b = 0.4 * np.ones(2)
# print((Val(a)-Val(b))/1e-7)
def Bound(dim):
    bound = [[0.1, 1], [0, 1]]
    return np.array(bound)

def Backtobd(x, v):
    bound = [[0.1, 1], [0, 1]]
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
    bound = [[0.1, 1], [0, 1]]
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
    bound = [[0.1, 1], [0, 1]]
    for i in range(len(x)):
        if bound[i][0] < x[i] < bound[i][1]:
            A = True
        else:
            A = False
            break
    return A
def Inition():
    return np.array([0.9 * np.random.rand() + 0.1, np.random.rand()])

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
    y = np.minimum(np.maximum(x,0.1), 1)
# if Indomain(x):
#     print("T")
# else:
#     print("F")
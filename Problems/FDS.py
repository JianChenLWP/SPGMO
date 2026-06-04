import numpy as np
import copy
F = "FDS"
dim = 10
width = 4
interval_start = -2
def Val(x):
    n = len(x)
    f1 = 1 / n ** 2 * np.sum(np.array([(k + 1) * (x[k] - k - 1) ** 4 for k in range(n)]))
    f2 = np.exp(np.sum(np.array([x[k] / n for k in range(n)]))) + np.sum(np.array([x[k] ** 2 for k in range(n)]))
    f3 = 1 / (n * (n + 1)) * np.sum(np.array([(k + 1) * (n - k) * np.exp(-x[k]) for k in range(n)]))
    return np.array([f1, f2, f3])
def Gra(x):
    n = len(x)
    Gra_1 = np.zeros(n)
    Gra_2 = np.zeros(n)
    Gra_3 = np.zeros(n)
    for k in range(n):
        Gra_1[k] = 4 / n ** 2 * (k + 1) * (x[k] - k -1) ** 3
        Gra_2[k] = 1 / n * np.exp(np.sum(np.array([x[k] / n for k in range(n)]))) + 2 * x[k]
        Gra_3[k] = - 1 / (n * (n + 1)) * (k + 1) * (n - k) * np.exp(-x[k])
    return np.vstack((Gra_1, Gra_2, Gra_3))

def g(x):
    n = len(x)
    g1 = 1 / n * np.sum(np.abs(x))
    return np.array([g1, g1, g1])


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

def Inition():
    return 4* np.random.rand(10) -2
# print(Inition())
import numpy as np
import copy
F = "SD"

def Val(x):
    b1 = np.array([1, np.sqrt(2), np.sqrt(2), 1])
    f = np.ones(5)
    for i in range(5):
        t = (i+1) / 5
        f[i] = (x[0] + t*x[1] - np.exp(t)) ** 2 + (x[2] + np.sin(t) * x[3] - np.cos(t)) **2
    return f
print(Val(np.array([2,2,2,2])))
def Gra(x):
    Gra = np.ones((5,4))
    for i in range(5):
        t = (i + 1) / 5
        a = x[0] + t*x[1] - np.exp(t)
        b = x[2] + np.sin(t) * x[3] - np.cos(t)
        Gra[i] = np.array([2*a, 2*t*a, 2*b, 2 * np.sin(t)*b])
    return Gra

def Inition():
    b1 = np.array([-25, -5, -5, -1])
    b2 = -b1
    bound = np.vstack((b1, b2)).T
    a = np.random.rand(4)
    point = (bound[:,1] - bound[:,0]) * a + bound[:,0]
    return point

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
    b1 = np.array([-25,-5,-5,-1])
    b2 = -b1
    bound = np.vstack((b1,b2)).T
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
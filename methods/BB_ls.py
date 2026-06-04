import numpy as np
import copy
import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
import sys
sys.path.append('../quad_prob')
import PG_solver as co

def MGD(Func,x, erro,k_max):
    list = x
    A = True
    ite = 0
    t = 1.0
    K = 0
    stepsize = []
    alpha_min = 0.001
    alpha_max = 1000
    n = len(x)
    g = Func.g(x)
    lam = 1 / len(g) * np.ones(len(g))
    l = 1 / n * np.ones(len(g))  # 这是由于L_1项的系数为1/n。
    x_old = x + 2 * 1e-3 * np.random.rand(n) - 1e-3
    G_old = Func.Gra(x_old)
    y = copy.deepcopy(x - x_old)
    C = True

    x_old = x
    while A:
        # print(ite)
        yk = x
        x_old = copy.deepcopy(x)

        G = Func.Gra(x)

        G_d = G - G_old

        G_old = copy.deepcopy(G)
        # print("y",y)

        alpha = np.dot(G_d, y) / np.dot(y, y)
        # print(np.shape(G_d))
        # alpha_neg = np.linalg.norm(G_d,aixs=1) / np.linalg.norm(y)
        # print(alpha_neg)
        for i in range(len(alpha)):
            if alpha[i] < 0:
                alpha[i] = np.linalg.norm(G_d[i]) / np.linalg.norm(y)
            elif alpha[i] == 0:
                alpha[i] = 0.001
        alpha = np.maximum(alpha, alpha_min)
        alpha = np.minimum(alpha, alpha_max)


        JF1 = G
        fy1 = Func.Val(yk)
        Fx1 = (Func.Val(x) + Func.g(x))

        l1 = 1 / n * np.ones(len(g))


        k  = 0
        B = True
        while B:
            k += 1
            JF = JF1 / alpha.reshape((len(alpha), 1))
            fy = fy1 / alpha
            Fx = Fx1 / alpha
            l = l1 / alpha


            lam = co.cond_gra_simp_prox(JF, fy, Fx, yk, 1, l, lam)

            x = co.soft(np.dot(lam, l) / 1, yk - np.dot(JF.T, lam) / 1) + 1e-16 * np.random.rand()

            theta = np.max(np.dot(JF,x-yk) + Func.g(x) / alpha + fy - Fx) + 0.5 * 1 * np.dot(x-yk,x-yk)

            if 1 / np.sum(lam / alpha) * np.linalg.norm(x - yk) < erro:
            # if np.linalg.norm(x - x_old) < 1e-6:
                A = False
                B = False
                C = False

            value = Func.Val(x)+Func.g(x) - Fx * alpha
            if np.all(value <= theta * alpha + 1e-5 * np.abs(theta * alpha)): #计算机误差导致
                B =False
                # print("alpha",alpha)
            else:
                for i in range(len(l)):
                    if value[i] > theta * alpha[i] + 1e-5 * np.abs(theta * alpha[i]):
                        alpha[i] = 2 * alpha[i]
        # print("value", np.linalg.norm(x - yk))
        y = x - x_old + 1e-16 * np.random.rand()
        if C:
            list = np.vstack((list, x))
            K += k
        stepsize.append(1.0)
        ite +=1


        if ite >= k_max:
            A = False

    return list, K, stepsize

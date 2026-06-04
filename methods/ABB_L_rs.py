import numpy as np
import copy
import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
import sys
sys.path.append('../quad_prob')
import PG_solver as co

def MGD(Func,x, erro,k_max,restart):
    list = x
    A = True
    ite = 0
    t = 1.0
    K = 0
    stepsize = []
    n = len(x)
    g = Func.g(x)
    lam = 1 / len(g) * np.ones(len(g))
    lam1 = 1 / len(g) * np.ones(len(g))

    C = True

    x_old = x
    L = Func.L
    dist = []
    theta_k = 1
    theta_k_1 = 1
    while A:
        # print(ite)
        if ite % restart == 0:
            theta_k = 1
            theta_k_1 = 1

        gamma = theta_k * (1 - theta_k_1) / theta_k_1
        yk = x + gamma * (x - x_old)

        x_old = copy.deepcopy(x)

        theta_k_1 = theta_k

        theta_k = (np.sqrt(theta_k ** 4 + 4 * theta_k ** 2) - theta_k ** 2) / 2


        alpha = L

        JF = Func.Gra(yk) / alpha.reshape((len(alpha), 1))
        fy = Func.Val(yk) / alpha
        Fx = (Func.Val(x) + Func.g(x)) / alpha
        l = 1 / n * np.ones(len(g)) / alpha

        lam = co.cond_gra_simp_prox(JF, fy, Fx, yk, 1, l, lam)

        x = co.soft(np.dot(lam, l) / 1, yk - np.dot(JF.T, lam) / 1)

        # if 1 / np.sum(lam / L) * np.linalg.norm(x - yk) + 1e-20 * np.random.rand() < erro:
        #     A = False
        #     C = False
        # if C:
        #     list = np.vstack((list, x))
        #     dist.append(1 / np.sum(lam / alpha) * np.linalg.norm(x - yk))
        # stepsize.append(1.0)
        # ite +=1

        ##############################公平停机准则###################################################
        g = Func.g(x)
        G = Func.Gra(x)

        l1 = 1 / n * np.ones(len(g))

        lam1 = co.cond_gra_simp_prox1(G, g, x, l1, lam1)
        y1 = co.soft(np.dot(lam1, l1), x - np.dot(G.T, lam1))
        v1 = y1 - x
        if np.linalg.norm(v1) < erro:
            # if np.linalg.norm(v) < erro:
            A = False
            C = False
        if C:
            list = np.vstack((list, x))
            dist.append(np.linalg.norm(v1))
        stepsize.append(1.0)
        ite +=1
        #################################################################################


        if ite >= k_max:
            A = False

    return list, K, stepsize, dist






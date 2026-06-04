import numpy as np
import copy
import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
import sys
sys.path.append('../quad_prob')
import PG_solver as co

def MGD(Func,x,erro,k_max):
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
    while A:
        # print(ite)
        yk = x

        x_old = copy.deepcopy(x)


        alpha = L

        JF = Func.Gra(yk) / alpha.reshape((len(alpha), 1))
        fy = Func.Val(yk) / alpha
        Fx = (Func.Val(x) + Func.g(x)) / alpha
        l = 1 / n * np.ones(len(g)) / alpha

        lam = co.cond_gra_simp_prox(JF, fy, Fx, yk, 1, l, lam)


        x = co.soft(np.dot(lam, l) / 1, yk - np.dot(JF.T, lam) / 1)

        if 1 / np.sum(lam / L) * np.linalg.norm(x - yk) < 1 * erro or np.linalg.norm(
                x - yk) < 1 * 1e-7:  # 防止proximal在大的光滑参数不稳定
            g = Func.g(x)
            G = Func.Gra(x)
            l1 = 1 / n * np.ones(len(g))

            lam1 = co.cond_gra_simp_prox1(G, g, x, l1, lam / L / np.sum(lam / L))
            # print("lam1", lam1)
            y1 = co.soft(np.dot(lam1, l1), x - np.dot(G.T, lam1))
            v1 = y1 - x
            if np.linalg.norm(v1) < erro:
                A = False
                C = False
        if C:
            dist.append(1 / np.sum(lam / L) * np.linalg.norm(x - yk))
            list = np.vstack((list, x))
        stepsize.append(1.0)
        ite +=1
# ###############################公平停机准则###################################################
#         g = Func.g(x)
#         G = Func.Gra(x)
#
#         l1 = 1 / n * np.ones(len(g))
#
#         lam1 = co.cond_gra_simp_prox1(G, g, x, l1, lam1)
#         y1 = co.soft(np.dot(lam1, l1), x - np.dot(G.T, lam1))
#         v1 = y1 - x
#         if np.linalg.norm(v1) < erro:
#             # if np.linalg.norm(v) < erro:
#             A = False
#             C = False
#         if C:
#             dist.append( np.linalg.norm(v1))
#             list = np.vstack((list, x))
#         stepsize.append(1.0)
#         ite +=1
# #################################################################################




        if ite >= k_max:
            A = False

    return list, K, stepsize, dist










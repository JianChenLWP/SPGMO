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
    list_a = []
    lam_old = 0.5 *  np.ones(2)
    eta = 0.85
    alpha_min = 0.001
    alpha_max = 1000
    n = len(x)
    g = Func.g(x)
    lam = 1 / len(g) * np.ones(len(g))
    lam1 = 1 / len(g) * np.ones(len(g))

    l1 = 1 / n * np.ones(len(g))  #这是由于L_1项的系数为1/n。
    x_old = x + 2 * 1e-3 *np.random.rand(n) - 1e-3
    G_old = Func.Gra(x_old)
    y = x - x_old
    while A:
        # print(x)
        # print("sum",np.sum(Func.Val(x)))
        # t = np.maximum(1, t)2
        B = True
        # C = True
        # D = True
        ite += 1
        # print(ite)
        # print("list: {}".format(list))
        F = Func.Val(x)
        G = Func.Gra(x)
        gra = copy.deepcopy(G)
        g = Func.g(x)


        if ite >= 1:
            G_d = G - G_old
            alpha = np.dot(G_d, y) / np.dot(y,y)
            # print(np.shape(G_d))
            # alpha_neg = np.linalg.norm(G_d,aixs=1) / np.linalg.norm(y)
            # print(alpha_neg)
            for i in range(len(alpha)):
                if alpha[i] < 0:
                    alpha[i] = np.linalg.norm(G_d[i]) / np.linalg.norm(y)
                elif alpha[i] == 0:
                    alpha[i] = 0.001
            alpha = np.maximum(alpha,alpha_min)
            alpha = np.minimum(alpha, alpha_max)
            G_s = G / alpha.reshape((len(alpha),1))
            g_s = g / alpha
            l = 1 / n * np.ones(len(g)) / alpha


        if ite > 1:
            lam_old = lam


        # print(G,g,x,l)
        lam = co.cond_gra_simp_prox1(G_s, g_s, x, l, lam)
        y = co.soft(np.dot(lam,l), x - np.dot(G_s.T, lam))
        # y = co.soft(np.dot(lam, l), x - np.dot(G.T, lam))
        v = y - x

        # t_x = np.max(np.dot(G, v) + Func.g(y) - g) + 0.5 * np.dot(v, v)
        psi_x = np.dot(G,v) + (Func.g(y) - g)

        # print("value",t_x)
        # print("value", np.linalg.norm(v))


        if 1 / np.sum(lam / alpha) * np.linalg.norm(x - y) < 1 * erro or np.linalg.norm(
                x - y) < 1 * 1e-7:  # 防止proximal在大的光滑参数不稳定
            g = Func.g(x)
            G = Func.Gra(x)
            l1 = 1 / n * np.ones(len(g))

            lam1 = co.cond_gra_simp_prox1(G, g, x, l1, lam / alpha / np.sum(lam / alpha))
            # print("lam1", lam1)
            y1 = co.soft(np.dot(lam1, l1), x - np.dot(G.T, lam1))
            v1 = y1 - x
            if np.linalg.norm(v1) < erro:
                A = False
                B = False

        k = 0
        while B:

            if Func.Ifindomain(x + t * v):
                if k > 20 or t < 1e-6:

                    # print(k,t)

                    B = False
                    A = False
                    # print("x:",x)
                    # print("end3",t)
                    # print(list_a)
                    # print(list_a)
                    # print("end3")
                    # print(x[0:5])
                elif np.all((Func.Val(x + t * v) + Func.g(x + t * v)) - (F + g) < t * 1e-4 * psi_x):

                    k += 1

                    if ite >= 1:
                        stepsize.append(t)
                    # print(stepsize)
                    x = x + t * v + 1e-16 * np.random.rand()


                    list = np.vstack((list, x))
                    K += k

                    G_old = gra
                    y = t * v


                    B = False
                    if ite >= k_max:
                        A = False

                    t = 1
                else:
                    t = 0.5 * t
                    k += 1
            else:
                t = 0.5 * t
                # print(x)
                # print(x + t * v)
    return list, K, stepsize

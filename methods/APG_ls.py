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
    n = len(x)
    g = Func.g(x)
    lam = 1 / len(g) * np.ones(len(g))
    lam1 = 1 / len(g) * np.ones(len(g))


    l = 1 / n * np.ones(len(g))  #这是由于L_1项的系数为1/n。
    x_old = x
    L = 1
    C = True
    theta_k = 1
    theta_k_1 = 1
    while A:
        # print(ite)
        ite += 1

        gamma = theta_k * (1 - theta_k_1) / theta_k_1
        yk = x + gamma * (x - x_old)

        x_old = copy.deepcopy(x)

        theta_k_1 = theta_k

        theta_k = (np.sqrt(theta_k ** 4 + 4 * theta_k ** 2) - theta_k ** 2) / 2


        fy = Func.Val(yk)
        Fx = Func.Val(x) + Func.g(x)

        JF = Func.Gra(yk)



        k = 0
        B = True
        while B:
            k += 1
            lam = co.cond_gra_simp_prox(JF, fy, Fx, yk, L, l, lam)


            x = co.soft(np.dot(lam, l) / L, yk - np.dot(JF.T, lam) / L) + 1e-16 * np.random.rand()


            theta = np.max(np.dot(JF,x-yk) + Func.g(x) + fy - Fx) + 0.5 * L * np.dot(x-yk,x-yk)

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
                    B = False
                    C = False

            if np.max(Func.Val(x)+Func.g(x)-Fx) <= theta + 1e-5 * np.abs(theta):
                B =False
            else:
                L = 2 * L
        L = 0.5 * L
        # print("L",L)
        if C:
            list = np.vstack((list, x))
            K += k
        # print("value",np.linalg.norm(x - yk))
        stepsize.append(1.0)




        if ite >= k_max:
            A = False

    return list, K, stepsize


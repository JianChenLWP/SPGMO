import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
import os
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist

# 使用相对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(parent_dir, 'quad_prob'))
import PG_solver as co

def MGD(Func,x, erro,k_max,restart):
    list = x
    A = True
    ite = 0
    rs = 0
    t = 1.0
    K = 0
    stepsize = []
    list_a = []
    lam_old = 0.5 * np.ones(2)
    eta = 0.85
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
    dist = []

    G = Func.Gra(x)

    G_d = G - G_old

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
    alpha_old = alpha
    theta_k = 1
    theta_k_1 = 1
    while A:

        if ite % restart == 0:
            theta_k = 1
            theta_k_1 = 1

        gamma = theta_k * (1 - theta_k_1) / theta_k_1
        yk = x + gamma * (x - x_old)

        x_old = copy.deepcopy(x)

        theta_k_1 = theta_k

        theta_k = (np.sqrt(theta_k ** 4 + 4 * theta_k ** 2) - theta_k ** 2) / 2




        JF1 = Func.Gra(yk)
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

            theta = np.dot(JF1,x-yk) + Func.g(x) + fy1 - Fx1 + 0.5 * 1 * np.dot(x-yk,x-yk) * alpha

            L = 1 / np.sum(lam / alpha)
            if np.max(np.dot(JF1,x-yk)) >= 0:
                B = False
                C = False
                ls = 1
                print("restart")
                alpha = alpha_old


            if L * np.linalg.norm(x - yk) < erro:
                A = False
                B = False
                C = False

            value = Func.Val(x) + Func.g(x) - Fx1
            if np.all(value <= theta + 1e-10 * np.abs(theta)): #计算机误差导致
                B =False
                print("alpha",alpha,value,theta)
            else:
                for i in range(len(l)):
                    if value[i] > theta[i] + 1e-10 * np.abs(theta[i]):
                        alpha[i] = 2 * alpha[i]
        # print("value", np.linalg.norm(x - yk))
        if C:
            alpha = alpha / 1.8
            alpha = np.maximum(alpha, alpha_min)
            list = np.vstack((list, x))
            dist.append(1 / np.sum(lam / L) * np.linalg.norm(x - yk))
            K += k
            ite += 1
            rs += 1
        stepsize.append(1.0)



        if ite >= k_max:
            A = False

    return list, K, stepsize, dist

import numpy as np
import copy
import matplotlib.pyplot as plt
import time
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
import sys
sys.path.append('../quad_prob')
sys.path.append('../problems')
import PG_solver as co
import DEB as Func

def MGD(Func,x, erro,k_max):
    list = x
    A = True
    ite = 0
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
    theta_k = 1
    theta_k_1 = 1
    while A:
        # print(ite)
        gamma = theta_k * (1 - theta_k_1) / theta_k_1

        theta_k_1 = theta_k

        theta_k = (np.sqrt(theta_k ** 4 + 4 * theta_k ** 2) - theta_k ** 2) / 2

        yk = x + gamma * (x - x_old)

        x_old = copy.deepcopy(x)


        G = Func.Gra(x)

        if ite < 1:
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
        else:
            alpha = alpha / 1.8
            alpha = np.maximum(alpha, alpha_min)

        JF1 = Func.Gra(yk)
        fy1 = Func.Val(yk)
        Fx1 = (Func.Val(x) + Func.g(x))

        l1 = 1 / n * np.ones(len(g))

        k = 0
        B = True
        while B:
            k += 1
            JF = JF1 / alpha.reshape((len(alpha), 1))
            fy = fy1 / alpha
            Fx = Fx1 / alpha
            l = l1 / alpha

            lam = co.cond_gra_simp_prox(JF, fy, Fx, yk, 1, l, lam)

            x = co.soft(np.dot(lam, l) / 1, yk - np.dot(JF.T, lam) / 1)

            # ##############################公平停机准则###################################################
            # g = Func.g(x)
            # G = Func.Gra(x)
            #
            # lam1 = co.cond_gra_simp_prox1(G, g, x, l1, lam1)
            # print("lam1", lam1)
            # y1 = co.soft(np.dot(lam1, l1), x - np.dot(G.T, lam1))
            # v1 = y1 - x
            # # if np.linalg.norm(v1) < 1e-14:
            # #     # if np.linalg.norm(v) < erro:
            # #     A = False
            # #     B = False
            # #     C = False
            # #################################################################################

            theta = np.dot(JF, x - yk) + Func.g(x) / alpha + fy - Fx + 0.5 * 1 * np.dot(x - yk, x - yk)

            if 1 / np.sum(lam / alpha) * np.linalg.norm(x - yk) < 1 * erro or np.linalg.norm(
                    x - yk) < 1 * 1e-7:  # 防止proximal在大的光滑参数不稳定
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
                    C = False

                # A = False
                # B = False
                # C = False

            value = Func.Val(x) + Func.g(x) - Fx * alpha
            if np.all(value <= theta * alpha + 1e-5 * np.abs(theta * alpha)):  # 计算机误差导致
                B = False
                # print("alpha", alpha)
            else:
                for i in range(len(l)):
                    if value[i] > theta[i] * alpha[i] + 1e-5 * np.abs(theta[i] * alpha[i]):
                        alpha[i] = 2 * alpha[i]
        # print("value", np.linalg.norm(x - yk), 1 / np.sum(lam / alpha))
        y = x - x_old + 1e-20
        if C:
            list = np.vstack((list, x))
            K += k
        stepsize.append(1.0)
        ite += 1

        if ite >= k_max:
            A = False

    return list, K, stepsize

# start = time.process_time()
# num = 200
# k_max = 500
#
# dim = Func.dim
# width = Func.width
# interval_start = Func.interval_start
# np.random.seed(1110)
# x = width * np.random.rand(num, dim) + interval_start
#
# # x = np.array([[-2.01,-2.01],[-2.01,-2.01]])
# # x[:][1] = 1
# erro = 1e-6
# sigma = 1e-4
# L = 0
# K = 0
# S = 0
# endpoint = []
#
#
# for i in range(num):
#     print("round: {}".format(i + 1))
#     list, k, s = MGD(Func,x[i], erro,k_max)
#     # print(list)
#     l = len(list)
#     L += l - 1
#     K += k
#     S += sum(s)
#     # print("stepsize:".format(s))
#     if len(np.shape(list)) == 1:
#         endpoint.append(list)
#     else:
#         endpoint.append(list[-1])
#     # print(list[-1])
#
# end = time.process_time()
# print('average Running time: %s Seconds' % ((end - start) / num))
# print('average iteration: {}'.format(L / num))
# print('average iteration function evaluation: {}'.format(K / num))
# print('average stepsize: {}'.format(S / (L)))
# # print('average stepsize: {}'.format(S / L))
# endpoint = np.array(endpoint)
#
# # print(endpoint)
# if len(Func.Val(x[0])) != 2:
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.set_xlabel('$x_1$')
#     ax.set_ylabel('$x_2$')
#     ax.set_zlabel('$x_3$')
#     # val = np.array([Func.Val(x) for x in endpoint])
#     ax.scatter(endpoint[:,0], endpoint[:,1], endpoint[:,2], s = 5, c = 'b')
#     plt.title("Variable Space")
#     # plt.savefig("FDS_BBMOx.png", dpi=500, bbox_inches='tight')
#     plt.show()
# else:
#     fig = plt.figure()
#     plt.xlabel('$x_1$')
#     plt.ylabel("$x_2$")
#     plt.scatter(endpoint[:,0], endpoint[:,1], s = 5, c = 'b')
#     # RPS = Func.RPS()
#     # plt.plot(RPS[:, 0], RPS[:, 1], c='k')
#     plt.title('{}, num: {}, dim: {}, interval: {}, \n ite: {}, f_eva: {}, time: {}s'.format(Func.F, num, dim,
#                                                                       [interval_start, interval_start + width], L / num,
#                                                              K / num, (end - start) / num))
#     plt.show()
#
#
# if len(Func.Val(x[0])) != 2:
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     val = np.array([Func.Val(x) + Func.g(x) for x in endpoint])
#     ax.scatter(val[:, 1], val[:, 0], val[:, 2], s=5, c='b')
#     # plt.savefig("FDS_OBBMO.png", dpi=500, bbox_inches='tight')
#     plt.title("Value Space")
#     # plt.savefig("FDS_SDMOy.png", dpi=500, bbox_inches='tight')
#     # plt.savefig("FDS_BBMO.png", dpi=500, bbox_inches='tight')
#     plt.show()
# else:
#     fig = plt.figure()
#     val = np.array([Func.Val(x) + Func.g(x) for x in endpoint])
#     plt.xlabel('$f_1$')
#     plt.ylabel("$f_2$")
#     plt.scatter(val[:,0], val[:,1], s = 5, c = 'r')
#     plt.title("Value Space")
#     # plt.savefig("PNR_BBMO.png",dpi=500,bbox_inches = 'tight')
#     plt.show()
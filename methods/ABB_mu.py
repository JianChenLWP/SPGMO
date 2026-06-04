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
import QPh as Func

def MGD(Func,x, erro,k_max):
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
    mu = np.sqrt(1 / Func.kappa)
    dist = []
    while A:
        # print(ite)
        gamma = (1 - mu) / (1 + mu)
        yk = x + gamma * (x-x_old)

        x_old = copy.deepcopy(x)


        alpha = L

        JF = Func.Gra(yk) / alpha.reshape((len(alpha), 1))
        fy = Func.Val(yk) / alpha
        Fx = (Func.Val(x) + Func.g(x)) / alpha
        l = 1 / n * np.ones(len(g)) / alpha

        lam = co.cond_gra_simp_prox(JF, fy, Fx, yk, 1, l, lam)

        x = co.soft(np.dot(lam, l) / 1, yk - np.dot(JF.T, lam) / 1)

        # if 1 / np.sum(lam / alpha) * np.linalg.norm(x - yk) < 1 * erro or np.linalg.norm(
        #         x - yk) < 1 * 1e-7:  # 防止proximal在大的光滑参数不稳定
        #     g = Func.g(x)
        #     G = Func.Gra(x)
        #     l1 = 1 / n * np.ones(len(g))
        #
        #     lam1 = co.cond_gra_simp_prox1(G, g, x, l1, lam / alpha / np.sum(lam / alpha))
        #     # print("lam1", lam1)
        #     y1 = co.soft(np.dot(lam1, l1), x - np.dot(G.T, lam1))
        #     v1 = y1 - x
        #     if np.linalg.norm(v1) < erro:
        #         A = False
        #         C = False
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


        # print("value", ite, 1 / np.sum(lam / alpha) * np.linalg.norm(x - yk))


        if ite >= k_max:
            A = False

    return list, K, stepsize,dist
# start = time.process_time()
# num = 100
# k_max = 3000
#
# dim = Func.dim
# width = Func.width
# interval_start = Func.interval_start
# np.random.seed(1110)
# x = width * np.random.rand(num, dim) + interval_start
#
# # x = np.array([[-2.01,-2.01],[-2.01,-2.01]])
# # x[:][1] = 1
# erro = 1e-4
# sigma = 1e-4
# L = 0
# K = 0
# S = 0
# endpoint = []
#
#
# for i in range(num):
#     print("round: {}".format(i + 1))
#     list, k, s,dist = MGD(Func,x[i], erro,k_max)
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
#
# #####################存储#################################
# # np.savetxt('C:\PycharmProjects\pythonProjectRL\Gradient Methods\DATA\WIT\WIT6_BBPGMO', endpoint)
#
# # np.savetxt('C:\PycharmProjects\pythonProjectRL\Gradient Methods\DATA\JOS1\JOS1a_BBPGMO', endpoint)
#
# # np.savetxt('C:\PycharmProjects\pythonProjectRL\Gradient Methods\DATA\DEB\FDS_BBPGMO', endpoint)
#
# # np.savetxt('C:\PycharmProjects\pythonProjectRL\Gradient Methods\DATA\Hil1\BK1_BBPGMO', endpoint)
# ##################################################
#
#
#
#









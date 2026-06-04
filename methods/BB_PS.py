import numpy as np
import copy
import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
import sys
sys.path.append('../quad_prob')
sys.path.append('../Problems')
import PS_solver as co
import FF100b as Func

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
    x_old = x + 2 * 1e-3 * np.random.rand(n) - 1e-3
    G_old = Func.Gra(x_old)
    lam = 1 / len(G_old) * np.ones(len(G_old))

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
        g = np.zeros(2)


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
        y = co.proj_unit_simplex_median(x - np.dot(G_s.T, lam))
        # y = co.soft(np.dot(lam, l), x - np.dot(G.T, lam))
        v = y - x

        # t_x = np.max(np.dot(G, v) + Func.g(y) - g) + 0.5 * np.dot(v, v)
        psi_x = np.dot(G,v)

        # print("value",t_x)
        # print("value", np.linalg.norm(v))
        print("ite", ite, "v", 1 / np.sum(lam / alpha) * np.linalg.norm(x - y), 1 / np.sum(lam / alpha))


        if 1 / np.sum(lam / alpha) * np.linalg.norm(x - y) < 1 * erro or np.linalg.norm(
                x - y) < 1 * 1e-7:  # 防止proximal在大的光滑参数不稳定
            g = np.zeros(2)
            G = Func.Gra(x)
            l1 = 1 / n * np.ones(len(g))

            lam1 = co.cond_gra_simp_prox1(G, g, x, l1, lam / alpha / np.sum(lam / alpha))
            # print("lam1", lam1)
            y1 = co.proj_unit_simplex_median(x - np.dot(G.T, lam1))
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
                elif np.all(Func.Val(x + t * v)  - F < t * 1e-4 * psi_x):

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

# num = 200
# erro = 1e-5
# k_max = 2000
# dim = Func.dim
# width = Func.width
# interval_start = Func.interval_start
# np.random.seed(1110)
# X = width * np.random.rand(num, dim) + interval_start
# X = X / np.sum(X, axis=1).reshape((len(X), 1))    #simplex feasible
#
# iter = []
# feval = []
# time = []
# endpoint = []
#
# for i in range(num):
#     start = tm.perf_counter()
#     list, k, s = MGD(Func,X[i], erro,k_max)
#     end = tm.perf_counter()
#     print("round", i,len(list)-1,round(1000 * (end - start), 2))
#
#     if len(np.shape(list)) == 1:
#                 endpoint.append(list)
#     else:
#         endpoint.append(list[-1])
#
#     iter.append(len(list)-1) # save iteration
#     feval.append(k) # save function evaluation
#     time.append(end - start) # save time
# print('iter: {}'.format(sum(iter) / num),
#         'feval: {}'.format(sum(feval) / num),
#         'time: {}'.format(round(1000 * (sum(time) / num), 2)))
#
# endpoint = np.array(endpoint)
#
# # print(endpoint)
# if len(Func.Val(X[0])) != 2:
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
#                                                                       [interval_start, interval_start + width], sum(iter) / num,
#                                                                                             sum(feval) / num , round(1000 * (sum(time) / num), 2)))
# # plt.savefig('C:\PycharmProjects\pythonProjectRL\Gradient Methods\PHOTO\JOS1\JOS1d_s_wo.png', dpi=500, bbox_inches='tight')
#
# # fig = plt.figure()
# # plt.xlabel('$x_1$')
# # plt.ylabel("$x_2$")
# # plt.scatter(endpoint[:,0], endpoint[:,1], s = 5, c = 'b')
# #   plt.savefig("JOS1d_s_wo.png",dpi=500,bbox_inches = 'tight')
#     plt.show()
#
# fig = plt.figure()
# val = np.array([Func.Val(x) for x in endpoint])
# plt.xlabel('$f_1$')
# plt.ylabel("$f_2$")
# plt.scatter(val[:,0], val[:,1], s = 5, c = 'r')
# plt.title("Value Space")
#     # plt.savefig("PNR_BBMO.png",dpi=500,bbox_inches = 'tight')
# plt.show()
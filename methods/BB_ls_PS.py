import numpy as np
import copy
import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
import sys
sys.path.append('../quad_prob')
import PS_solver as co

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
    x_old = x + 2 * 1e-3 * np.random.rand(n) - 1e-3
    G_old = Func.Gra(x_old)
    lam = 1 / len(G_old) * np.ones(len(G_old))
    y = copy.deepcopy(x - x_old)
    C = True

    x_old = x
    theta_k = 1
    theta_k_1 = 1
    while A:

        yk = x



        G = Func.Gra(x)
        g = np.ones(2)

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
        Fx1 = Func.Val(x)

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

            x = co.proj_unit_simplex_median(yk - np.dot(JF.T, lam) / 1)

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

            theta = np.dot(JF, x - yk) + fy - Fx + 0.5 * 1 * np.dot(x - yk, x - yk)

            # print("ite", ite, "v", 1 / np.sum(lam / alpha) * np.linalg.norm(x - yk), 1 / np.sum(lam / alpha))

            if 1 / np.sum(lam / alpha) * np.linalg.norm(x - yk) < 1 * erro or np.linalg.norm(
                    x - yk) < 1 * 1e-7:  # 防止proximal在大的光滑参数不稳定
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
                    C = False

                # A = False
                # B = False
                # C = False

            value = Func.Val(x) - Fx * alpha
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

# num = 200
# erro = 1e-6
# k_max = 2000
# dim = Func.dim
# width = Func.width
# interval_start = Func.interval_start
# np.random.seed(1110)
# X = width * np.random.rand(num, dim) + interval_start
# X = X / np.sum(X, axis=1).reshape((len(X), 1))    #simplex feasible
#
# # X = np.random.dirichlet(alpha=np.ones(dim), size=num)
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
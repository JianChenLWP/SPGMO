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
    lam_old = 0.5 *  np.ones(2)
    eta = 0.85
    n = len(x)
    g = np.ones(2)
    lam = 1 / len(g) * np.ones(len(g))


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
        Fx = Func.Val(x)

        JF = Func.Gra(yk)



        k = 0
        B = True
        while B:
            k += 1
            lam = co.cond_gra_simp_prox(JF, fy, Fx, yk, L, l, lam)


            x = co.proj_unit_simplex_median(yk - np.dot(JF.T, lam) / L) + 1e-16 * np.random.rand()


            theta = np.max(np.dot(JF,x-yk) + fy - Fx) + 0.5 * L * np.dot(x-yk,x-yk)

            # print("ite", ite, "v", 1 / np.sum(lam / L) * np.linalg.norm(x - yk), 1 / np.sum(lam / L))

            if 1 / np.sum(lam / L) * np.linalg.norm(x - yk) < 1 * erro or np.linalg.norm(
                    x - yk) < 1 * 1e-7:  # 防止proximal在大的光滑参数不稳定
                g = np.zeros(2)
                G = Func.Gra(x)
                l1 = 1 / n * np.ones(len(g))

                lam1 = co.cond_gra_simp_prox1(G, g, x, l1, lam / L / np.sum(lam / L))
                # print("lam1", lam1)
                y1 = co.proj_unit_simplex_median(x - np.dot(G.T, lam1))
                v1 = y1 - x
                if np.linalg.norm(v1) < erro:
                    A = False
                    B = False
                    C = False


            if np.max(Func.Val(x)-Fx) <= theta + 1e-5 * np.abs(theta):
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

# num = 20
# erro = 1e-6
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
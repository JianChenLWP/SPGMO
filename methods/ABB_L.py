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


def MGD(Func, x, erro, k_max):
    """
    实现加速近端梯度方法 (Accelerated Proximal Gradient Method)
    
    参数:
    ------
    Func : object
        问题函数对象，需要包含以下方法：
        - g(x): 返回非光滑部分
        - Gra(x): 返回梯度
        - Val(x): 返回函数值
        - L: Lipschitz常数
    x : np.ndarray
        初始点
    erro : float
        收敛容差
    k_max : int
        最大迭代次数
    
    返回:
    -------
    trajectory : np.ndarray
        迭代轨迹
    K : int
        函数评估次数（当前实现中为0，保留接口）
    stepsize : list
        步长记录
    dist : list
        距离记录
    """
    trajectory = x.reshape(1, -1)  # 使用更具描述性的变量名
    continue_iteration = True
    iteration = 0
    K = 0  # 函数评估次数（当前未使用）
    stepsize = []
    n = len(x)
    
    # 获取初始非光滑部分
    g_val = Func.g(x)
    
    # 初始化拉格朗日乘子
    lam = np.ones(len(g_val)) / len(g_val)
    lam1 = np.ones(len(g_val)) / len(g_val)
    
    save_trajectory = True
    x_old = x.copy()
    L = Func.L
    dist = []
    theta_k = 1.0
    theta_k_1 = 1.0
    
    while continue_iteration:
        # Nesterov加速步骤
        gamma = theta_k * (1 - theta_k_1) / theta_k_1
        yk = x + gamma * (x - x_old)
        
        # 更新theta参数
        theta_k_1 = theta_k
        theta_k = (np.sqrt(theta_k ** 4 + 4 * theta_k ** 2) - theta_k ** 2) / 2
        
        x_old = x.copy()
        alpha = L
        
        # 计算梯度和函数值
        JF = Func.Gra(yk) / alpha.reshape((len(alpha), 1))
        fy = Func.Val(yk) / alpha
        Fx = (Func.Val(x) + Func.g(x)) / alpha
        l = np.ones(len(g_val)) / n / alpha
        
        # 求解条件梯度问题
        lam = co.cond_gra_simp_prox(JF, fy, Fx, yk, 1, l, lam)
        
        # 近端映射更新
        x = co.soft(np.dot(lam, l), yk - np.dot(JF.T, lam))

        # print("value", iteration, 1 / np.sum(lam / alpha) * np.linalg.norm(x - yk))
        
        # # 检查收敛条件
        # if (1 / np.sum(lam / alpha) * np.linalg.norm(x - yk) < erro or
        #     np.linalg.norm(x - yk) < 1e-7):
        #
        #     # 公平停机准则
        #     g_val = Func.g(x)
        #     G = Func.Gra(x)
        #     l1 = np.ones(len(g_val)) / n
        #
        #     lam1 = co.cond_gra_simp_prox1(G, g_val, x, l1, lam / alpha / np.sum(lam / alpha))
        #     y1 = co.soft(np.dot(lam1, l1), x - np.dot(G.T, lam1))
        #     v1 = y1 - x
        #
        #     if np.linalg.norm(v1) < erro:
        #         continue_iteration = False
        #         save_trajectory = False
        #
        # # 保存轨迹和距离
        # if save_trajectory:
        #     trajectory = np.vstack((trajectory, x))
        #     dist.append(1 / np.sum(lam / alpha) * np.linalg.norm(x - yk))
        #
        # stepsize.append(1.0)
        # iteration += 1

        ##############################公平停机准则###################################################
        g = Func.g(x)
        G = Func.Gra(x)

        l1 = 1 / n * np.ones(len(g))

        lam1 = co.cond_gra_simp_prox1(G, g, x, l1, lam1)
        y1 = co.soft(np.dot(lam1, l1), x - np.dot(G.T, lam1))
        v1 = y1 - x
        if np.linalg.norm(v1) < erro:
            # if np.linalg.norm(v) < erro:
            continue_iteration = False
            save_trajectory = False
        if save_trajectory:
            trajectory = np.vstack((trajectory, x))
            dist.append( np.linalg.norm(v1))
        stepsize.append(1.0)
        iteration += 1
        #################################################################################
        
        # 最大迭代次数检查
        if iteration >= k_max:
            continue_iteration = False
    
    return trajectory, K, stepsize, dist



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


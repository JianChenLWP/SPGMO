import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
from scipy.optimize import minimize
import time as tm
import os
import sys
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
from scipy.optimize import minimize
import time as tm

# 使用相对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))  # 上两级目录

sys.path.append(os.path.join(parent_dir, 'methods'))
sys.path.append(os.path.join(parent_dir, 'Problems'))
sys.path.append(os.path.join(parent_dir, 'data', 'QP'))

import PG_L as M1
import APG_L as M2
import APG_mu as M3
import BB_L as M4
import ABB_L as M5
import ABB_mu as M6

import QPa as QP1
import QPb as QP2
import QPc as QP3
import QPd as QP4
import QPe as QP5
import QPf as QP6
import QPg as QP7
import QPh as QP8


######################################### problem QP ############################################
problems = [QP1, QP2, QP3, QP4, QP5, QP6,QP7,QP8]                                                    #
Problem = ["QPa","QPb","QPc","QPd","QPe","QPf","QPg","QPh"]
methods = [M1,M2,M3,M4,M5,M6]#                                                                      #
Method = ["PGMO","APGMO","APGMO-sc","SPGMO","ASPGMO","ASPGMO-sc"]                                            #
#################################################################################################
#
# methods = [M3,M4]#                                                                      #
# Method = ["APGMO-sc","SPGMO"]

def pareto_front(data):
    # 获取维数
    num_dimensions = data.shape[1]
    # 初始化一个布尔数组，表示每个向量是否在 Pareto 前沿上
    is_pareto = np.ones(data.shape[0], dtype=bool)

    # 遍历数组中的每个向量
    for i, point in enumerate(data):
        if is_pareto[i]:  # 仅当当前点尚未被标记为支配时
            # 创建支配关系的布尔数组
            is_dominated = np.all(data >= point, axis=1)
            is_strictly_dominated = is_dominated & np.any(data > point, axis=1)
            # 如果当前点被支配，标记它
            is_pareto[is_strictly_dominated] = False

    # 返回 Pareto 前沿的向量
    return data[is_pareto]

# ######################################### problem QP ############################################
# problems = [QP1, QP2, QP3, QP4, QP5, QP6]                                                    #
# methods = [M2,M3,M4]                                                                         #
# Method = ["VMMO","BB","BBQN"]                                                                #
# Problem = ["QPa","QPb","QPc","QPd","QPe","QPf"]                                              #
# #################################################################################################
# F = []
# r = 4
# Func = problems[r]
# for method in methods:
#     c = methods.index(method)
#     A = np.loadtxt(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\BB_QN\data\QP\e'+Method[c] + Problem[r])
#     for x in A:
#         F.append(Func.Val(x))
# F = np.array(F)
# PF = pareto_front(F)
# # 将每个数组的行转换为集合中的 tuple
# set1 = set(map(tuple, F[2*len(A):(1+2)*len(A),:]))
# set2 = set(map(tuple, PF))
#
# # 求交集并计算交集数量
# intersection_count = len(set1 & set2)
# print(len(PF), F.shape,len(set1 & set2))
# #
# # F1 = F[0:200,:]
# # F2 = F[200:400,:]
# # F3 = F[400:600,:]
# # plt.scatter(F1[:, 0], F1[:, 1],  label='Group 1', s = 1, c = 'b')
# # plt.scatter(F2[:, 0], F2[:, 1],  label='Group 2', s = 1, c = 'r')
# # plt.scatter(F3[:, 0], F3[:, 1],  label='Group 3', s = 1, c = 'g')
# set1 = set(map(tuple, F))
# set2 = set(map(tuple, PF))
#
# t_ps = [len(PF) / (len(set(map(tuple, F[i*len(A):(1+i)*len(A),:])) & set(map(tuple, PF))) + 1) for i in range(len(methods))]
# print(len(F),len(PF),t_ps)
# plt.scatter(F[:, 0], F[:, 1],  label='Group 1', s = 5, c = 'b')
# plt.scatter(PF[:, 0], PF[:, 1],  label='Group 2', s = 1, c = 'r')
#
# # 添加图例和标签
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Scatter Plot with Two Colors")
# plt.legend()
#
#
# plt.show()






t_ps = []
for Func in problems: # index of problem
    r = problems.index(Func)
    F = [] #save function value
    for method in methods:
        c = methods.index(method)  # index of method
        print(Method[c], Problem[r])
        data_file = os.path.join(parent_dir, 'data', 'QP', 'L2000' + Method[c] + Problem[r])
        A = np.loadtxt(data_file)
        for x in A:
            F.append(Func.Val(x))
    F = np.array(F)
    PF = pareto_front(F)
    t_ps.append([len(PF) / (len(set(map(tuple, F[i * len(A):(1 + i) * len(A), :])) & set(map(tuple, PF))) + 1) for i in
                 range(len(methods))])




alg = ["PGMO$_L$","APGMO$_L$","APGMO_sc","SPGMO$_L$","ASPGMO$_L$","ASPGMO_sc"]


color = ['purple','orange','black','green', 'red','blue']

linestyles = [ '-.', '--',':', '-.','--',':']

# ##################3-4##############
# alg = ["APGMO_sc","SPGMO$_L$"]
#
#
# color = ['purple','green']
#
# linestyles = [ ':', '-.']


A = np.array(t_ps)  # 性能比率

minA = np.min(A, axis=1).reshape(-1, 1)  # 每行的最小值，即最优解
ratios = A / minA  # 性能比率
print(A,ratios)


# 计算 τ 的范围（取最大倍率作为上限）
tau_max = np.max(ratios)
tau_values = np.linspace(1, 1.05 * tau_max, 10000)  # 定义 τ 的范围

# tau_values = np.sort(np.unique(ratios.flatten()))
# 计算每个算法的性能剖面
profiles = []
for i in range(ratios.shape[1]):
    profile = [np.mean(ratios[:, i] <= tau) for tau in tau_values]
    profiles.append(profile)


# 设置字体和标签大小
plt.rcParams.update({
    'font.size': 16,        # 全局字体大小
    'axes.titlesize': 18,   # 图标题字体大小
    'axes.labelsize': 16,   # x 和 y 轴标签字体大小
    'xtick.labelsize': 14,  # x 轴刻度字体大小
    'ytick.labelsize': 14,  # y 轴刻度字体大小
    'legend.fontsize': 14   # 图例字体大小
})

# 绘制性能剖面图
plt.figure(figsize=(8, 6))
for i, profile in enumerate(profiles):
    plt.plot(tau_values, profile, color=color[i], linestyle=linestyles[i], label=alg[i],  linewidth=2)

plt.xscale('log')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\rho(\tau)$')
plt.title('Purity')
plt.ylim([0, 1.05])
plt.xlim([1, 1.05 * tau_max])
plt.legend(loc='lower right')
plt.grid(True, which="both", ls="--")
plt.savefig('Purity_QP2000'+".png",dpi=500,bbox_inches = 'tight')
plt.show()




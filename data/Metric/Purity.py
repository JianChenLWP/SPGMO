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
import DD1 as F1
import DEB as F2
import Far1 as F3
import FDS as F4
import FF1 as F5
import Hil1 as F6
import JOS_ill1 as F7
import JOS_ill2 as F8
import VU1 as F9
import WIT1 as F10
import WIT2 as F11
import WIT3 as F12



import PG as M1
import APG_ls as M2
import BB as M3
import ABB_ls as M4



######################################### ordinary problem ############################################
problems = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12]    #
Problem = ["DD1","DEB","Far1","FDS","FF1","Hil1","Imb1","Imb2","VU1","WIT1","WIT2","WIT3"]        #
methods = [M1,M2,M3,M4]                                                                        #
Method = ["PGMO","APGMO","SPGMO","ASPGMO"]                                                           #
######################################################################################################

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
# r = 14
# Func = problems[r]
# for method in methods:
#     c = methods.index(method)
#     A = np.loadtxt(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\BB_QN\data\Ordinary\e'+Method[c] + Problem[r])
#     for x in A:
#         F.append(Func.Val(x))
# F = np.array(F)
# PF = pareto_front(F)
# t_ps = [len(PF) / len(np.intersect1d(PF, F[i*len(A):(i+1)*len(A),:])) for i in range(len(methods))]
# print(len(F),len(PF),t_ps)
# plt.scatter(F[:, 0], F[:, 1],  label='Group 1', s = 5, c = 'b')
# plt.scatter(PF[:, 0], PF[:, 1],  label='Group 2', s = 1, c = 'r')
#
# # 添加图例和标签
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Scatter Plot with Two Colors")
# plt.legend()

# 显示图形
# plt.show()

t_ps = []
for Func in problems: # index of problem
    r = problems.index(Func)
    F = [] #save function value
    for method in methods:
        c = methods.index(method)  # index of method
        print(Method[c], Problem[r])
        data_file = os.path.join(parent_dir, 'data', 'Ordinary', 'e2000' + Method[c] + Problem[r])
        A = np.loadtxt(data_file)
        for x in A:
            F.append(Func.Val(x))
    F = np.array(F) + 1e-6 * np.random.rand()
    PF = pareto_front(F)
    # t_ps.append([len(PF) / len(np.intersect1d(PF, F[i*len(A):(i+1)*len(A),:])) for i in range(len(methods))])

    t_ps.append([len(PF) / (len(set(map(tuple, F[i * len(A):(1 + i) * len(A), :])) & set(map(tuple, PF))) + 1) for i in
            range(len(methods))])




alg = ["PGMO_ls","APGMO_bt","SPGMO_ls","ASPGMO_bt"]


color = ['orange','blue', 'green', 'red']
linestyles = ['--', '-.', '--',':']

A = np.array(t_ps)  # 性能比率

minA = np.min(A, axis=1).reshape(-1, 1)  # 每行的最小值，即最优解
ratios = A / minA  # 性能比率

# 计算 τ 的范围（取最大倍率作为上限）
tau_max = np.max(ratios)
tau_values = np.linspace(1, 1.05* tau_max, 10000)  # 定义 τ 的范围

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

# plt.xscale('log')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\rho(\tau)$')
plt.title('Purity')
plt.ylim([0, 1.05])
plt.xlim([1, 1.05* tau_max])
plt.legend(loc='lower right')
plt.grid(True, which="both", ls="--")
plt.savefig('Purity'+".png",dpi=500,bbox_inches = 'tight')
plt.show()




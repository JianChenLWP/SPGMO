import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
from scipy.optimize import minimize
import time as tm
import sys
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\Problem')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\Problem\test_problem')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\BB_QN\methods')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\BB_QN\problems')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\BB_QN\data\Ordinary')
import BK1 as F1
import DD1 as F2
import DEB as F3
import Far1 as F4
import FDS as F5
import FF1 as F6
import Hil1 as F7
import JOS_ill1 as F8
import JOS_ill2 as F9
import JOS1a as F10
import JOS1b as F11
import JOS1c as F12
import JOS1d as F13
import LE1 as F14
import PNR as F15
import VU1 as F16
import WIT1 as F17
import WIT2 as F18
import WIT3 as F19
import WIT4 as F20
import WIT5 as F21
import WIT6 as F22


import BBDMO as M3
import BBQN_wolfe as M4
import QN_Wolfe as M1
import VMMO_Wolfe as M2



######################################### ordinary problem ############################################
problems = [F1,F2,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22]    #
Problem = ["BK1","DD1","Far1","FDS","FF1","Hil1","Imb1","Imb2","JOS1a","JOS1b",          #
           "JOS1c","JOS1d","LE1","PNR","VU1","WIT1","WIT2","WIT3","WIT4","WIT5","WIT6"]        #
methods = [M1,M2,M3,M4]                                                                        #
Method = ["QNMO","VMMO","BB","BBQN"]                                                           #
######################################################################################################



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
    t_s = []
    for method in methods:
        F = []  # save function value
        c = methods.index(method)  # index of method
        print(Method[c], Problem[r])
        A = np.loadtxt(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\BB_QN\data\Ordinary\e'+Method[c] + Problem[r])
        for x in A:
            F.append(Func.Val(x))
        F = np.sort(np.array(F), axis=0) + 1e-6 * np.random.rand() #对每一列升序排列

        delta = np.abs(F[1:, :] - F[:-1, :])
        print(delta)
        J = []
        for i in range(delta.shape[1]):
            delta_c = delta[:,i]                   # 每一列分别求
            t_s_i = (delta_c[0] + delta_c[-1] + np.sum(np.abs(delta_c - np.mean(delta_c)))) / \
            (delta_c[0] + delta_c[-1] + (len(delta_c) - 1) * np.mean(delta_c))
            J.append(t_s_i)
        t_s.append(max(J))
    t_ps.append(t_s)

print(t_ps)

alg = ['QNMO','MQNMO','BBDMO','BBQNMO']


color = ['orange','blue', 'green', 'red']
linestyles = ['--', '-.', ':', '--']

A = np.array(t_ps)  # 性能比率

minA = np.min(A, axis=1).reshape(-1, 1)  # 每行的最小值，即最优解
ratios = A / minA  # 性能比率

# 计算 τ 的范围（取最大倍率作为上限）
tau_max = np.max(ratios)
tau_values = np.linspace(1, 1.05*tau_max, 10000)  # 定义 τ 的范围

# 计算每个算法的性能剖面
profiles = []
for i in range(ratios.shape[1]):
    profile = [np.mean(ratios[:, i] <= tau) for tau in tau_values]
    profiles.append(profile)

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
plt.title('Spread '+r'$\Gamma$')
plt.ylim([0, 1.05])
plt.xlim([1, 1.05*tau_max])
plt.legend(loc='lower right')
plt.grid(True, which="both", ls="--")
plt.savefig('Gamma'+".png",dpi=500,bbox_inches = 'tight')
plt.show()




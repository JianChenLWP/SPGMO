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
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo\methods')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo\problems')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo\data\QP')

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



######################################### problem QP ############################################
problems = [QP1, QP2, QP3, QP4, QP5, QP6]                                                    #
methods = [M1,M2,M3,M4,M5,M6]
Problem = ["QPa","QPb","QPc","QPd","QPe","QPf"]        #                                                                      #
Method = ["PGMO","APGMO","APGMO-sc","SPGMO","ASPGMO","ASPGMO-sc"]                                            #
#################################################################################################



t_ps = []
for Func in problems: # index of problem
    r = problems.index(Func)
    t_s = []
    for method in methods:
        F = []  # save function value
        c = methods.index(method)  # index of method
        print(Method[c], Problem[r])
        A = np.loadtxt(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo\data\QP\L2000'+Method[c]+Problem[r])
        for x in A:
            F.append(Func.Val(x))
        F = np.sort(np.array(F), axis=0) #对每一列升序排列

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




alg = ["PGMO$_L$","APGMO$_L$","APGMO_sc","SPGMO$_L$","ASPGMO$_L$","ASPGMO_sc"]


color = ['black','orange', 'purple','green', 'red','blue']

linestyles = [ '-.', '--',':', '-.','--',':']

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
plt.xlim([1, 1.05 * tau_max])
plt.legend(loc='lower right')
plt.grid(True, which="both", ls="--")
plt.savefig('Gamma_QP2000'+".png",dpi=500,bbox_inches = 'tight')
plt.show()




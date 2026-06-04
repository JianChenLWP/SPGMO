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
import PG_ls_PS as M1
import APG_ls_PS as M2
import BB_ls_PS as M3
import ABB_ls_PS as M4

import FF25a as QP1
import FF25b as QP2
import FF25c as QP3
import FF25d as QP4

import FF32a as QP5
import FF32b as QP6
import FF32c as QP7
import FF32d as QP8

import FF48a as QP9
import FF48b as QP10
import FF48c as QP11
import FF48d as QP12

import FF100a as QP13
import FF100b as QP14
import FF100c as QP15
import FF100d as QP16





######################################### ordinary problem ############################################
problems = [QP1,QP2,QP3,QP4,QP5,QP6,QP7,QP8,QP9,QP10,QP11,QP12,QP13,QP14,QP15,QP16]    #
Problem = ["FF25a","FF25b","FF25c","FF25d","FF32a","FF32b","FF32c","FF32d","FF48a","FF48b","FF48c","FF48d","FF100a","FF100b","FF100c","FF100d"]        #
methods = [M1,M2,M3,M4]                                                                        #
Method = ["PGMO","APGMO","SPGMO","ASPGMO"]                                                           #
######################################################################################################
# methods = [M3]                                                                        #
# Method = ["SPGMO"]


# ######################################### problem QP ############################################
# problems = [QP1, QP2, QP3, QP4, QP5, QP6]                                                    #
# methods = [M2,M3,M4]                                                                         #
# Method = ["VMMO","BB","BBQN"]                                                                #
# Problem = ["QPa","QPb","QPc","QPd","QPe","QPf"]                                              #
# #################################################################################################



r = 15
Func = problems[r]
for method in methods:
    c = methods.index(method)
    file_path = os.path.join(parent_dir, 'data', 'PS', 'PS' + Method[c] + Problem[r])
    A = np.loadtxt(file_path)
    F = []
    for x in A:
        F.append(Func.Val(x))

    F = np.array(F)

    if len(F[0]) == 2:
        plt.scatter(F[:, 0], F[:, 1], s=5, c='k')

        # 添加图例和标签
        plt.xlabel("$F_{1}(x)$")
        plt.ylabel("$F_{2}(x)$")
        plt.title(Problem[r])
        # plt.savefig(Method[c] + Problem[r] + ".png", dpi=500, bbox_inches='tight')
        # plt.legend()

        plt.show()
    else:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel("$F_{1}(x)$")
        ax.set_ylabel("$F_{2}(x)$")
        ax.set_zlabel("$F_{3}(x)$")
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=5, c='k')
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title(Problem[r])
        # plt.savefig(Method[c] + Problem[r] + ".png", dpi=500, bbox_inches='tight')
        plt.show()











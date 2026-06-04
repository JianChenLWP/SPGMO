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
methods = [M3,M4]                                                                        #
Method = ["SPGMO","ASPGMO"]


# ######################################### problem QP ############################################
# problems = [QP1, QP2, QP3, QP4, QP5, QP6]                                                    #
# methods = [M2,M3,M4]                                                                         #
# Method = ["VMMO","BB","BBQN"]                                                                #
# Problem = ["QPa","QPb","QPc","QPd","QPe","QPf"]                                              #
# #################################################################################################



r = 11
Func = problems[r]
for method in methods:
    c = methods.index(method)
    file_path = os.path.join(parent_dir, 'data', 'Ordinary', 'e2000' + Method[c] + Problem[r])
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











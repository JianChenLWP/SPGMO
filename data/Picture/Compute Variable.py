import numpy as np
import matplotlib.pyplot as plt
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
problems = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12]
Problem = ["DD1", "DEB", "Far1", "FDS", "FF1", "Hil1", "Imb1", "Imb2", "VU1", "WIT1", "WIT2", "WIT3"]

method = "SPGMO_ls"
Method = "SPGMO"
######################################################################################################


r = 3   # 问题编号，例如 WIT3


file_path = os.path.join(parent_dir, 'data', 'Ordinary', 'e2000' + Method + Problem[r])



A = np.loadtxt(file_path)
A = np.atleast_2d(A)

if A.shape[1] < 2:
    print(f"{Method} - {Problem[r]} 的自变量维度小于 2，无法作图。")
else:
    plt.figure(figsize=(7, 5.5), dpi=120)

    plt.scatter(
        A[:, 0],
        A[:, 1],
        s=5, c='r'
    )

    plt.xlabel(r"$x_1$", fontsize=13)
    plt.ylabel(r"$x_2$", fontsize=13)
    plt.title(f"{method} on {Problem[r]}", fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.35)


    plt.tight_layout()

    # 如果想保存图片，取消下面这行注释
    plt.savefig(f"{Problem[r]}.png", dpi=500, bbox_inches='tight')

    plt.show()

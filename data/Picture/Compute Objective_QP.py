import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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

######################################### ordinary problem ############################################
problems = [QP1,QP2,QP3,QP4,QP5,QP6,QP7,QP8]    #
Problem = ["QPa","QPb","QPc","QPd","QPe","QPf","QPg","QPh"]        #
methods = [M1,M2,M3,M4,M5,M6]                                                                        #
Method = ["PGMO","APGMO","APGMO-sc","SPGMO","ASPGMO","ASPGMO-sc"]                                                         #
######################################################################################################

alg = ["PGMO$_L$","APGMO$_L$","APGMO_sc","SPGMO$_L$","ASPGMO$_L$","ASPGMO_sc"]


color = ['purple','orange','black','green', 'red','blue']

markers = [
    '*',
    '^',
    '.',
    'o',
    's',
    'D'
]

####################################
# methods = [M4,M5,M6]
# Method = ["SPGMO","ASPGMO","ASPGMO-sc"]


# alg = ["SPGMO$_L$","ASPGMO$_L$","ASPGMO_sc"]
#
#
# color = ['green', 'red','blue']
#
# markers = [
#     'o',
#     's',
#     'D'
# ]



r = 5

Func = problems[r]
# 设置字体和标签大小
plt.rcParams.update({
    'font.size': 16,        # 全局字体大小
    'axes.titlesize': 18,   # 图标题字体大小
    'axes.labelsize': 16,   # x 和 y 轴标签字体大小
    'xtick.labelsize': 14,  # x 轴刻度字体大小
    'ytick.labelsize': 14,  # y 轴刻度字体大小
    'legend.fontsize': 14   # 图例字体大小
})

fig, ax_main = plt.subplots(figsize=(8, 6))

#添加子图
left, bottom, width, height = [0.3, 0.25, 0.3, 0.3]  # 子图的位置和大小
ax_inset = fig.add_axes([left, bottom, width, height])

# 子图缩放范围
zoom_x_min, zoom_x_max = -0.2*1e8, 1.5*1e8
zoom_y_min, zoom_y_max = -4*1e4, -3*1e4
ax_inset.set_xlim(zoom_x_min, zoom_x_max)
ax_inset.set_ylim(zoom_y_min, zoom_y_max)

# 遍历方法并绘图
for method in methods:
    c = methods.index(method)  # 获取当前方法的索引
    # 加载数据
    file_path = os.path.join(parent_dir, 'data', 'QP', 'L2000' + Method[c] + Problem[r])
    A = np.loadtxt(file_path)

    # 计算目标函数值
    F = np.array([Func.Val(x) for x in A])

    # 绘制主图
    ax_main.scatter(F[:, 0], F[:, 1], label=alg[c], marker=markers[c], s=5, c=color[c])

    # 筛选在子图范围内的数据并绘制
    mask = (F[:, 0] >= zoom_x_min) & (F[:, 0] <= zoom_x_max) & \
           (F[:, 1] >= zoom_y_min) & (F[:, 1] <= zoom_y_max)
    ax_inset.scatter(F[mask, 0], F[mask, 1], marker=markers[c], s=5, c=color[c])

# 设置主图的标签、标题和图例
ax_main.set_xlabel("$F_{1}(x)$")
ax_main.set_ylabel("$F_{2}(x)$")
ax_main.set_title(Problem[r])
ax_main.legend(loc='lower right')

###强制主图和子图的 x 轴、y 轴都使用科学计数法
for ax in [ax_main, ax_inset]:
    formatter_x = ScalarFormatter(useMathText=True)
    formatter_y = ScalarFormatter(useMathText=True)

    formatter_x.set_scientific(True)
    formatter_y.set_scientific(True)

    formatter_x.set_powerlimits((0, 0))
    formatter_y.set_powerlimits((0, 0))

    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)

    ax.ticklabel_format(
        axis="both",
        style="scientific",
        scilimits=(0, 0),
        useMathText=True
    )

    ax.xaxis.get_offset_text().set_fontsize(14)
    ax.yaxis.get_offset_text().set_fontsize(14)

plt.savefig(Problem[r]+".png",dpi=500,bbox_inches = 'tight')
plt.tight_layout()
plt.show()









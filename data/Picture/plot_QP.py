import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter
import time
import sys
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\Problem')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\Problem\test_problem')
# sys.path.append(r'C:\PycharmProjects\pythonProjectRL\EMO\Problem\ZDT')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo\quad_prob')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo\methods')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo\problems')
from matplotlib.ticker import LogLocator, LogFormatter
import QP_test as Func

import ABB_ls_rs as M1
import ABB_L_rs as M2
import ABB_mu as M3


methods = [M1,M2]#                                                                      #

Method = ["ASPGMO_ls$","ASPGMO$_L$","ASPGMO_sc"]

color = ['green', 'red','blue']


num = 5
k_max = 2000
restart = 300



dim = Func.dim
width = Func.width
interval_start = Func.interval_start
np.random.seed(1110)
X = width * np.random.rand(num, dim) + interval_start

# x = np.array([[-2.01,-2.01],[-2.01,-2.01]])
# x[:][1] = 1
erro = 1e-120

# 生成一些示例数据
x = np.linspace(0, k_max, k_max) #横坐标

for c in range(len(methods)):
    print(c)
    list, k, s, dist = methods[c].MGD(Func, X[0], erro, k_max, restart)
    plt.plot(x, dist, linestyle='--', linewidth=0.5, color=color[c], label=Method[c])

list, k, s, dist = M3.MGD(Func, X[0], erro, k_max)
plt.plot(x, dist, linestyle='--', linewidth=0.5, color=color[2], label=Method[2])







# 设置 y 轴范围从 1e-16 到 1e4，并将 y 轴设置为对数刻度
plt.ylim(1e-16, 1e4)
plt.yscale('log')

# 设置 y 轴主刻度和次刻度
from matplotlib.ticker import LogLocator, FuncFormatter

# 主刻度位置（10的幂次）
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0))  # 主刻度
# 次刻度位置（2到9的倍数）
plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))  # 次刻度

# 自定义主刻度标签（10^a）
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"$10^{{{int(np.log10(y))}}}$"))

# 确保次刻度显示
plt.tick_params(axis='y', which='minor', length=4, color='gray', labelsize=8)  # 调整次刻度的外观
plt.tick_params(axis='y', which='major', length=6, color='black', labelsize=10)  # 调整主刻度的外观

# 添加标题和标签
plt.xlabel("k")
plt.ylabel(r"error")
plt.legend()

# 保存并显示图像
plt.grid(True, which="both", ls="--")

# plt.savefig("QPf_norm6", dpi=500, bbox_inches='tight')

plt.show()


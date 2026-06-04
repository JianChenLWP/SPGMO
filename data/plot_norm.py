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
import qpim1 as Func
import BB_L as alg1
import ABB_L as alg2
import ABB_mu as alg3



start = time.process_time()

num = 2
k_max = 3000


dim = Func.dim
width = Func.width
interval_start = Func.interval_start
np.random.seed(1110)
x = width * np.random.rand(num, dim) + interval_start

# x = np.array([[-2.01,-2.01],[-2.01,-2.01]])
# x[:][1] = 1
erro = 1e-120
sigma = 1e-4
L = 0
K = 0
S = 0
endpoint50 = []
endpoint100 = []
endpoint500 = []



list, k, s, dist1 = alg1.MGD(Func,x[0],erro,k_max)

list2, k2, s2, dist2 = alg2.MGD(Func,x[0],erro,k_max)

list3, k3, s3, dist3 = alg3.MGD(Func,x[0],erro,k_max)




# 生成一些示例数据
x = np.linspace(0, k_max, k_max)

y1 = dist1

y2 = dist2

y3 = dist3

# 创建散点图
plt.plot(x, y1, linestyle='--', linewidth=0.5, color='r', label=r'SPGMO$_L$')
plt.plot(x, y2, linestyle='--', linewidth=0.5, color='g', label=r'ASPGMO$_L$')
plt.plot(x, y3, linestyle='--', linewidth=0.5, color='b', label=r'ASPGMO_sc')

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

plt.grid(True, which="both", ls="--")
# 保存并显示图像
plt.savefig("QPd_norm", dpi=500, bbox_inches='tight')

plt.show()


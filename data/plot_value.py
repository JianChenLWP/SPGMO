import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter
import time
import sys
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\Problem')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\Problem\test_problem')
# sys.path.append(r'C:\PycharmProjects\pythonProjectRL\EMO\Problem\ZDT')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo\quad_prob')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo')
import qpim1 as Func
import BB_L as alg1
import ABB_L as alg2
import ABB_mu as alg3



start = time.process_time()
num = 50
k_m = 500


dim = Func.dim
width = Func.width
interval_start = Func.interval_start
np.random.seed(1110)
x = width * np.random.rand(num, dim) + interval_start

# x = np.array([[-2.01,-2.01],[-2.01,-2.01]])
# x[:][1] = 1
erro = 1e-8
sigma = 1e-4
L = 0
K = 0
S = 0
endpoint50 = []
endpoint100 = []
endpoint500 = []



list1, k, s, dist1 = alg1.MGD(x[0], erro, sigma, k_m)

list2, k2, s2, dist2 = alg2.MGD(x[0], erro, sigma, k_m)

list3, k3, s3, dist3 = alg3.MGD(x[0], erro, sigma, k_m)




# 生成一些示例数据
x = np.linspace(0, k_m+1, k_m+1)

Val1 = np.array([Func.Val(i) + Func.g(i) for i in list1])
Val2 = np.array([Func.Val(i) + Func.g(i) for i in list2])
Val3 = np.array([Func.Val(i) + Func.g(i) for i in list3])




# 创建散点图
plt.plot(x, Val1[:,0],   linestyle='--', linewidth=0.3, color='r', label = r'SPGMO')

plt.plot(x, Val2[:,0],  linestyle='--', linewidth=0.3, color='g', label = r'ASPGMO')

plt.plot(x, Val3[:,0],  linestyle='--', linewidth=0.3, color='b', label = r'ASPGMO-sc')

# 设置 y 轴范围从 1 (10^0) 到 10^{-6}

# plt.ylim(1e3,1e6)

# 将 y 轴设置为对数刻度
plt.yscale('log')

# 设置 y 轴的刻度位置
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0))  # 主要刻度以10为底

# #
# def scientific_notation_formatter(val, pos):
#     if val == 0:
#         return "0"  # 如果值为0，显示为0
#     exponent = int(np.log10(abs(val)))  # 获取指数（对数刻度的绝对值）
#     base = 10 ** exponent
#     coefficient = val / base  # 获取系数
#     return f"{coefficient:.0f}×$10^{{{exponent}}}$"  # 格式化为 LaTeX 的 10^x 形式
#
# # 应用自定义格式
# plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))

# 应用自定义格式
# plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))
#
# 添加标题和标签

plt.xlabel("k")
plt.ylabel('$F_{1}$')


plt.legend()

# plt.savefig("QPf_value1",dpi=500,bbox_inches = 'tight')
plt.show()


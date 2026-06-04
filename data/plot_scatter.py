import numpy as np
import copy
import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# from SCI.spatial.distance import cdist
import time
import sys
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\Problem')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\Problem\test_problem')
# sys.path.append(r'C:\PycharmProjects\pythonProjectRL\EMO\Problem\ZDT')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo\quad_prob')
sys.path.append(r'C:\PycharmProjects\pythonProjectRL\Gradient Methods\BB_step\proximal_gradient_mo')
import qpim1 as Func
import APG_ls as alg

pro1 = '_bbL'
pro2 = '_abbL'
pro3 = '_abbmu'
# pro4 = '_abbls'

num = '500'

s1 = np.loadtxt(pro1+num)
s2 = np.loadtxt(pro2+num)
s3 = np.loadtxt(pro3+num)
# s4 = np.loadtxt(pro4+num)

fig = plt.figure()
val1 = np.array([Func.Val(x) + Func.g(x) for x in s1])
val2 = np.array([Func.Val(x) + Func.g(x) for x in s2])
val3 = np.array([Func.Val(x) + Func.g(x) for x in s3])
# val4 = np.array([Func.Val(x) + Func.g(x) for x in s4])
plt.xlabel('$F_1$')
plt.ylabel("$F_2$")
plt.scatter(val1[:,0], val1[:,1], marker = '^', s = 5, c = 'r', label = r'SPGMO')

plt.scatter(val2[:,0], val2[:,1], marker = '*', s = 5, c = 'g',label = r'ASPGMO')

plt.scatter(val3[:,0], val3[:,1], marker = '.', s = 2, c = 'b', label = r'ASPGMO-sc')
plt.title("Value Space")
plt.legend()
# plt.scatter(val4[:,0], val4[:,1], s = 2, c = 'k')


# 添加一个嵌入的子图，并设置指定区域的范围
# 这里定义了一个子图位置： [左边缘, 下边缘, 宽度, 高度]
left, bottom, width, height = [0.3, 0.3, 0.4, 0.4]
ax_inset = fig.add_axes([left, bottom, width, height])

# 设置子图显示的区域（缩放范围）
zoom_x_min, zoom_x_max = -5000, 20000
zoom_y_min, zoom_y_max = -200, 800



# 在子图中绘制指定区域的数据点
ax_inset.scatter(val1[:,0], val1[:,1], marker = '^', s = 5, c = 'r')
ax_inset.scatter(val2[:,0], val2[:,1], marker = '*', s = 5, c = 'g')
ax_inset.scatter(val3[:,0], val3[:,1], marker = '.', s = 2, c = 'b')


ax_inset.set_xlim(zoom_x_min, zoom_x_max)
ax_inset.set_ylim(zoom_y_min, zoom_y_max)


plt.savefig("QPd"+num,dpi=500,bbox_inches = 'tight')

plt.show()



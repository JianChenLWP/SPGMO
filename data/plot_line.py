import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter

# 生成一些示例数据
x = np.linspace(0, 500, 500)
y = 10 ** (-x / 2)  # 创建一个在对数刻度下变化的数据集



# 创建散点图
plt.scatter(x, y)

# 设置 y 轴范围从 1 (10^0) 到 10^{-6}
plt.ylim(1e-6,1)

# 将 y 轴设置为对数刻度
plt.yscale('log')

# 设置 y 轴的刻度位置
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0))  # 主要刻度以10为底

# 使用 FuncFormatter 自定义 y 轴标签格式为 10^{x} 的形式
def scientific_notation_formatter(val, pos):
    exponent = int(np.log10(val))  # 获取指数
    return f"$10^{{{exponent}}}$"  # 格式化为 LaTeX 表示的 10^x 形式

# 应用自定义格式
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))

# 添加标题和标签

plt.xlabel("k")
plt.ylabel(r"$\|\|x^{k+1}-y^{k}\|\|$")

plt.show()

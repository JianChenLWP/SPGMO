import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter

# 生成示例数据
x = np.linspace(0, 10, 100)
y = 10 ** (-x / 2)  # 在对数刻度下变化的数据集

plt.figure(figsize=(8, 6))

# 绘制散点图
plt.scatter(x, y)

# 设置 y 轴范围从 10^{-8} 到 10^2
plt.ylim(1e-8, 1e2)

# 设置 y 轴为对数刻度
plt.yscale('log')

# 设置主要刻度和子刻度
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))  # 主要刻度以10为底
plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))  # 子刻度在主要刻度之间

# 使用自定义格式将 y 轴标签格式化为 10^x 的形式
def scientific_notation_formatter(val, pos):
    exponent = int(np.log10(val))
    return f"$10^{{{exponent}}}$"

# 应用自定义格式
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation_formatter))

# 添加标题和标签
plt.title("Y轴为 $10^{-8}$ 到 $10^{2}$ 等距刻度的散点图")
plt.xlabel("X轴")
plt.ylabel("Y轴 (对数刻度)")

plt.grid(True, which="both", linestyle='--', linewidth=0.5)  # 显示主网格和次网格
plt.show()

import numpy as np
import pandas as pd
from pathlib import Path


F = "FF100MEOP"
dim = 100
width = 1
interval_start = 0


# 当前 FF25.py 所在文件夹，也就是 Problems 文件夹
THIS_DIR = Path(__file__).resolve().parent

# 对应你图片中的位置：
# Problems/ff_benchmark_data/FF25/FF25_stats.pkl
STATS_PATH = THIS_DIR / "ff_benchmark_data" / "FF100MEOP" / "FF100MEOP_stats.pkl"

if not STATS_PATH.exists():
    raise FileNotFoundError(f"Cannot find FF25 data file:\n{STATS_PATH}")

stats = pd.read_pickle(STATS_PATH)

V = stats["covariance"].to_numpy(dtype=float)
a = stats["mean"].to_numpy(dtype=float)

def Val(x):
    f1 = - np.dot(a,x)
    f2 = 1 / 2 * np.dot(x,np.dot(V,x))
    return np.array([f1, f2])
def Gra(x):
    Gra_1 = - a
    Gra_2 = np.dot(V,x)
    return np.vstack((Gra_1, Gra_2))

def Ifindomain(x):
    for i in range(len(x)):
        if -1e12 < x[i] < 1e12:
            A = True
        else:
            A = False
            break
    return A

# print(Val(np.ones(100)))
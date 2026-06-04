import numpy as np
import pandas as pd
from pathlib import Path

F = "FF25b"
dim = 25
width = 1
interval_start = 0

imb = 10


# 当前 FF25.py 所在文件夹，也就是 Problems 文件夹
THIS_DIR = Path(__file__).resolve().parent

# 对应你图片中的位置：
# Problems/ff_benchmark_data/FF25/FF25_stats.pkl
STATS_PATH = THIS_DIR / "ff_benchmark_data" / "FF25" / "FF25_stats.pkl"

if not STATS_PATH.exists():
    raise FileNotFoundError(f"Cannot find FF25 data file:\n{STATS_PATH}")

stats = pd.read_pickle(STATS_PATH)

V = stats["covariance"].to_numpy(dtype=float)
a = stats["mean"].to_numpy(dtype=float)

def Val(x):
    f1 = - np.dot(a,x)
    f2 = imb * np.dot(x,np.dot(V,x))
    return np.array([f1, f2])
def Gra(x):
    Gra_1 = - a
    Gra_2 = 2 * imb * np.dot(V,x)
    return np.vstack((Gra_1, Gra_2))

def Ifindomain(x):
    for i in range(len(x)):
        if -1e12 < x[i] < 1e12:
            A = True
        else:
            A = False
            break
    return A

# print(Val(np.ones(25)))
from pathlib import Path
import sys

# 获取项目根目录（ZZB）
PROJECT_ROOT = Path.cwd()
#print("project root =", PROJECT_ROOT)

# 加入 Python 搜索路径
sys.path.append(str(PROJECT_ROOT))

from Src.simulation import simulate_once

from Src.config import *

import numpy as np

# taus = np.array([60, 120, 200])

# result = simulate_once(
#     fs=FS,
#     T=T,
#     B=B,
#     taus=taus,
#     target_index=1,
#     h=8,
# )

# print("Original")

# print(result["taus"])

# print()

# print("Shifted")

# print(result["shifted_taus"])

# print()

# print("Bhattacharyya Distance")
# print(result["BD"])

# print()

# print("Bhattacharyya Error Probability")
# print(result["Pe"])

taus = [60,120,200]

for h in range(41):

    DB = simulate_once(
        fs=FS,
        T=T,
        B=B,
        taus=taus,
        target_index=1,
        h=h,
    )

    print(h, DB)
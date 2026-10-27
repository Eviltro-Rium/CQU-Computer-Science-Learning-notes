import numpy as np

#正向传播
I = 1 - np.exp(-1) # 初始猜测 I_0
results = []

for n in range(0, 50):  # 从1到50
    results.append((n, I))
    I = 1 - n*I  # 计算 I_n
    print(f"第{1+n}次迭代：I_{n} = {I}")

print(f"计算得到的 I_50: {I}")

import numpy as np

#反向传播
I = 100.0  # 初始猜测 I_50 = 100
results = []

for n in range(50, 0, -1):  # 从50反向到1
    results.append((n, I))
    I = (1 - I) / n  # 计算 I_{n-1}
    print(f"第{51-n}次迭代：I_{n} = {I}")

I_0 = 1 - 1/np.e  # I_0 的精确值
print(f"计算得到的 I_0: {I}")
print(f"精确的 I_0: {I_0}")
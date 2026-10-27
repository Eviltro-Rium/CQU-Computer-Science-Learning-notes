import numpy as np

A = np.array([[10, -1, -2],
              [-1, 10, -2],
              [-1, -1, 5]], dtype=np.float64)

b = np.array([7.2, 8.3, 4.2], dtype=np.float64)

x = np.zeros(3, dtype=np.float64)

for k in range(10):

    for i in range(3):
        tmp = 0.0

        # 遍历当前方程的所有系数（除了对角线元素）
        for j in range(3):
            if (i != j):  # 排除对角线元素（i == j 的情况）
                tmp += A[i, j] * x[j]

        # 高斯-赛德尔迭代法：x_i = (b_i - Σ(a_ij * x_j)) / a_ii
        x[i] = (b[i] - tmp) / A[i][i]

#打印结果
    print(f"第{k + 1}次迭代：{x[0]:.5f}, {x[1]:.5f}, {x[2]:.5f}")


import numpy as np

#A为系数矩阵，b为向量
def solve(A, b):

    A = np.array(A,dtype=float)
    b = np.array(b,dtype=float)

    # 增广
    n = len(b)
    matrix = np.hstack((A, b.reshape(n, 1)))

    for col in range(n):
        # 找到主元行
        pivot_row = np.argmax(np.abs(matrix[col:, col])) + col

        # 交换当前行与主元行
        matrix[[col, pivot_row]] = matrix[[pivot_row, col]]

        # 归一化主元行
        pivot = matrix[col, col]
        matrix[col] /= pivot

        # 高斯消元法
        for row in range(col + 1, n):
            factor = matrix[row, col]
            matrix[row] -= factor * matrix[col]

    x = np.zeros(n)
    #回带
    for i in range(n - 1, -1, -1):
        x[i] = matrix[i, n] - np.dot(matrix[i, i + 1:n], x[i + 1:n])

    return x

if __name__ == "__main__":
    A = np.array([[1.1348, 3.8326, 1.1651, 3.4017],
                  [0.5301, 1.7875, 2.5330, 1.5435],
                  [3.4129, 4.9317, 8.7643, 1.3142],
                  [1.2371, 4.9998, 10.6721, 0.0147]])

    b = np.array([9.5432, 6.3941, 18.4231, 16.9237])

    x = solve(A, b)

    for i in range(len(x)):
        print(f"x{i + 1} = {x[i]:.6f}")
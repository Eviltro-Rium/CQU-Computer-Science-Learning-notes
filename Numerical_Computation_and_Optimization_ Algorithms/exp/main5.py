import numpy as np

def build_matrix(x, y, f_0, f_n, len):
    h = np.diff(x)
    f = np.zeros(len+1)
    g = np.zeros(len)

    #把端点倒数和差商放入一个向量中
    f[0] = f_0
    f[len] = f_n
    for i in range(1, len):
        f[i] = (y[i] - y[i-1])/h[i-1]


    for i in range(len):
        g[i] = 6*(f[i+1] - f[i])

    #构建方程组
    A = np.zeros((len, len))
    for i in range(len):
        if(i == 0):
            A[i][0] = 2*h[0]
            A[i][1] = h[0]
        elif(i == len - 1):
            A[len-1][len-1] = 2*h[len-2]
            A[len-1][len-2] = h[len-2]
        else:
            A[i][i-1] = h[i-1]
            A[i][i] = 2*(h[i] + h[i-1])
            A[i][i+1] = h[i]

    return A,g

#直接调用第二次实验写的解方程的函数（高斯消元法）
def solve(A, b):

    A = np.array(A,dtype=float)
    b = np.array(b,dtype=float)

    # 创建增广矩阵
    n = len(b)
    matrix = np.hstack((A, b.reshape(n, 1)))

    for col in range(n):
        # 找到主元行（当前列中绝对值最大的元素所在行）
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
    #回带求解
    for i in range(n - 1, -1, -1):
        x[i] = matrix[i, n] - np.dot(matrix[i, i + 1:n], x[i + 1:n])

    return x


def evaluate_spline(x_data, y_data, M, x_target):
    """
    使用标准的三次样条插值公式
    """
    n = len(x_data)

    # 找到所在区间
    i = np.searchsorted(x_data, x_target) - 1
    i = max(0, min(i, n - 2))

    h_i = x_data[i + 1] - x_data[i]
    x1 = x_data[i]
    x2 = x_data[i + 1]

    # 标准三次样条公式
    term1 = M[i] * (x2 - x_target) ** 3 / (6 * h_i)
    term2 = M[i + 1] * (x_target - x1) ** 3 / (6 * h_i)
    term3 = (y_data[i] - M[i] * h_i ** 2 / 6) * (x2 - x_target) / h_i
    term4 = (y_data[i + 1] - M[i + 1] * h_i ** 2 / 6) * (x_target - x1) / h_i

    y_target = term1 + term2 + term3 + term4
    return y_target


if __name__ == "__main__":
    x = np.array([0.52, 3.1, 8.0, 17.95, 28.65, 39.62, 50.65,
                  78, 104.6, 156.6, 208.6, 260.7, 312.5,
                  364.4, 416.3, 468, 494, 507, 520])
    y = np.array([5.28794,9.4,13.84,20.2,24.9,28.44,31.1,
                  35, 36.5, 36.6, 34.6, 31.6, 31.0,
                  20.9, 14.8, 7.8, 3.7, 1.5, 0.2])
    A, b = build_matrix(x, y, 1.8648, -0.046115, len(x))

    #求取解向量M
    M = solve(A, b)

    print("解出的 M 向量:")
    for i, m_val in enumerate(M):
        print(f"M[{i}] = {m_val:.6f}")

    # 需要计算纵坐标的横坐标点
    target_x = np.array([2, 4, 6, 12, 16, 30, 60, 110, 180, 280, 400, 515])

    print("\n在指定横坐标处的纵坐标:")
    print("x\t\ty")
    print("-" * 20)
    for x_val in target_x:
        y_val = evaluate_spline(x, y, M, x_val)
        print(f"{x_val}\t\t{y_val:.4f}")

    # 验证插值准确性
    print("\n验证插值准确性:")
    print("x_data\t\ty_data\t\t\ty_interp\t\t\t误差")
    print("-" * 70)
    max_error = 0
    for i in range(len(x)):
        y_interp = evaluate_spline(x, y, M, x[i])
        error = abs(y_interp - y[i])
        max_error = max(max_error, error)
        print(f"{x[i]:6.2f}\t{y[i]:15.10f}\t{y_interp:15.10f}\t{error:.10e}")

    print(f"\n最大误差: {max_error:.10e}")
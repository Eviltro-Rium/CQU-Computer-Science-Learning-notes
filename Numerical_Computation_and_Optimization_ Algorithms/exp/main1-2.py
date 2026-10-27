import numpy as np

#原函数
def fun(x):
    x1, x2 = x
    return 60 - 10 * x1 - 4 * x2 + x1 ** 2 + x2 ** 2 - x1 * x2

#梯度
def gradient(x):
    x1, x2 = x
    return np.array([2 * x1 - x2 - 10,
                     2 * x2 - x1 - 4])

#海森矩阵的逆
def hessian_inv():
    """返回矩阵的逆（常数矩阵）"""
    return np.linalg.inv(np.array([[2, -1], [-1, 2]]))


def newton(x0, max_iter=10):
    H = hessian_inv()
    x = np.array(x0)

    print(f"初始点: {x}")

    for i in range(max_iter):
        grad = gradient(x)
        print(f"第{i+1}次迭代: x = {x}, f(x) = {fun(x):.6f}, ∇f = {grad}")

        if np.linalg.norm(grad) < 1e-10:
            #跳出循环
            break

        #更新x
        x = x - H @ grad

    return x


if __name__ == "__main__":
    ans = newton([0, 0])
    print(f"\n最优解: {ans}")
    print(f"最小值: {fun(ans)}")


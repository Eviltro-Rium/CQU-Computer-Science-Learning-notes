#Euler算法
import numpy as np

def wei_fen_fang_cheng(x, y):
    #微分方程右侧函数
    return y - 2 * x / y


def exact_solution(x):
    #精确解
    return np.sqrt(1 + 2 * x)


def euler(f, x0, y0, h, n_steps):

    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    for i in range(n_steps):
        y = y + h * f(x, y)
        x = x + h

        x_values.append(x)
        y_values.append(y)

    return np.array(x_values), np.array(y_values)



if __name__ == "__main__":

    x0 = 0.0  # x
    y0 = 1.0  # y
    h = 0.1  # 步长
    n_steps = 10

    # 求解
    x_num, y_num = euler(wei_fen_fang_cheng, x0, y0, h, n_steps)

    # 计算精确解
    y_exact = exact_solution(x_num)

    # 计算误差
    loss = y_num - y_exact

    # 输出结果
    print("Euler法数值解与精确解比较:")
    print("x\t\tEuler解\t\t精确解\t\t误差")
    print("-" * 55)

    for i in range(len(x_num)):
        print(f"{x_num[i]:.1f}\t\t{y_num[i]:.6f}\t\t{y_exact[i]:.6f}\t\t{loss[i]:.6f}")

    print(f"\n最大绝对误差: {np.max(np.abs(loss)):.6f}")
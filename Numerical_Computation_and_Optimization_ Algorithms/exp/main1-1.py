import math
#黄金分割比0.618
rho = (math.sqrt(5) - 1) / 2

#原函数
def fun(x):
    return 8 * math.exp(1 - x) + 7 * math.log(x)


def loop(a, b, tolerance):

#计数器
    iterations = 0

#压缩区间小于目标区间时跳出
    while (b - a) > tolerance:
        iterations += 1

        x1 = b - rho * (b - a)
        x2 = a + rho * (b - a)

        f1 = fun(x1)
        f2 = fun(x2)

        print(f"第 {iterations} 次迭代: [{a:.4f}, {b:.4f}], "
              f"x1={x1:.4f}, x2={x2:.4f}, "
              f"f1={f1:.4f}, f2={f2:.4f}")

        if f1 < f2:
            b = x2
            #极小值点在[a, x2]
        else:
            a = x1
            #极小值点在[x1, b]

    return (a, b)


if __name__ == "__main__":
    a, b, t = 1.0, 3.0, 0.23
    final = loop(a, b, t)
    print(f"\n最终区间: {final}")
    print(f"区间长度: {final[1] - final[0]:.4f}")
    print(f"近似解: {(final[0] + final[1]) / 2:.4f}")
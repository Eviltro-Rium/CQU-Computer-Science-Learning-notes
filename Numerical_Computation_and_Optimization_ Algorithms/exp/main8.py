import numpy as np
import matplotlib.pyplot as plt
import random
import math


def get_cities():
    cities = [
        (16.47, 96.10), (16.47, 94.44), (20.09, 92.54), (22.39, 93.37), (25.23, 97.24),
        (17.20, 96.29), (16.30, 97.38), (14.05, 98.12), (16.53, 97.38), (21.52, 95.59)
    ]
    return cities


def calculate_distance(path, cities):
    """计算路径总距离"""
    total = 0
    n = len(path)
    for i in range(n):
        x1, y1 = cities[path[i]]
        x2, y2 = cities[path[(i + 1) % n]]
        total += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total


def get_neighbor(path):
    """生成邻域解 - 使用2-opt操作"""
    neighbor = path.copy()
    i, j = sorted(random.sample(range(len(path)), 2))
    neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
    return neighbor


def simulated_annealing_tsp(cities):
    """精简版模拟退火算法"""
    n = len(cities)

    # 初始解
    current_path = list(range(n))
    random.shuffle(current_path)
    current_dist = calculate_distance(current_path, cities)

    # 记录最优解
    best_path = current_path.copy()
    best_dist = current_dist

    # 退火参数
    temp = 1000
    cooling_rate = 0.995
    min_temp = 1e-5

    distances = []

    iteration = 0
    while temp > min_temp and iteration < 5000:
        # 生成邻域解
        neighbor_path = get_neighbor(current_path)
        neighbor_dist = calculate_distance(neighbor_path, cities)

        # 计算能量差
        delta = neighbor_dist - current_dist

        # 接受准则
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_path = neighbor_path
            current_dist = neighbor_dist

            # 更新最优解
            if current_dist < best_dist:
                best_path = current_path.copy()
                best_dist = current_dist
                print(f"Iter {iteration}: New best = {best_dist:.2f}, Temp = {temp:.2f}")

        distances.append(current_dist)
        temp *= cooling_rate
        iteration += 1

    return best_path, best_dist, distances


def plot_results(cities, best_path, distances):
    """绘制结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 距离进化曲线
    ax1.plot(distances, 'b-', alpha=0.7)
    ax1.axhline(y=min(distances), color='r', linestyle='--',
                label=f'Best: {min(distances):.2f}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Distance')
    ax1.set_title('Distance Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 最优路径图
    path_coords = [cities[i] for i in best_path] + [cities[best_path[0]]]
    x_coords = [c[0] for c in path_coords]
    y_coords = [c[1] for c in path_coords]

    ax2.plot(x_coords, y_coords, 'bo-', linewidth=2, markersize=8)
    ax2.scatter(x_coords, y_coords, c='red', s=100)

    # 标注城市
    for i, (x, y) in enumerate(cities):
        ax2.text(x, y, f' {i + 1}', fontsize=12)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Optimal Route: {calculate_distance(best_path, cities):.2f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 主程序
def main():
    cities = get_cities()
    print(f"Solving TSP for {len(cities)} cities")

    # 运行模拟退火
    best_path, best_dist, history = simulated_annealing_tsp(cities)

    print(f"\nOptimal path: {[i + 1 for i in best_path]}")
    print(f"Total distance: {best_dist:.2f}")

    # 绘制结果
    plot_results(cities, best_path, history)


if __name__ == "__main__":
    main()
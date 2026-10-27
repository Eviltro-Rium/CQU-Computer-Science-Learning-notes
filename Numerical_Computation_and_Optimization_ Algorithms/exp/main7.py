import numpy as np
import matplotlib.pyplot as plt
import random


def get_cities():
    cities = [
        (16.47, 96.10), (16.47, 94.44), (20.09, 92.54), (22.39, 93.37), (25.23, 97.24),
        (17.20, 96.29), (16.30, 97.38), (14.05, 98.12), (16.53, 97.38), (21.52, 95.59)
    ]
    return cities


#距离计算
def calculate_distance(path, cities):
    total_distance = 0
    num_cities = len(path)

    for i in range(num_cities):
        city1 = path[i]
        city2 = path[(i + 1) % num_cities]

        x1, y1 = cities[city1]
        x2, y2 = cities[city2]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance

    return total_distance


def initialize_population(pop_size, num_cities):
    """初始化种群 - 随机生成路径"""
    population = []
    for _ in range(pop_size):
        individual = list(range(num_cities))
        random.shuffle(individual)
        population.append(individual)
    return population


def crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))

    # 从parent1复制中间段
    child = [-1] * size
    child[point1:point2] = parent1[point1:point2]

    # 从parent2填充剩余位置
    pointer = point2
    for gene in parent2:
        if gene not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = gene
            pointer += 1

    return child


def mutate(individual, mutation_rate):
    """交换变异"""
    if random.random() < mutation_rate:
        point1, point2 = random.sample(range(len(individual)), 2)
        individual[point1], individual[point2] = individual[point2], individual[point1]
    return individual


def genetic_algorithm_tsp(cities, pop_size=50, generations=200):
    """遗传算法求解TSP"""
    num_cities = len(cities)

    # 初始化种群
    population = initialize_population(pop_size, num_cities)
    best_distance = float('inf')
    best_path = None
    best_distances = []  # 记录每代最佳距离

    for generation in range(generations):
        # 评估种群
        distances = [calculate_distance(ind, cities) for ind in population]

        # 更新最佳解
        current_best = min(distances)
        if current_best < best_distance:
            best_distance = current_best
            best_path = population[distances.index(current_best)].copy()

        best_distances.append(best_distance)

        # 选择（简化版：选择前50%的个体）
        sorted_population = [x for _, x in sorted(zip(distances, population))]
        selected = sorted_population[:pop_size // 2]

        # 交叉和变异生成新种群
        new_population = selected.copy()  # 保留优秀个体

        while len(new_population) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, 0.1)
            new_population.append(child)

        population = new_population

        # 每10代打印进度
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Distance = {best_distance:.2f}")

    return best_path, best_distance, best_distances


def plot_results(cities, best_path, best_distances):
    """绘制结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制进化曲线
    ax1.plot(best_distances, 'b-', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Distance')
    ax1.set_title('Evolution of Best Path Distance')
    ax1.grid(True, alpha=0.3)

    path_coords = [cities[i] for i in best_path] + [cities[best_path[0]]]
    x_coords = [coord[0] for coord in path_coords]
    y_coords = [coord[1] for coord in path_coords]

    ax2.plot(x_coords, y_coords, 'bo-', linewidth=2, markersize=8)
    ax2.scatter(x_coords, y_coords, c='red', s=100, zorder=5)

    for i, (x, y) in enumerate(cities):
        ax2.text(x, y, f' {i + 1}', fontsize=12, ha='left', va='bottom')

    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_title(f'Optimal TSP Route\nTotal Distance: {calculate_distance(best_path, cities):.2f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 主程序
def main():
    cities = get_cities()
    print(f"处理 {len(cities)} 个城市的TSP问题")

    # 运行遗传算法
    best_path, best_distance, history = genetic_algorithm_tsp(cities)

    print(f"\n最优路径: {[i + 1 for i in best_path]}")
    print(f"总距离: {best_distance:.2f}")

    plot_results(cities, best_path, history)


if __name__ == "__main__":
    main()
"""
遗传算法求解函数最大值示例

使用经典遗传算法寻找函数的最大值
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GeneticAlgorithm:
    def __init__(self, fitness_func, bounds, pop_size=50, mutation_rate=0.1, crossover_rate=0.8, generations=100):
        """
        初始化遗传算法
        
        参数:
            fitness_func: 适应度函数
            bounds: 搜索空间的边界，格式为[(x_min, x_max), (y_min, y_max)]
            pop_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            generations: 迭代代数
        """
        self.fitness_func = fitness_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.n_vars = len(bounds)
        
        # 初始化种群
        self.population = self._init_population()
        self.best_individuals = []
        
    def _init_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.pop_size):
            individual = []
            for i in range(self.n_vars):
                gene = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
                individual.append(gene)
            population.append(individual)
        return np.array(population)
    
    def _calculate_fitness(self, individual):
        """计算个体的适应度"""
        return self.fitness_func(*individual)
    
    def _select_parents(self, fitnesses):
        """使用轮盘赌选择父代"""
        fitnesses = np.maximum(fitnesses, 0)  # 确保适应度非负
        total_fitness = np.sum(fitnesses)
        
        if total_fitness == 0:
            # 如果所有适应度都为0，则等概率选择
            probs = np.ones(len(fitnesses)) / len(fitnesses)
        else:
            probs = fitnesses / total_fitness
            
        # 选择两个父代索引
        parent_idxs = np.random.choice(len(fitnesses), size=2, p=probs)
        return parent_idxs
    
    def _crossover(self, parent1, parent2):
        """执行交叉操作"""
        if np.random.rand() > self.crossover_rate:
            return parent1, parent2
        
        # 单点交叉
        crossover_point = np.random.randint(1, self.n_vars)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        return child1, child2
    
    def _mutate(self, individual):
        """执行变异操作"""
        mutated = individual.copy()
        for i in range(self.n_vars):
            if np.random.rand() < self.mutation_rate:
                # 在指定范围内进行变异
                mutated[i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
        return mutated
    
    def run(self, verbose=True):
        """运行遗传算法"""
        best_fitness_history = []
        
        for gen in range(self.generations):
            # 计算适应度
            fitnesses = np.array([self._calculate_fitness(ind) for ind in self.population])
            
            # 记录当代最佳个体
            best_idx = np.argmax(fitnesses)
            best_individual = self.population[best_idx]
            best_fitness = fitnesses[best_idx]
            self.best_individuals.append((best_individual, best_fitness))
            best_fitness_history.append(best_fitness)
            
            if verbose and gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {best_fitness}, Best individual = {best_individual}")
            
            # 创建新一代种群
            new_population = []
            
            # 精英保留策略 - 保留最佳个体
            new_population.append(best_individual)
            
            # 生成剩余个体
            while len(new_population) < self.pop_size:
                # 选择父代
                parent_idxs = self._select_parents(fitnesses)
                parent1, parent2 = self.population[parent_idxs[0]], self.population[parent_idxs[1]]
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                # 添加到新种群
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # 更新种群
            self.population = np.array(new_population)
        
        # 返回最佳个体和适应度历史
        best_individual, best_fitness = max(self.best_individuals, key=lambda x: x[1])
        return best_individual, best_fitness, best_fitness_history

# 示例：寻找函数 f(x, y) = sin(x) * cos(y) 在指定范围内的最大值
def example_function(x, y):
    return np.sin(x) * np.cos(y)

def run_example():
    """运行遗传算法示例"""
    bounds = [(-5, 5), (-5, 5)]  # x和y的取值范围
    
    # 初始化并运行遗传算法
    ga = GeneticAlgorithm(
        fitness_func=example_function,
        bounds=bounds,
        pop_size=100,
        mutation_rate=0.2,
        crossover_rate=0.8,
        generations=100
    )
    
    best_solution, best_fitness, fitness_history = ga.run()
    
    print("\n最优解:")
    print(f"x = {best_solution[0]}, y = {best_solution[1]}")
    print(f"最大值 = {best_fitness}")
    
    # 可视化结果
    visualize_results(best_solution, fitness_history, ga.best_individuals)
    
    return best_solution, best_fitness

def visualize_results(best_solution, fitness_history, best_individuals):
    """可视化算法结果"""
    # 绘制适应度变化曲线
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history)
    plt.title('遗传算法收敛过程')
    plt.xlabel('迭代次数')
    plt.ylabel('最佳适应度')
    plt.grid(True)
    plt.savefig('genetic_algorithm_convergence.png')
    
    # 绘制目标函数和最优解
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = example_function(X, Y)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.scatter(best_solution[0], best_solution[1], example_function(*best_solution), 
               color='red', s=100, marker='*', label='最优解')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    ax.set_title('目标函数与最优解')
    fig.colorbar(surface, ax=ax, shrink=0.5)
    plt.legend()
    plt.savefig('genetic_algorithm_solution.png')
    
    # 绘制种群迁移动画 (只保存前20代的前50个最佳个体)
    plt.figure(figsize=(10, 8))
    individuals = [ind[0] for ind in best_individuals[:20]]
    
    def update(frame):
        plt.clf()
        plt.contourf(X, Y, Z, 50, cmap='viridis')
        plt.colorbar()
        
        x_vals = [ind[0] for ind in individuals[:frame+1]]
        y_vals = [ind[1] for ind in individuals[:frame+1]]
        
        plt.scatter(x_vals, y_vals, color='white', edgecolor='black')
        if frame > 0:
            plt.plot([x_vals[i] for i in range(frame+1)], 
                     [y_vals[i] for i in range(frame+1)], 
                     'r-', alpha=0.6)
        
        plt.title(f'种群迁移过程 - 第{frame+1}代')
        plt.xlabel('X')
        plt.ylabel('Y')
    
    ani = FuncAnimation(plt.gcf(), update, frames=min(20, len(individuals)), interval=500)
    plt.tight_layout()
    
if __name__ == "__main__":
    run_example()
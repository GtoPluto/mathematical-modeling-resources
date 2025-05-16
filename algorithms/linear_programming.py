"""
线性规划问题求解示例

使用PuLP库实现线性规划模型的构建与求解
"""

import numpy as np
import pulp as pl
import matplotlib.pyplot as plt

def solve_linear_programming_example():
    """
    解决一个简单的线性规划问题示例:
    
    最大化: z = 3x + 5y
    约束条件:
    2x + 3y <= 12
    -x + y <= 3
    x >= 0
    y >= 0
    """
    # 创建问题实例
    problem = pl.LpProblem("简单线性规划示例", pl.LpMaximize)
    
    # 定义决策变量
    x = pl.LpVariable("x", lowBound=0)
    y = pl.LpVariable("y", lowBound=0)
    
    # 设置目标函数
    problem += 3 * x + 5 * y, "目标函数"
    
    # 添加约束条件
    problem += 2 * x + 3 * y <= 12, "约束条件1"
    problem += -x + y <= 3, "约束条件2"
    
    # 求解问题
    problem.solve()
    
    # 输出结果
    print("状态:", pl.LpStatus[problem.status])
    print("最优解:")
    print(f"x = {pl.value(x)}")
    print(f"y = {pl.value(y)}")
    print(f"目标函数值 = {pl.value(problem.objective)}")
    
    # 可视化结果
    plot_solution(pl.value(x), pl.value(y))
    
    return {
        "status": pl.LpStatus[problem.status],
        "x": pl.value(x),
        "y": pl.value(y),
        "objective": pl.value(problem.objective)
    }

def plot_solution(x_optimal, y_optimal):
    """绘制可行域和最优解"""
    # 创建数据点
    x = np.linspace(0, 6, 100)
    
    # 约束条件
    y1 = (12 - 2*x) / 3  # 2x + 3y = 12
    y2 = x + 3  # -x + y = 3
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制约束线
    plt.plot(x, y1, 'b-', label='2x + 3y = 12')
    plt.plot(x, y2, 'g-', label='-x + y = 3')
    plt.axvline(x=0, color='r', linestyle='-', label='x = 0')
    plt.axhline(y=0, color='m', linestyle='-', label='y = 0')
    
    # 填充可行域
    y1_valid = np.maximum(0, y1)
    y2_valid = np.maximum(0, y2)
    y_min = np.minimum(y1_valid, y2_valid)
    plt.fill_between(x, 0, y_min, where=(x >= 0) & (y_min >= 0), alpha=0.2, color='gray')
    
    # 标记最优解
    plt.plot(x_optimal, y_optimal, 'ro', markersize=10, label=f'最优解 ({x_optimal:.2f}, {y_optimal:.2f})')
    
    # 设置图表属性
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('线性规划问题的可行域和最优解')
    plt.legend()
    plt.savefig('linear_programming_solution.png')
    
if __name__ == "__main__":
    solve_linear_programming_example()
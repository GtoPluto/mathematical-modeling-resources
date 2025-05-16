"""
流行病传播模型：SIR模型案例研究

经典SIR模型的实现与仿真，用于模拟疾病在人群中的传播过程
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from matplotlib.animation import FuncAnimation

class SIRModel:
    """SIR (Susceptible-Infected-Recovered) 模型实现"""
    
    def __init__(self, beta, gamma, population):
        """
        初始化SIR模型
        
        参数:
            beta: 传染率 - 易感人群转变为感染者的概率系数
            gamma: 恢复率 - 感染者转变为康复者的概率系数
            population: 总人口数
        """
        self.beta = beta
        self.gamma = gamma
        self.population = population
        
        # 计算基本再生数R0 (流行病学重要指标)
        self.R0 = beta / gamma
        
    def model_equations(self, t, y):
        """
        SIR模型的微分方程组
        
        参数:
            t: 时间点
            y: 系统状态 [S, I, R]
                S: 易感人群数量
                I: 感染人群数量
                R: 康复人群数量
                
        返回:
            dy/dt: 各状态的变化率
        """
        S, I, R = y
        
        # SIR模型的微分方程:
        # dS/dt = -beta * S * I / N
        # dI/dt = beta * S * I / N - gamma * I
        # dR/dt = gamma * I
        
        dSdt = -self.beta * S * I / self.population
        dIdt = self.beta * S * I / self.population - self.gamma * I
        dRdt = self.gamma * I
        
        return [dSdt, dIdt, dRdt]
    
    def simulate(self, initial_state, t_span, t_eval=None):
        """
        模拟SIR模型随时间的演化
        
        参数:
            initial_state: 初始状态 [S0, I0, R0]
            t_span: 模拟的时间范围 [t_start, t_end]
            t_eval: 需要输出结果的时间点
            
        返回:
            t: 时间点
            y: 对应时间点的系统状态
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
            
        solution = solve_ivp(
            self.model_equations,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='RK45'
        )
        
        return solution.t, solution.y
    
    def plot_simulation(self, t, y, title=None, save_path=None):
        """
        绘制SIR模型的模拟结果
        
        参数:
            t: 时间点
            y: 系统状态
            title: 图表标题
            save_path: 保存图表的路径
        """
        S, I, R = y
        
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, 'b-', label='易感人群 (S)')
        plt.plot(t, I, 'r-', label='感染人群 (I)')
        plt.plot(t, R, 'g-', label='康复人群 (R)')
        
        plt.grid(True)
        plt.xlabel('时间 (天)')
        plt.ylabel('人口数')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'SIR模型模拟 (β={self.beta:.4f}, γ={self.gamma:.4f}, R₀={self.R0:.2f})')
            
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.tight_layout()
        
    def create_animation(self, t, y, save_path=None):
        """
        创建SIR模型动态演变的动画
        
        参数:
            t: 时间点
            y: 系统状态
            save_path: 保存动画的路径
        """
        S, I, R = y
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置坐标轴范围
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0, self.population * 1.1)
        
        # 创建三条线
        line_S, = ax.plot([], [], 'b-', label='易感人群 (S)')
        line_I, = ax.plot([], [], 'r-', label='感染人群 (I)')
        line_R, = ax.plot([], [], 'g-', label='康复人群 (R)')
        
        # 设置标题和标签
        ax.set_title(f'SIR模型动态演变 (R₀={self.R0:.2f})')
        ax.set_xlabel('时间 (天)')
        ax.set_ylabel('人口数')
        ax.grid(True)
        ax.legend()
        
        # 添加文本标签显示当前时间
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def init():
            line_S.set_data([], [])
            line_I.set_data([], [])
            line_R.set_data([], [])
            time_text.set_text('')
            return line_S, line_I, line_R, time_text
        
        def update(frame):
            # 更新每条线的数据
            line_S.set_data(t[:frame], S[:frame])
            line_I.set_data(t[:frame], I[:frame])
            line_R.set_data(t[:frame], R[:frame])
            time_text.set_text(f'天数: {t[frame-1]:.1f}')
            return line_S, line_I, line_R, time_text
        
        # 创建动画
        ani = FuncAnimation(fig, update, frames=len(t), 
                           init_func=init, blit=True, interval=50)
        
        plt.tight_layout()
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=30)
            
        return ani
            
    def find_epidemic_peak(self, t, y):
        """
        找出疫情高峰点
        
        参数:
            t: 时间点
            y: 系统状态
            
        返回:
            peak_time: 疫情高峰时间
            peak_infected: 高峰时感染人数
        """
        S, I, R = y
        peak_idx = np.argmax(I)
        peak_time = t[peak_idx]
        peak_infected = I[peak_idx]
        
        return peak_time, peak_infected
    
    def estimate_final_size(self, t, y):
        """
        估计疫情最终规模
        
        参数:
            t: 时间点
            y: 系统状态
            
        返回:
            final_infected_percent: 最终感染比例
        """
        S, I, R = y
        final_susceptible = S[-1]
        final_recovered = R[-1]
        
        # 最终感染总人数（包括已康复）
        final_infected_total = final_recovered
        final_infected_percent = final_infected_total / self.population * 100
        
        return final_infected_percent
    
    def parameter_sensitivity_analysis(self, initial_state, t_span, beta_range, gamma_range):
        """
        参数敏感性分析
        
        参数:
            initial_state: 初始状态
            t_span: 时间范围
            beta_range: 传染率范围
            gamma_range: 恢复率范围
            
        返回:
            results: 参数敏感性分析结果
        """
        results = []
        
        for beta in beta_range:
            for gamma in gamma_range:
                # 创建新的模型实例
                temp_model = SIRModel(beta, gamma, self.population)
                
                # 运行模拟
                t, y = temp_model.simulate(initial_state, t_span)
                
                # 获取结果指标
                peak_time, peak_infected = temp_model.find_epidemic_peak(t, y)
                final_size = temp_model.estimate_final_size(t, y)
                
                # 存储结果
                results.append({
                    'beta': beta,
                    'gamma': gamma,
                    'R0': beta / gamma,
                    'peak_time': peak_time,
                    'peak_infected': peak_infected,
                    'peak_percent': peak_infected / self.population * 100,
                    'final_size_percent': final_size
                })
        
        return pd.DataFrame(results)
    
    def plot_sensitivity_analysis(self, results, metric='final_size_percent', save_path=None):
        """
        绘制参数敏感性分析结果
        
        参数:
            results: 敏感性分析结果
            metric: 要绘制的指标
            save_path: 保存图表的路径
        """
        # 准备数据
        beta_values = sorted(results['beta'].unique())
        gamma_values = sorted(results['gamma'].unique())
        
        # 创建网格
        X, Y = np.meshgrid(gamma_values, beta_values)
        Z = np.zeros(X.shape)
        
        # 填充Z值
        for i, beta in enumerate(beta_values):
            for j, gamma in enumerate(gamma_values):
                mask = (results['beta'] == beta) & (results['gamma'] == gamma)
                Z[i, j] = results[mask][metric].values[0]
        
        # 绘制热图
        plt.figure(figsize=(12, 8))
        
        # 绘制等高线
        contour = plt.contourf(X, Y, Z, 20, cmap='viridis')
        plt.colorbar(contour, label=f'{metric}')
        
        # 添加R0=1的分隔线
        R0_1_gamma = np.linspace(min(gamma_values), max(gamma_values), 100)
        R0_1_beta = R0_1_gamma  # 当R0=1时，beta = gamma
        plt.plot(R0_1_gamma, R0_1_beta, 'r--', label='R₀ = 1')
        
        plt.xlabel('恢复率 (γ)')
        plt.ylabel('传染率 (β)')
        plt.title(f'参数敏感性分析 - {metric}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.tight_layout()

def example_covid19():
    """COVID-19简化版模型示例"""
    # 初始参数设置
    population = 10000
    initial_infected = 10
    initial_recovered = 0
    initial_susceptible = population - initial_infected - initial_recovered
    
    # 初始状态
    initial_state = [initial_susceptible, initial_infected, initial_recovered]
    
    # 设置传染率和恢复率
    # 这些值是简化的，真实COVID-19的参数更复杂
    beta = 0.3   # 传染率
    gamma = 0.1  # 恢复率 (假设平均恢复时间为10天)
    
    # 创建模型
    model = SIRModel(beta, gamma, population)
    
    # 打印模型参数
    print(f"模型参数:")
    print(f"总人口: {population}")
    print(f"初始感染人数: {initial_infected}")
    print(f"传染率 (β): {beta}")
    print(f"恢复率 (γ): {gamma}")
    print(f"基本再生数 (R₀): {model.R0:.2f}")
    
    # 模拟200天
    t_span = [0, 200]
    t, y = model.simulate(initial_state, t_span)
    
    # 绘制结果
    model.plot_simulation(t, y, title="COVID-19 SIR模型简化版", 
                         save_path="covid19_sir_model.png")
    
    # 查找疫情高峰
    peak_time, peak_infected = model.find_epidemic_peak(t, y)
    peak_percent = peak_infected / population * 100
    
    # 估计最终规模
    final_size_percent = model.estimate_final_size(t, y)
    
    print("\n模型预测结果:")
    print(f"疫情高峰出现在第 {peak_time:.1f} 天")
    print(f"高峰时感染人数: {int(peak_infected)} 人 ({peak_percent:.2f}%)")
    print(f"最终感染总人数比例: {final_size_percent:.2f}%")
    
    # 创建动画 (可选)
    # model.create_animation(t, y, save_path="covid19_sir_animation.gif")
    
    # 参数敏感性分析
    beta_range = np.linspace(0.1, 0.5, 5)
    gamma_range = np.linspace(0.05, 0.25, 5)
    
    sensitivity_results = model.parameter_sensitivity_analysis(
        initial_state, t_span, beta_range, gamma_range
    )
    
    # 绘制敏感性分析结果
    model.plot_sensitivity_analysis(sensitivity_results, 
                                   metric='peak_percent',
                                   save_path="sir_sensitivity_peak.png")
    
    model.plot_sensitivity_analysis(sensitivity_results, 
                                   metric='final_size_percent',
                                   save_path="sir_sensitivity_final.png")
    
    return model, t, y, sensitivity_results

if __name__ == "__main__":
    example_covid19()
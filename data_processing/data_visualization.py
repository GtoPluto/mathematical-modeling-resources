"""
数据可视化示例

展示常用的数据可视化方法和技巧
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_sample_data():
    """加载示例数据集"""
    # 使用鸢尾花数据集
    iris = load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])
    
    # 将目标变量转换为类别
    df['species'] = df['target'].map({
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    })
    
    return df

def basic_visualizations(df):
    """基础可视化方法"""
    # 设置Seaborn样式
    sns.set(style="whitegrid")
    
    # 1. 散点图矩阵
    plt.figure(figsize=(12, 10))
    scatter_matrix = sns.pairplot(df, 
                                  hue='species', 
                                  palette='viridis',
                                  diag_kind='kde',
                                  markers=['o', 's', 'D'])
    plt.suptitle('鸢尾花数据集特征散点图矩阵', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('iris_pairplot.png')
    
    # 2. 箱线图
    plt.figure(figsize=(14, 7))
    features = iris['feature_names']
    
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='species', y=feature, data=df, palette='Set3')
        plt.title(f'{feature}按种类分布')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('iris_boxplots.png')
    
    # 3. 小提琴图
    plt.figure(figsize=(14, 7))
    
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i+1)
        sns.violinplot(x='species', y=feature, data=df, palette='Set2',
                      inner='quartile')
        plt.title(f'{feature}的小提琴图')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('iris_violinplots.png')
    
    # 4. 热图 - 相关性矩阵
    plt.figure(figsize=(10, 8))
    correlation = df[features].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
               mask=mask, vmin=-1, vmax=1, center=0,
               square=True, linewidths=.5)
    
    plt.title('特征相关性矩阵', fontsize=15)
    plt.tight_layout()
    plt.savefig('iris_correlation_heatmap.png')

def advanced_visualizations(df):
    """高级可视化方法"""
    features = iris['feature_names']
    X = df[features].values
    y = df['target'].values
    
    # 1. PCA降维可视化
    # 标准化数据
    X_std = StandardScaler().fit_transform(X)
    
    # 应用PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # 创建结果DataFrame
    pca_df = pd.DataFrame(data=X_pca, columns=['主成分1', '主成分2'])
    pca_df['species'] = df['species']
    
    # 绘制PCA结果
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='主成分1', y='主成分2', hue='species', 
                   data=pca_df, palette='viridis', s=100, alpha=0.8)
    
    # 添加解释方差比例
    explained_variance = pca.explained_variance_ratio_
    plt.xlabel(f'主成分1 (解释方差: {explained_variance[0]:.2%})', fontsize=12)
    plt.ylabel(f'主成分2 (解释方差: {explained_variance[1]:.2%})', fontsize=12)
    
    plt.title('鸢尾花数据PCA降维可视化', fontsize=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('iris_pca.png')
    
    # 2. 绘制特征重要性
    # 计算特征在主成分中的权重
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=features
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pca_components, annot=True, cmap='YlGnBu')
    plt.title('PCA特征权重', fontsize=15)
    plt.tight_layout()
    plt.savefig('pca_feature_weights.png')
    
    # 3. 3D散点图
    from mpl_toolkits.mplot3d import Axes3D
    
    # 选择三个特征
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置点的颜色和标记
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    markers = {'setosa': 'o', 'versicolor': '^', 'virginica': 's'}
    
    # 绘制散点图
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        ax.scatter(subset[features[0]], 
                  subset[features[1]], 
                  subset[features[2]],
                  c=colors[species],
                  marker=markers[species],
                  s=60,
                  alpha=0.8,
                  label=species)
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.set_title('鸢尾花数据集三维可视化', fontsize=15)
    ax.legend()
    
    # 设置观察角度
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig('iris_3d_scatter.png')

def interactive_visualization_example():
    """交互式可视化示例说明"""
    instructions = """
    # 交互式可视化

    除了静态图像外，数学建模中常用交互式可视化工具可以更好地探索数据。
    推荐以下交互式可视化库：

    ## Plotly
    ```python
    import plotly.express as px
    import plotly.graph_objects as go
    
    # 使用Plotly Express创建交互式散点图
    fig = px.scatter(df, x='sepal_length', y='sepal_width', 
                    color='species', hover_data=['petal_length', 'petal_width'])
    fig.show()
    ```

    ## Bokeh
    ```python
    from bokeh.plotting import figure, show
    from bokeh.palettes import Category10
    from bokeh.models import ColumnDataSource, HoverTool
    
    # 创建数据源
    source = ColumnDataSource(df)
    
    # 创建图形
    p = figure(title="鸢尾花数据可视化", 
              x_axis_label='萼片长度', 
              y_axis_label='萼片宽度')
    
    # 添加悬停工具
    hover = HoverTool(tooltips=[
        ("种类", "@species"),
        ("萼片长度", "@sepal_length"),
        ("萼片宽度", "@sepal_width"),
        ("花瓣长度", "@petal_length"),
        ("花瓣宽度", "@petal_width")
    ])
    p.add_tools(hover)
    
    # 绘制散点图
    for i, species in enumerate(df['species'].unique()):
        source_species = ColumnDataSource(df[df['species'] == species])
        p.circle(x='sepal_length', y='sepal_width', source=source_species,
                color=Category10[3][i], legend_label=species, size=8)
    
    show(p)
    ```

    ## 动态图表
    对于时间序列数据，可以创建动态演变的图表：
    
    ```python
    from matplotlib.animation import FuncAnimation
    
    # 创建动画函数
    def animate(i):
        plt.cla()  # 清除当前轴
        plt.scatter(df['sepal_length'][:i+1], df['sepal_width'][:i+1], 
                   c=df['target'][:i+1])
        plt.title(f'数据点: {i+1}/{len(df)}')
    
    # 创建动画
    ani = FuncAnimation(plt.gcf(), animate, frames=len(df), interval=100)
    ```
    """
    
    with open('interactive_visualization_guide.md', 'w') as f:
        f.write(instructions)
    
    return instructions

def run_visualization_demo():
    """运行所有可视化示例"""
    print("加载数据...")
    df = load_sample_data()
    
    print("生成基础可视化...")
    basic_visualizations(df)
    
    print("生成高级可视化...")
    advanced_visualizations(df)
    
    print("生成交互式可视化指南...")
    interactive_visualization_example()
    
    print("所有可视化已完成!")
    
if __name__ == "__main__":
    run_visualization_demo()
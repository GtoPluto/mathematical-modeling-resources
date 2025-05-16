# 安装指南

本仓库包含多个数学建模相关的Python实现，需要安装一些依赖库才能正常运行。

## 系统要求

- Python 3.8 或更高版本
- pip (Python包管理器)

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/GtoPluto/mathematical-modeling-resources.git
cd mathematical-modeling-resources
```

### 2. 创建虚拟环境（推荐）

使用虚拟环境可以避免依赖冲突问题。

在Windows上:
```bash
python -m venv venv
venv\Scripts\activate
```

在macOS/Linux上:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 验证安装

安装完成后，可以运行示例代码来验证环境配置是否正确：

```bash
python algorithms/linear_programming.py
```

如果没有错误信息并且成功输出结果，说明环境配置成功。

## 常见问题

### 1. 安装PuLP时出错

如果在安装PuLP库时遇到问题，可能需要先安装其依赖的求解器：

```bash
# 在Windows上
pip install pulp
# 然后手动下载CBC求解器，或使用conda安装
conda install -c conda-forge coincbc
```

### 2. Matplotlib显示问题

如果运行代码时Matplotlib无法显示图形，可能是缺少GUI后端：

```bash
# 对于Debian/Ubuntu
sudo apt-get install python3-tk

# 对于Windows
# 通常不需要额外安装，如有问题尝试重新安装matplotlib
pip uninstall matplotlib
pip install matplotlib
```

### 3. 其他依赖问题

如果遇到其他依赖问题，请检查您的Python版本是否满足要求，并尝试单独安装有问题的包：

```bash
pip install <问题包名>
```

## 更新代码

要获取最新版本的代码，可以执行：

```bash
git pull origin main
```

## 联系方式

如果在安装过程中遇到任何问题，请通过GitHub Issues反馈。
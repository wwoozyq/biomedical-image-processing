# 快速开始

## 安装依赖

```bash
pip install numpy matplotlib jupyter
```

## 运行实验

```bash
cd lab/lab1
jupyter notebook lab1.ipynb
```

## 常见问题

### 中文显示乱码

**macOS:**
```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
```

**Windows:**
```python
plt.rcParams['font.sans-serif'] = ['SimHei']
```

**Linux:**
```python
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
```

# 亥姆霍兹线圈优化系统 - 快速使用指南

## 🚀 立即开始

### 1. 环境准备
```bash
pip install numpy matplotlib scipy seaborn pandas
```

### 2. 运行完整演示（推荐）
```bash
python fixed_demo.py
```
**这将运行完整的优化演示，包括算法对比、磁场可视化、HTML报告生成**

### 3. 运行简单测试
```bash
python simple_test.py
```
**快速验证系统功能是否正常**

## 📊 演示结果

运行 `fixed_demo.py` 后，您将看到：

### 优化结果
- **DE算法**表现最优 (适应度: 0.002859)
- **PSO算法**执行最快 (25.284s)
- 所有算法都达到**100%成功率**

### 生成的文件
- `helmholtz_algorithm_comparison.png` - 算法对比图
- `helmholtz_field_analysis.png` - 磁场分析图
- `helmholtz_optimization_report.html` - HTML优化报告

## 🎯 快速使用示例

```python
from helmholtz_objectives import create_helmholtz_bounds, FieldUniformityObjective
from algorithm_fusion import AlgorithmFusionOptimizer

# 创建优化问题
bounds = create_helmholtz_bounds(1)  # 1对线圈
objective = FieldUniformityObjective(bounds, target_field=0.1)

# 使用融合算法优化
optimizer = AlgorithmFusionOptimizer(
    bounds, objective, max_iterations=30, population_size=20, fusion_strategy="adaptive")

solution, fitness = optimizer.optimize()
print(f"最优解: {solution}")
print(f"最优适应度: {fitness}")
```

## 🔧 算法选择建议

- **追求最优解**: 使用 **DE算法**
- **追求速度**: 使用 **PSO算法**
- **复杂问题**: 使用 **融合算法**

## 📈 可视化功能

```python
from visualization import HelmholtzVisualizer
from helmholtz_coil import create_optimized_helmholtz_system

# 创建可视化器
visualizer = HelmholtzVisualizer()

# 创建线圈系统
system = create_optimized_helmholtz_system([0.1, 1.0, 100, 0.1])

# 绘制磁场分布
visualizer.plot_field_distribution(system, title="磁场分布")

# 绘制磁场均匀性分析
visualizer.plot_field_uniformity(system, title="磁场均匀性分析")
```

## ⚠️ 注意事项

1. **首次运行**: 建议先运行 `simple_test.py` 验证环境
2. **计算时间**: 亥姆霍兹线圈计算较复杂，请耐心等待
3. **可视化**: 图表会自动保存为PNG文件
4. **中文显示**: 系统支持中文，如有乱码请检查字体

## 🎉 系统特色

- ✅ **算法融合**: 自动选择最佳算法组合
- ✅ **丰富可视化**: 磁场分布、算法对比、收敛分析
- ✅ **专业报告**: 自动生成HTML优化报告
- ✅ **高性能**: 并行计算支持
- ✅ **易用性**: 简洁的API接口

---

**开始使用**: `python fixed_demo.py`  
**快速测试**: `python simple_test.py`  
**详细文档**: 查看 `README.md`

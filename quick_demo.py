"""
快速演示程序
展示亥姆霍兹线圈优化系统的核心功能
"""

import numpy as np
import time
from helmholtz_objectives import create_helmholtz_bounds, FieldUniformityObjective
from algorithm_fusion import AlgorithmFusionOptimizer, compare_fusion_strategies
from optimization_algorithms import PSO, DE, GA, run_optimization_comparison

def quick_demo():
    """快速演示"""
    print("亥姆霍兹线圈优化系统 - 快速演示")
    print("=" * 50)
    
    # 创建参数边界（1对线圈）
    bounds = create_helmholtz_bounds(1)
    print(f"参数边界: {bounds}")
    print(f"参数说明: [半径(m), 电流(A), 匝数, 间距(m)]")
    
    # 创建目标函数
    objective = FieldUniformityObjective(bounds, target_field=0.1)
    print(f"目标: 磁场均匀性优化，目标磁场强度 0.1T")
    
    # 测试参数
    test_params = np.array([0.1, 1.0, 100, 0.1])
    print(f"\n测试参数: {test_params}")
    test_fitness = objective(test_params)
    print(f"测试适应度: {test_fitness:.6f}")
    
    print("\n" + "=" * 50)
    print("1. 单算法优化比较")
    print("=" * 50)
    
    # 比较不同算法
    algorithms = {
        'PSO': PSO(bounds, objective, max_iterations=20, population_size=15),
        'DE': DE(bounds, objective, max_iterations=20, population_size=15),
        'GA': GA(bounds, objective, max_iterations=20, population_size=15)
    }
    
    results = {}
    for name, algorithm in algorithms.items():
        print(f"\n运行 {name}...")
        start_time = time.time()
        solution, fitness = algorithm.optimize()
        execution_time = time.time() - start_time
        
        results[name] = {
            'solution': solution,
            'fitness': fitness,
            'time': execution_time
        }
        
        print(f"最优解: {solution}")
        print(f"最优适应度: {fitness:.6f}")
        print(f"执行时间: {execution_time:.2f}s")
    
    # 找出最佳算法
    best_algo = min(results.keys(), key=lambda x: results[x]['fitness'])
    print(f"\n最佳算法: {best_algo}")
    
    print("\n" + "=" * 50)
    print("2. 融合算法优化")
    print("=" * 50)
    
    # 测试融合算法
    fusion_strategies = ["adaptive", "parallel"]
    
    for strategy in fusion_strategies:
        print(f"\n测试 {strategy} 融合策略...")
        optimizer = AlgorithmFusionOptimizer(
            bounds, objective, max_iterations=15, population_size=10, fusion_strategy=strategy)
        
        start_time = time.time()
        solution, fitness = optimizer.optimize()
        execution_time = time.time() - start_time
        
        print(f"融合结果: {solution}")
        print(f"适应度: {fitness:.6f}")
        print(f"时间: {execution_time:.2f}s")
    
    print("\n" + "=" * 50)
    print("3. 性能总结")
    print("=" * 50)
    
    # 性能总结
    print("单算法性能:")
    for name, result in results.items():
        print(f"  {name}: 适应度={result['fitness']:.6f}, 时间={result['time']:.2f}s")
    
    print("\n融合算法优势:")
    print("  - 自动选择最佳算法")
    print("  - 并行计算提高效率")
    print("  - 自适应参数调整")
    print("  - 早停机制节省时间")
    
    print("\n优化建议:")
    print("  - 对于简单问题，使用单算法即可")
    print("  - 对于复杂问题，推荐使用融合算法")
    print("  - 根据计算资源选择并行策略")
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)

if __name__ == "__main__":
    quick_demo()

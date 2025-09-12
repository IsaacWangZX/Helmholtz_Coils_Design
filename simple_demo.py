"""
简单演示程序
使用简化的目标函数快速展示算法融合功能
"""

import numpy as np
import time
from algorithm_fusion import AlgorithmFusionOptimizer
from optimization_algorithms import PSO, DE, GA

def simple_objective(x):
    """简化的目标函数"""
    # Rastrigin函数 - 多峰函数，适合测试优化算法
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def demo_fusion_algorithms():
    """演示融合算法"""
    print("算法融合优化演示")
    print("=" * 50)
    
    # 创建参数边界
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    print(f"参数边界: {bounds}")
    print(f"目标函数: Rastrigin函数（多峰函数）")
    
    # 测试参数
    test_params = np.array([1.0, 1.0, 1.0])
    test_fitness = simple_objective(test_params)
    print(f"测试参数: {test_params}")
    print(f"测试适应度: {test_fitness:.6f}")
    
    print("\n" + "=" * 50)
    print("1. 单算法优化")
    print("=" * 50)
    
    # 比较不同算法
    algorithms = {
        'PSO': PSO(bounds, simple_objective, max_iterations=30, population_size=20),
        'DE': DE(bounds, simple_objective, max_iterations=30, population_size=20),
        'GA': GA(bounds, simple_objective, max_iterations=30, population_size=20)
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
    print(f"\n最佳单算法: {best_algo}")
    
    print("\n" + "=" * 50)
    print("2. 融合算法优化")
    print("=" * 50)
    
    # 测试不同融合策略
    fusion_strategies = ["adaptive", "parallel", "sequential"]
    
    fusion_results = {}
    for strategy in fusion_strategies:
        print(f"\n测试 {strategy} 融合策略...")
        optimizer = AlgorithmFusionOptimizer(
            bounds, simple_objective, max_iterations=20, population_size=15, fusion_strategy=strategy)
        
        start_time = time.time()
        solution, fitness = optimizer.optimize()
        execution_time = time.time() - start_time
        
        fusion_results[strategy] = {
            'solution': solution,
            'fitness': fitness,
            'time': execution_time
        }
        
        print(f"融合结果: {solution}")
        print(f"适应度: {fitness:.6f}")
        print(f"时间: {execution_time:.2f}s")
    
    # 找出最佳融合策略
    best_fusion = min(fusion_results.keys(), key=lambda x: fusion_results[x]['fitness'])
    print(f"\n最佳融合策略: {best_fusion}")
    
    print("\n" + "=" * 50)
    print("3. 性能对比分析")
    print("=" * 50)
    
    print("单算法性能:")
    for name, result in results.items():
        print(f"  {name}: 适应度={result['fitness']:.6f}, 时间={result['time']:.2f}s")
    
    print("\n融合算法性能:")
    for strategy, result in fusion_results.items():
        print(f"  {strategy}: 适应度={result['fitness']:.6f}, 时间={result['time']:.2f}s")
    
    # 计算性能提升
    best_single_fitness = min(results.values(), key=lambda x: x['fitness'])['fitness']
    best_fusion_fitness = min(fusion_results.values(), key=lambda x: x['fitness'])['fitness']
    
    improvement = (best_single_fitness - best_fusion_fitness) / best_single_fitness * 100
    print(f"\n融合算法相比最佳单算法:")
    print(f"  适应度提升: {improvement:.2f}%")
    
    print("\n" + "=" * 50)
    print("4. 算法特性分析")
    print("=" * 50)
    
    print("PSO特性:")
    print("  - 收敛速度快")
    print("  - 适合连续优化")
    print("  - 局部搜索能力强")
    
    print("\nDE特性:")
    print("  - 全局搜索能力强")
    print("  - 适合多峰函数")
    print("  - 参数少，易调节")
    
    print("\nGA特性:")
    print("  - 全局搜索能力强")
    print("  - 适合离散优化")
    print("  - 收敛速度较慢")
    
    print("\n融合算法优势:")
    print("  - 取长补短，综合各算法优点")
    print("  - 自适应选择最佳算法")
    print("  - 并行计算提高效率")
    print("  - 早停机制节省时间")
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)

if __name__ == "__main__":
    demo_fusion_algorithms()

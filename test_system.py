"""
系统测试脚本
验证亥姆霍兹线圈优化系统是否正常工作
"""

import numpy as np
import time
from helmholtz_objectives import create_helmholtz_bounds, FieldUniformityObjective
from algorithm_fusion import AlgorithmFusionOptimizer
from optimization_algorithms import PSO, DE, GA

def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    # 创建简单的测试目标函数
    def simple_objective(x):
        return np.sum(x**2)
    
    bounds = [(-5, 5), (-5, 5)]
    
    # 测试PSO算法
    print("测试PSO算法...")
    pso = PSO(bounds, simple_objective, max_iterations=20, population_size=10)
    solution, fitness = pso.optimize()
    print(f"PSO结果: 解={solution}, 适应度={fitness:.6f}")
    
    # 测试DE算法
    print("测试DE算法...")
    de = DE(bounds, simple_objective, max_iterations=20, population_size=10)
    solution, fitness = de.optimize()
    print(f"DE结果: 解={solution}, 适应度={fitness:.6f}")
    
    # 测试GA算法
    print("测试GA算法...")
    ga = GA(bounds, simple_objective, max_iterations=20, population_size=10)
    solution, fitness = ga.optimize()
    print(f"GA结果: 解={solution}, 适应度={fitness:.6f}")
    
    print("基本功能测试完成！\n")

def test_helmholtz_objectives():
    """测试亥姆霍兹线圈目标函数"""
    print("测试亥姆霍兹线圈目标函数...")
    
    # 创建线圈参数边界
    bounds = create_helmholtz_bounds(1)  # 1对线圈
    print(f"参数边界: {bounds}")
    
    # 测试参数
    test_params = np.array([0.1, 1.0, 100, 0.1])  # 半径, 电流, 匝数, 间距
    
    # 测试磁场均匀性目标函数
    try:
        uniformity_obj = FieldUniformityObjective(bounds, target_field=0.1)
        fitness = uniformity_obj(test_params)
        print(f"磁场均匀性目标函数: 适应度={fitness:.6f}")
    except Exception as e:
        print(f"磁场均匀性目标函数测试失败: {e}")
    
    print("亥姆霍兹线圈目标函数测试完成！\n")

def test_fusion_optimizer():
    """测试融合优化器"""
    print("测试融合优化器...")
    
    # 创建简单的测试目标函数
    def test_objective(x):
        return np.sum(x**2) + 0.1 * np.sin(np.sum(x))
    
    bounds = [(-3, 3), (-3, 3), (-3, 3)]
    
    # 测试自适应融合策略
    print("测试自适应融合策略...")
    optimizer = AlgorithmFusionOptimizer(
        bounds, test_objective, max_iterations=20, population_size=15, fusion_strategy="adaptive")
    
    start_time = time.time()
    solution, fitness = optimizer.optimize()
    execution_time = time.time() - start_time
    
    print(f"自适应融合结果: 解={solution}")
    print(f"适应度={fitness:.6f}, 时间={execution_time:.2f}s")
    
    # 测试并行融合策略
    print("测试并行融合策略...")
    optimizer = AlgorithmFusionOptimizer(
        bounds, test_objective, max_iterations=20, population_size=15, fusion_strategy="parallel")
    
    start_time = time.time()
    solution, fitness = optimizer.optimize()
    execution_time = time.time() - start_time
    
    print(f"并行融合结果: 解={solution}")
    print(f"适应度={fitness:.6f}, 时间={execution_time:.2f}s")
    
    print("融合优化器测试完成！\n")

def test_performance_comparison():
    """测试性能比较"""
    print("测试性能比较...")
    
    # 创建测试目标函数
    def benchmark_function(x):
        # Rastrigin函数
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    bounds = [(-5, 5), (-5, 5)]
    
    # 比较不同算法
    algorithms = {
        'PSO': PSO(bounds, benchmark_function, max_iterations=30, population_size=20),
        'DE': DE(bounds, benchmark_function, max_iterations=30, population_size=20),
        'GA': GA(bounds, benchmark_function, max_iterations=30, population_size=20)
    }
    
    results = {}
    for name, algorithm in algorithms.items():
        print(f"运行 {name}...")
        start_time = time.time()
        solution, fitness = algorithm.optimize()
        execution_time = time.time() - start_time
        
        results[name] = {
            'solution': solution,
            'fitness': fitness,
            'time': execution_time
        }
        
        print(f"{name}: 适应度={fitness:.6f}, 时间={execution_time:.2f}s")
    
    # 找出最佳算法
    best_algo = min(results.keys(), key=lambda x: results[x]['fitness'])
    print(f"最佳算法: {best_algo}")
    
    print("性能比较测试完成！\n")

def main():
    """主测试函数"""
    print("亥姆霍兹线圈优化系统测试")
    print("=" * 50)
    
    try:
        # 运行所有测试
        test_basic_functionality()
        test_helmholtz_objectives()
        test_fusion_optimizer()
        test_performance_comparison()
        
        print("所有测试完成！系统运行正常。")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

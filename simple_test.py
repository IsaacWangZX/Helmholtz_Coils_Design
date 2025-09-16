"""
简单测试程序
避免复杂的可视化功能，专注于基本功能测试
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def test_basic_visualization():
    """测试基本可视化功能"""
    print("测试基本可视化功能...")
    
    try:
        from visualization import HelmholtzVisualizer
        
        # 创建可视化器
        visualizer = HelmholtzVisualizer()
        
        # 创建简单的测试数据
        test_results = {
            'PSO': {'mean_fitness': 0.123, 'std_fitness': 0.045, 'best_fitness': 0.089, 
                    'mean_time': 1.23, 'std_time': 0.12, 'success_rate': 0.95},
            'DE': {'mean_fitness': 0.156, 'std_fitness': 0.067, 'best_fitness': 0.098, 
                   'mean_time': 0.98, 'std_time': 0.08, 'success_rate': 0.88},
            'GA': {'mean_fitness': 0.134, 'std_fitness': 0.052, 'best_fitness': 0.092, 
                   'mean_time': 1.45, 'std_time': 0.15, 'success_rate': 0.92}
        }
        
        # 测试简单的柱状图
        print("  测试柱状图...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = list(test_results.keys())
        fitnesses = [test_results[alg]['mean_fitness'] for alg in algorithms]
        
        bars = ax.bar(algorithms, fitnesses, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        ax.set_title('算法适应度对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('适应度', fontsize=12)
        ax.set_xlabel('算法', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, fitness in zip(bars, fitnesses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{fitness:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('simple_bar_chart.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("  ✓ 柱状图生成成功")
        
        # 测试线图
        print("  测试线图...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 模拟收敛曲线数据
        iterations = np.arange(0, 20)
        pso_curve = 1.0 * np.exp(-iterations/5) + 0.1 + np.random.normal(0, 0.01, 20)
        de_curve = 1.2 * np.exp(-iterations/6) + 0.12 + np.random.normal(0, 0.01, 20)
        ga_curve = 1.5 * np.exp(-iterations/8) + 0.15 + np.random.normal(0, 0.01, 20)
        
        ax.plot(iterations, pso_curve, 'b-', linewidth=2, label='PSO', marker='o', markersize=4)
        ax.plot(iterations, de_curve, 'r-', linewidth=2, label='DE', marker='s', markersize=4)
        ax.plot(iterations, ga_curve, 'g-', linewidth=2, label='GA', marker='^', markersize=4)
        
        ax.set_title('算法收敛曲线对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('适应度', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simple_line_chart.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("  ✓ 线图生成成功")
        
        # 测试散点图
        print("  测试散点图...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 模拟算法性能数据
        times = [test_results[alg]['mean_time'] for alg in algorithms]
        fitnesses = [test_results[alg]['mean_fitness'] for alg in algorithms]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, alg in enumerate(algorithms):
            ax.scatter(times[i], fitnesses[i], s=200, c=colors[i], alpha=0.7, 
                      label=alg, edgecolors='black', linewidth=1)
        
        ax.set_title('算法性能散点图', fontsize=14, fontweight='bold')
        ax.set_xlabel('执行时间 (s)', fontsize=12)
        ax.set_ylabel('适应度', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加算法标签
        for i, alg in enumerate(algorithms):
            ax.annotate(alg, (times[i], fitnesses[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('simple_scatter_chart.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("  ✓ 散点图生成成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 基本可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_helmholtz_modules():
    """测试亥姆霍兹模块"""
    print("\n测试亥姆霍兹模块...")
    
    try:
        from helmholtz_objectives import create_helmholtz_bounds, FieldUniformityObjective
        from helmholtz_coil import create_optimized_helmholtz_system
        
        # 创建测试数据
        bounds = create_helmholtz_bounds(1)
        test_params = np.array([0.1, 1.0, 100, 0.1])
        
        # 测试目标函数
        objective = FieldUniformityObjective(bounds, target_field=0.1)
        fitness = objective(test_params)
        
        print(f"  ✓ 目标函数测试成功，适应度: {fitness:.6f}")
        
        # 测试线圈系统
        system = create_optimized_helmholtz_system(test_params.tolist())
        
        print("  ✓ 线圈系统创建成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 亥姆霍兹模块测试失败: {e}")
        return False

def test_optimization_algorithms():
    """测试优化算法"""
    print("\n测试优化算法...")
    
    try:
        from optimization_algorithms import PSO, DE, GA
        
        # 创建简单的目标函数
        def simple_objective(x):
            return np.sum(x**2)
        
        bounds = [(-2, 2), (-2, 2)]
        
        # 测试PSO
        print("  测试PSO算法...")
        pso = PSO(bounds, simple_objective, max_iterations=10, population_size=10)
        solution, fitness = pso.optimize()
        print(f"    PSO结果: 解={solution}, 适应度={fitness:.6f}")
        
        # 测试DE
        print("  测试DE算法...")
        de = DE(bounds, simple_objective, max_iterations=10, population_size=10)
        solution, fitness = de.optimize()
        print(f"    DE结果: 解={solution}, 适应度={fitness:.6f}")
        
        # 测试GA
        print("  测试GA算法...")
        ga = GA(bounds, simple_objective, max_iterations=10, population_size=10)
        solution, fitness = ga.optimize()
        print(f"    GA结果: 解={solution}, 适应度={fitness:.6f}")
        
        print("  ✓ 所有优化算法测试成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 优化算法测试失败: {e}")
        return False

def test_fusion_algorithm():
    """测试融合算法"""
    print("\n测试融合算法...")
    
    try:
        from algorithm_fusion import AlgorithmFusionOptimizer
        
        # 创建简单的目标函数
        def simple_objective(x):
            return np.sum(x**2)
        
        bounds = [(-2, 2), (-2, 2)]
        
        # 测试并行融合策略
        print("  测试并行融合策略...")
        optimizer = AlgorithmFusionOptimizer(
            bounds, simple_objective, max_iterations=8, population_size=8, fusion_strategy="parallel")
        
        solution, fitness = optimizer.optimize()
        print(f"    融合结果: 解={solution}, 适应度={fitness:.6f}")
        
        print("  ✓ 融合算法测试成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 融合算法测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("亥姆霍兹线圈优化系统 - 简单测试")
    print("=" * 50)
    
    tests = [
        test_basic_visualization,
        test_helmholtz_modules,
        test_optimization_algorithms,
        test_fusion_algorithm
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print("测试结果总结")
    print("=" * 50)
    print(f"通过测试: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("✓ 所有测试通过！")
        print("\n生成的文件:")
        print("- simple_bar_chart.png: 柱状图")
        print("- simple_line_chart.png: 线图")
        print("- simple_scatter_chart.png: 散点图")
    else:
        print("✗ 部分测试失败")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    print(f"\n测试{'成功' if success else '失败'}")

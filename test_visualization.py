"""
可视化功能测试
测试各种可视化功能是否正常工作
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from helmholtz_objectives import create_helmholtz_bounds, FieldUniformityObjective
from helmholtz_coil import create_optimized_helmholtz_system
from visualization import HelmholtzVisualizer, create_comparison_report
from optimization_algorithms import PSO, DE, GA

def test_basic_visualization():
    """测试基本可视化功能"""
    print("测试基本可视化功能...")
    
    try:
        # 创建测试数据
        bounds = create_helmholtz_bounds(1)
        test_params = np.array([0.1, 1.0, 100, 0.1])
        
        # 创建线圈系统
        system = create_optimized_helmholtz_system(test_params.tolist())
        
        # 创建可视化器
        visualizer = HelmholtzVisualizer()
        
        # 测试磁场分布图
        print("测试磁场分布图...")
        fig1 = visualizer.plot_field_distribution(
            system, region_size=0.02, resolution=10, title="测试磁场分布")
        plt.savefig('test_field_distribution.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print("✓ 磁场分布图保存成功")
        
        # 测试磁场均匀性分析图
        print("测试磁场均匀性分析图...")
        fig2 = visualizer.plot_field_uniformity(
            system, region_size=0.02, resolution=10, title="测试磁场均匀性")
        plt.savefig('test_field_uniformity.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print("✓ 磁场均匀性分析图保存成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本可视化测试失败: {e}")
        return False

def test_algorithm_comparison():
    """测试算法对比可视化"""
    print("\n测试算法对比可视化...")
    
    try:
        # 创建简化的目标函数
        def simple_objective(x):
            return np.sum(x**2)
        
        bounds = [(-2, 2), (-2, 2)]
        
        # 运行算法
        algorithms = {
            'PSO': PSO(bounds, simple_objective, max_iterations=10, population_size=10),
            'DE': DE(bounds, simple_objective, max_iterations=10, population_size=10),
            'GA': GA(bounds, simple_objective, max_iterations=10, population_size=10)
        }
        
        results = {}
        for name, algorithm in algorithms.items():
            solution, fitness = algorithm.optimize()
            results[name] = {
                'mean_fitness': fitness,
                'std_fitness': 0.01,
                'best_fitness': fitness,
                'mean_time': 0.1,
                'std_time': 0.01,
                'success_rate': 1.0
            }
        
        # 创建可视化器
        visualizer = HelmholtzVisualizer()
        
        # 测试算法对比图
        print("测试算法对比图...")
        fig1 = visualizer.plot_algorithm_comparison(results, title="测试算法对比")
        plt.savefig('test_algorithm_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print("✓ 算法对比图保存成功")
        
        # 测试优化总结图
        print("测试优化总结图...")
        fig2 = visualizer.plot_optimization_summary(results, title="测试优化总结")
        plt.savefig('test_optimization_summary.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print("✓ 优化总结图保存成功")
        
        # 测试HTML报告
        print("测试HTML报告生成...")
        create_comparison_report(results, 'test_comparison_report.html')
        print("✓ HTML报告生成成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 算法对比可视化测试失败: {e}")
        return False

def test_convergence_curves():
    """测试收敛曲线可视化"""
    print("\n测试收敛曲线可视化...")
    
    try:
        # 创建简化的目标函数
        def simple_objective(x):
            return np.sum(x**2)
        
        bounds = [(-2, 2), (-2, 2)]
        
        # 创建算法
        algorithms = [
            PSO(bounds, simple_objective, max_iterations=15, population_size=10),
            DE(bounds, simple_objective, max_iterations=15, population_size=10)
        ]
        
        # 运行算法
        for algorithm in algorithms:
            algorithm.optimize()
        
        # 创建可视化器
        visualizer = HelmholtzVisualizer()
        
        # 测试收敛曲线图
        print("测试收敛曲线图...")
        fig = visualizer.plot_convergence_curves(algorithms, title="测试收敛曲线")
        plt.savefig('test_convergence_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ 收敛曲线图保存成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 收敛曲线可视化测试失败: {e}")
        return False

def test_pareto_front():
    """测试帕累托前沿可视化"""
    print("\n测试帕累托前沿可视化...")
    
    try:
        # 创建测试数据
        pareto_solutions = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([1.0, 1.0]),
            np.array([0.2, 0.8]),
            np.array([0.8, 0.2])
        ]
        
        pareto_objectives = [
            [0.0, 2.0],
            [0.5, 0.5],
            [2.0, 0.0],
            [0.68, 0.68],
            [0.68, 0.68]
        ]
        
        # 创建可视化器
        visualizer = HelmholtzVisualizer()
        
        # 测试帕累托前沿图
        print("测试帕累托前沿图...")
        fig = visualizer.plot_pareto_front(
            pareto_solutions, pareto_objectives, title="测试帕累托前沿")
        plt.savefig('test_pareto_front.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("✓ 帕累托前沿图保存成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 帕累托前沿可视化测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("亥姆霍兹线圈可视化功能测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(test_basic_visualization())
    test_results.append(test_algorithm_comparison())
    test_results.append(test_convergence_curves())
    test_results.append(test_pareto_front())
    
    # 统计结果
    passed = sum(test_results)
    total = len(test_results)
    
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    print(f"通过测试: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("✓ 所有可视化功能测试通过！")
        print("\n生成的测试文件:")
        print("- test_field_distribution.png: 磁场分布图")
        print("- test_field_uniformity.png: 磁场均匀性分析图")
        print("- test_algorithm_comparison.png: 算法对比图")
        print("- test_optimization_summary.png: 优化总结图")
        print("- test_convergence_curves.png: 收敛曲线图")
        print("- test_pareto_front.png: 帕累托前沿图")
        print("- test_comparison_report.html: HTML对比报告")
    else:
        print("✗ 部分测试失败，请检查错误信息")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

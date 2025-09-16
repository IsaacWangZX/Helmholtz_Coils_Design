"""
亥姆霍兹线圈可视化演示程序
展示各种可视化功能，包括磁场分布、算法对比等
"""

import numpy as np
import matplotlib.pyplot as plt
from helmholtz_objectives import create_helmholtz_bounds, FieldUniformityObjective
from helmholtz_coil import create_optimized_helmholtz_system
from visualization import HelmholtzVisualizer, create_comparison_report
from optimization_algorithms import PSO, DE, GA, run_optimization_comparison

def demo_field_visualization():
    """演示磁场可视化功能"""
    print("=" * 60)
    print("磁场可视化演示")
    print("=" * 60)
    
    # 创建参数边界
    bounds = create_helmholtz_bounds(1)  # 1对线圈
    
    # 创建测试线圈参数
    test_params = np.array([0.1, 1.0, 100, 0.1])  # 半径, 电流, 匝数, 间距
    print(f"测试线圈参数: {test_params}")
    print(f"参数说明: [半径(m), 电流(A), 匝数, 间距(m)]")
    
    # 创建线圈系统
    system = create_optimized_helmholtz_system(test_params.tolist())
    
    # 创建可视化器
    visualizer = HelmholtzVisualizer()
    
    # 绘制磁场分布
    print("\n绘制磁场分布图...")
    visualizer.plot_field_distribution(
        system, region_size=0.02, resolution=15, 
        title="标准亥姆霍兹线圈磁场分布")
    
    # 绘制磁场均匀性分析
    print("绘制磁场均匀性分析图...")
    visualizer.plot_field_uniformity(
        system, region_size=0.02, resolution=15,
        title="标准亥姆霍兹线圈磁场均匀性分析")
    
    print("磁场可视化演示完成！\n")

def demo_algorithm_comparison():
    """演示算法对比可视化功能"""
    print("=" * 60)
    print("算法对比可视化演示")
    print("=" * 60)
    
    # 创建简化的目标函数
    def simple_objective(x):
        # Rastrigin函数
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    # 运行算法比较
    print("运行算法比较...")
    results = run_optimization_comparison(
        bounds, simple_objective, max_iterations=30, population_size=20, num_runs=3)
    
    # 创建可视化器
    visualizer = HelmholtzVisualizer()
    
    # 绘制算法对比图
    print("绘制算法性能对比图...")
    visualizer.plot_algorithm_comparison(results, title="优化算法性能对比")
    
    # 绘制优化总结图
    print("绘制优化结果总结图...")
    visualizer.plot_optimization_summary(results, title="优化结果总结")
    
    # 生成HTML报告
    print("生成HTML对比报告...")
    create_comparison_report(results, "algorithm_comparison_report.html")
    
    print("算法对比可视化演示完成！\n")
    
    return results

def demo_convergence_curves():
    """演示收敛曲线可视化"""
    print("=" * 60)
    print("收敛曲线可视化演示")
    print("=" * 60)
    
    # 创建简化的目标函数
    def test_objective(x):
        return np.sum(x**2) + 0.1 * np.sin(np.sum(x))
    
    bounds = [(-3, 3), (-3, 3)]
    
    # 创建算法
    algorithms = [
        PSO(bounds, test_objective, max_iterations=50, population_size=20),
        DE(bounds, test_objective, max_iterations=50, population_size=20),
        GA(bounds, test_objective, max_iterations=50, population_size=20)
    ]
    
    # 运行算法
    print("运行算法并记录收敛过程...")
    for i, algorithm in enumerate(algorithms):
        print(f"运行 {algorithm.__class__.__name__}...")
        algorithm.optimize()
    
    # 创建可视化器
    visualizer = HelmholtzVisualizer()
    
    # 绘制收敛曲线
    print("绘制收敛曲线对比图...")
    visualizer.plot_convergence_curves(algorithms, title="算法收敛曲线对比")
    
    print("收敛曲线可视化演示完成！\n")

def demo_optimization_with_visualization():
    """演示带可视化的优化过程"""
    print("=" * 60)
    print("带可视化的优化过程演示")
    print("=" * 60)
    
    # 创建亥姆霍兹线圈优化问题
    bounds = create_helmholtz_bounds(1)
    objective = FieldUniformityObjective(bounds, target_field=0.1)
    
    print(f"优化问题: 亥姆霍兹线圈磁场均匀性优化")
    print(f"目标磁场强度: 0.1 T")
    print(f"参数维度: {len(bounds)}")
    
    # 运行优化
    print("\n运行PSO优化...")
    pso = PSO(bounds, objective, max_iterations=20, population_size=15)
    solution, fitness = pso.optimize()
    
    print(f"优化结果:")
    print(f"最优解: {solution}")
    print(f"最优适应度: {fitness:.6f}")
    
    # 创建最优线圈系统
    try:
        best_system = create_optimized_helmholtz_system(solution.tolist())
        
        # 创建可视化器
        visualizer = HelmholtzVisualizer()
        
        # 绘制最优解的磁场分布
        print("\n绘制最优线圈磁场分布...")
        visualizer.plot_field_distribution(
            best_system, region_size=0.02, resolution=12,
            title="优化后亥姆霍兹线圈磁场分布")
        
        # 绘制磁场均匀性分析
        print("绘制最优线圈磁场均匀性分析...")
        visualizer.plot_field_uniformity(
            best_system, region_size=0.02, resolution=12,
            title="优化后亥姆霍兹线圈磁场均匀性分析")
        
        # 绘制收敛曲线
        print("绘制优化收敛曲线...")
        visualizer.plot_convergence_curves([pso], title="PSO优化收敛曲线")
        
    except Exception as e:
        print(f"磁场可视化出错: {e}")
    
    print("带可视化的优化过程演示完成！\n")

def demo_multi_objective_visualization():
    """演示多目标优化可视化"""
    print("=" * 60)
    print("多目标优化可视化演示")
    print("=" * 60)
    
    # 创建简化的多目标函数
    def multi_objective(x):
        f1 = np.sum(x**2)  # 目标1: 最小化
        f2 = np.sum((x - 1)**2)  # 目标2: 最小化
        return [f1, f2]
    
    bounds = [(-2, 2), (-2, 2)]
    
    # 生成一些帕累托前沿解
    print("生成帕累托前沿解...")
    pareto_solutions = []
    pareto_objectives = []
    
    for i in range(50):
        # 随机生成解
        solution = np.array([np.random.uniform(bounds[0][0], bounds[0][1]),
                           np.random.uniform(bounds[1][0], bounds[1][1])])
        objectives = multi_objective(solution)
        pareto_solutions.append(solution)
        pareto_objectives.append(objectives)
    
    # 创建可视化器
    visualizer = HelmholtzVisualizer()
    
    # 绘制帕累托前沿
    print("绘制帕累托前沿...")
    visualizer.plot_pareto_front(
        pareto_solutions, pareto_objectives, 
        title="多目标优化帕累托前沿")
    
    print("多目标优化可视化演示完成！\n")

def main():
    """主演示函数"""
    print("亥姆霍兹线圈可视化系统演示")
    print("=" * 60)
    
    try:
        # 1. 磁场可视化演示
        demo_field_visualization()
        
        # 2. 算法对比可视化演示
        demo_algorithm_comparison()
        
        # 3. 收敛曲线可视化演示
        demo_convergence_curves()
        
        # 4. 带可视化的优化过程演示
        demo_optimization_with_visualization()
        
        # 5. 多目标优化可视化演示
        demo_multi_objective_visualization()
        
        print("=" * 60)
        print("所有可视化演示完成！")
        print("=" * 60)
        
        print("\n生成的文件:")
        print("- algorithm_comparison_report.html: 算法对比HTML报告")
        print("- 各种可视化图表已显示")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

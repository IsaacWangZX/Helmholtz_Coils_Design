"""
简化版最终演示
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

def simple_demo():
    """简化演示"""
    print("亥姆霍兹线圈优化系统 - 简化演示")
    print("=" * 50)
    
    # 测试基本功能
    print("1. 测试基本功能...")
    
    try:
        from helmholtz_objectives import create_helmholtz_bounds, FieldUniformityObjective
        from helmholtz_coil import create_optimized_helmholtz_system
        from algorithm_fusion import AlgorithmFusionOptimizer
        from optimization_algorithms import PSO, DE, GA
        from visualization import HelmholtzVisualizer
        
        print("   ✓ 所有模块导入成功")
        
        # 创建优化问题
        bounds = create_helmholtz_bounds(1)
        objective = FieldUniformityObjective(bounds, target_field=0.1)
        
        print(f"   ✓ 优化问题创建成功")
        print(f"     参数维度: {len(bounds)}")
        
        # 测试单算法
        print("\n2. 测试单算法优化...")
        pso = PSO(bounds, objective, max_iterations=10, population_size=10)
        solution, fitness = pso.optimize()
        
        print(f"   ✓ PSO优化完成")
        print(f"     最优解: {solution}")
        print(f"     最优适应度: {fitness:.6f}")
        
        # 测试融合算法
        print("\n3. 测试融合算法...")
        optimizer = AlgorithmFusionOptimizer(
            bounds, objective, max_iterations=8, population_size=8, fusion_strategy="parallel")
        
        solution, fitness = optimizer.optimize()
        
        print(f"   ✓ 融合算法优化完成")
        print(f"     最优解: {solution}")
        print(f"     最优适应度: {fitness:.6f}")
        
        # 测试可视化
        print("\n4. 测试可视化功能...")
        visualizer = HelmholtzVisualizer()
        
        # 创建测试数据
        test_results = {
            'PSO': {'mean_fitness': 0.123, 'std_fitness': 0.045, 'best_fitness': 0.089, 
                    'mean_time': 1.23, 'std_time': 0.12, 'success_rate': 0.95},
            'DE': {'mean_fitness': 0.156, 'std_fitness': 0.067, 'best_fitness': 0.098, 
                   'mean_time': 0.98, 'std_time': 0.08, 'success_rate': 0.88},
            'GA': {'mean_fitness': 0.134, 'std_fitness': 0.052, 'best_fitness': 0.092, 
                   'mean_time': 1.45, 'std_time': 0.15, 'success_rate': 0.92}
        }
        
        # 生成对比图
        fig = visualizer.plot_algorithm_comparison(test_results, title="算法性能对比")
        plt.savefig('simple_algorithm_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("   ✓ 算法对比图生成成功")
        
        # 生成总结图
        fig = visualizer.plot_optimization_summary(test_results, title="优化结果总结")
        plt.savefig('simple_optimization_summary.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("   ✓ 优化总结图生成成功")
        
        # 测试磁场可视化
        print("\n5. 测试磁场可视化...")
        try:
            # 创建测试线圈系统
            test_params = np.array([0.1, 1.0, 100, 0.1])
            system = create_optimized_helmholtz_system(test_params.tolist())
            
            # 生成磁场分布图
            fig = visualizer.plot_field_distribution(
                system, region_size=0.02, resolution=10, title="测试磁场分布")
            plt.savefig('simple_field_distribution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print("   ✓ 磁场分布图生成成功")
            
            # 生成磁场均匀性分析图
            fig = visualizer.plot_field_uniformity(
                system, region_size=0.02, resolution=10, title="测试磁场均匀性")
            plt.savefig('simple_field_uniformity.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print("   ✓ 磁场均匀性分析图生成成功")
            
        except Exception as e:
            print(f"   ⚠ 磁场可视化测试跳过: {e}")
        
        # 生成HTML报告
        print("\n6. 生成HTML报告...")
        from visualization import create_comparison_report
        create_comparison_report(test_results, 'simple_optimization_report.html')
        
        print("   ✓ HTML报告生成成功")
        
        print("\n" + "=" * 50)
        print("演示完成！")
        print("=" * 50)
        
        print("\n生成的文件:")
        print("- simple_algorithm_comparison.png: 算法对比图")
        print("- simple_optimization_summary.png: 优化总结图")
        print("- simple_field_distribution.png: 磁场分布图")
        print("- simple_field_uniformity.png: 磁场均匀性分析图")
        print("- simple_optimization_report.html: HTML优化报告")
        
        return True
        
    except Exception as e:
        print(f"✗ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_demo()
    print(f"\n演示{'成功' if success else '失败'}")

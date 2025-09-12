"""
亥姆霍兹线圈优化主程序
演示如何使用融合算法优化亥姆霍兹线圈设计
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

# 导入自定义模块
from helmholtz_objectives import (
    FieldUniformityObjective, FieldStrengthObjective, 
    EfficiencyObjective, RobustObjective, create_helmholtz_bounds
)
from algorithm_fusion import AlgorithmFusionOptimizer, compare_fusion_strategies
from optimization_algorithms import (
    run_optimization_comparison, plot_optimization_comparison,
    create_optimization_suite
)

class HelmholtzOptimizationDemo:
    """亥姆霍兹线圈优化演示类"""
    
    def __init__(self, num_coil_pairs: int = 2, target_field: float = 0.1):
        """
        初始化优化演示
        
        Args:
            num_coil_pairs: 线圈对数量
            target_field: 目标磁场强度 (T)
        """
        self.num_coil_pairs = num_coil_pairs
        self.target_field = target_field
        self.bounds = create_helmholtz_bounds(num_coil_pairs)
        
        print(f"亥姆霍兹线圈优化演示")
        print(f"线圈对数量: {num_coil_pairs}")
        print(f"目标磁场强度: {target_field} T")
        print(f"参数维度: {len(self.bounds)}")
        print(f"参数边界: {self.bounds}")
    
    def create_objective_functions(self) -> Dict[str, callable]:
        """创建目标函数"""
        objectives = {
            'uniformity': FieldUniformityObjective(
                self.bounds, self.target_field, region_size=0.02, resolution=15),
            'strength': FieldStrengthObjective(
                self.bounds, self.target_field, region_size=0.02, resolution=15),
            'efficiency': EfficiencyObjective(
                self.bounds, self.target_field, region_size=0.02, resolution=15),
            'robust': RobustObjective(
                self.bounds, self.target_field, region_size=0.02, resolution=15)
        }
        return objectives
    
    def run_single_objective_optimization(self, objective_name: str, 
                                        max_iterations: int = 50, 
                                        population_size: int = 30) -> Dict:
        """运行单目标优化"""
        print(f"\n=== 运行单目标优化: {objective_name} ===")
        
        objectives = self.create_objective_functions()
        objective_func = objectives[objective_name]
        
        # 使用融合算法优化
        optimizer = AlgorithmFusionOptimizer(
            self.bounds, objective_func, max_iterations, population_size, "adaptive")
        
        start_time = time.time()
        solution, fitness = optimizer.optimize()
        execution_time = time.time() - start_time
        
        print(f"最优解: {solution}")
        print(f"最优适应度: {fitness:.6f}")
        print(f"执行时间: {execution_time:.2f}s")
        
        return {
            'solution': solution,
            'fitness': fitness,
            'execution_time': execution_time,
            'objective_name': objective_name
        }
    
    def run_multi_objective_optimization(self, max_iterations: int = 50, 
                                        population_size: int = 30) -> Dict:
        """运行多目标优化"""
        print(f"\n=== 运行多目标优化 ===")
        
        # 创建多目标函数
        from helmholtz_objectives import MultiObjectiveHelmholtz
        multi_obj = MultiObjectiveHelmholtz(
            self.bounds, self.target_field, region_size=0.02, resolution=15)
        
        # 使用NSGA-II算法
        from optimization_algorithms import NSGA2
        nsga2 = NSGA2(self.bounds, [multi_obj], max_iterations, population_size)
        
        start_time = time.time()
        pareto_solutions, pareto_objectives = nsga2.optimize()
        execution_time = time.time() - start_time
        
        print(f"帕累托前沿解数量: {len(pareto_solutions)}")
        print(f"执行时间: {execution_time:.2f}s")
        
        # 显示前几个解
        for i, (sol, obj) in enumerate(zip(pareto_solutions[:5], pareto_objectives[:5])):
            print(f"解 {i+1}: {sol}")
            print(f"目标值: {obj}")
        
        return {
            'pareto_solutions': pareto_solutions,
            'pareto_objectives': pareto_objectives,
            'execution_time': execution_time
        }
    
    def compare_fusion_strategies(self, max_iterations: int = 30, 
                                population_size: int = 20) -> Dict:
        """比较不同融合策略"""
        print(f"\n=== 比较融合策略 ===")
        
        # 使用磁场均匀性作为目标函数
        objectives = self.create_objective_functions()
        objective_func = objectives['uniformity']
        
        # 比较不同融合策略
        results = compare_fusion_strategies(
            self.bounds, objective_func, max_iterations, population_size)
        
        return results
    
    def compare_algorithms(self, max_iterations: int = 30, 
                         population_size: int = 20, num_runs: int = 3) -> Dict:
        """比较不同优化算法"""
        print(f"\n=== 比较优化算法 ===")
        
        # 使用磁场均匀性作为目标函数
        objectives = self.create_objective_functions()
        objective_func = objectives['uniformity']
        
        # 运行算法比较
        results = run_optimization_comparison(
            self.bounds, objective_func, max_iterations, population_size, num_runs)
        
        return results
    
    def run_comprehensive_optimization(self, max_iterations: int = 50, 
                                     population_size: int = 30) -> Dict:
        """运行综合优化"""
        print(f"\n=== 综合优化 ===")
        
        all_results = {}
        
        # 1. 单目标优化
        objectives = self.create_objective_functions()
        for obj_name in ['uniformity', 'strength', 'efficiency']:
            print(f"\n--- {obj_name} 优化 ---")
            result = self.run_single_objective_optimization(
                obj_name, max_iterations, population_size)
            all_results[f'single_{obj_name}'] = result
        
        # 2. 多目标优化
        print(f"\n--- 多目标优化 ---")
        multi_result = self.run_multi_objective_optimization(max_iterations, population_size)
        all_results['multi_objective'] = multi_result
        
        # 3. 融合策略比较
        print(f"\n--- 融合策略比较 ---")
        fusion_results = self.compare_fusion_strategies(max_iterations//2, population_size//2)
        all_results['fusion_comparison'] = fusion_results
        
        # 4. 算法比较
        print(f"\n--- 算法比较 ---")
        algo_results = self.compare_algorithms(max_iterations//2, population_size//2, 2)
        all_results['algorithm_comparison'] = algo_results
        
        return all_results
    
    def visualize_results(self, results: Dict):
        """可视化结果"""
        print(f"\n=== 结果可视化 ===")
        
        # 绘制融合策略比较
        if 'fusion_comparison' in results:
            from algorithm_fusion import plot_fusion_comparison
            plot_fusion_comparison(results['fusion_comparison'])
        
        # 绘制算法比较
        if 'algorithm_comparison' in results:
            plot_optimization_comparison(results['algorithm_comparison'])
        
        # 绘制单目标优化结果
        single_results = {k: v for k, v in results.items() if k.startswith('single_')}
        if single_results:
            self._plot_single_objective_results(single_results)
    
    def _plot_single_objective_results(self, results: Dict):
        """绘制单目标优化结果"""
        objectives = list(results.keys())
        fitnesses = [results[obj]['fitness'] for obj in objectives]
        times = [results[obj]['execution_time'] for obj in objectives]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 适应度比较
        ax1.bar(objectives, fitnesses)
        ax1.set_title('单目标优化适应度比较')
        ax1.set_ylabel('适应度')
        ax1.tick_params(axis='x', rotation=45)
        
        # 执行时间比较
        ax2.bar(objectives, times)
        ax2.set_title('单目标优化执行时间比较')
        ax2.set_ylabel('执行时间 (s)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results: Dict) -> str:
        """生成优化报告"""
        report = []
        report.append("=" * 50)
        report.append("亥姆霍兹线圈优化报告")
        report.append("=" * 50)
        report.append(f"线圈对数量: {self.num_coil_pairs}")
        report.append(f"目标磁场强度: {self.target_field} T")
        report.append(f"参数维度: {len(self.bounds)}")
        report.append("")
        
        # 单目标优化结果
        single_results = {k: v for k, v in results.items() if k.startswith('single_')}
        if single_results:
            report.append("单目标优化结果:")
            report.append("-" * 30)
            for obj_name, result in single_results.items():
                report.append(f"{obj_name}:")
                report.append(f"  最优适应度: {result['fitness']:.6f}")
                report.append(f"  执行时间: {result['execution_time']:.2f}s")
                report.append(f"  最优解: {result['solution']}")
                report.append("")
        
        # 多目标优化结果
        if 'multi_objective' in results:
            multi_result = results['multi_objective']
            report.append("多目标优化结果:")
            report.append("-" * 30)
            report.append(f"帕累托前沿解数量: {len(multi_result['pareto_solutions'])}")
            report.append(f"执行时间: {multi_result['execution_time']:.2f}s")
            report.append("")
        
        # 融合策略比较
        if 'fusion_comparison' in results:
            fusion_results = results['fusion_comparison']
            report.append("融合策略比较:")
            report.append("-" * 30)
            for strategy, result in fusion_results.items():
                report.append(f"{strategy}:")
                report.append(f"  最优适应度: {result['best_fitness']:.6f}")
                report.append(f"  执行时间: {result['execution_time']:.2f}s")
                report.append("")
        
        # 算法比较
        if 'algorithm_comparison' in results:
            algo_results = results['algorithm_comparison']
            report.append("算法比较:")
            report.append("-" * 30)
            for algo_name, result in algo_results.items():
                report.append(f"{algo_name}:")
                report.append(f"  平均适应度: {result['mean_fitness']:.6f} ± {result['std_fitness']:.6f}")
                report.append(f"  最佳适应度: {result['best_fitness']:.6f}")
                report.append(f"  平均时间: {result['mean_time']:.2f}s")
                report.append(f"  成功率: {result['success_rate']:.2%}")
                report.append("")
        
        report.append("=" * 50)
        
        return "\n".join(report)

def main():
    """主函数"""
    print("亥姆霍兹线圈优化系统")
    print("=" * 50)
    
    # 创建优化演示
    demo = HelmholtzOptimizationDemo(num_coil_pairs=1, target_field=0.1)
    
    # 运行综合优化
    results = demo.run_comprehensive_optimization(max_iterations=30, population_size=20)
    
    # 可视化结果
    demo.visualize_results(results)
    
    # 生成报告
    report = demo.generate_report(results)
    print(report)
    
    # 保存报告到文件
    with open('optimization_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n优化报告已保存到 optimization_report.txt")

if __name__ == "__main__":
    main()

"""
算法融合模块
实现多种优化算法的智能融合，取长补短，提高亥姆霍兹线圈设计优化效率
"""

import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Dict, Callable, Optional, Union
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from optimization_algorithms import PSO, DE, GA, NSGA2, OptimizationAlgorithm

class AlgorithmType(Enum):
    """算法类型枚举"""
    PSO = "PSO"
    DE = "DE"
    GA = "GA"
    NSGA2 = "NSGA2"

@dataclass
class AlgorithmPerformance:
    """算法性能记录"""
    algorithm_type: AlgorithmType
    best_fitness: float
    execution_time: float
    convergence_iterations: int
    stability: float  # 解的稳定性
    diversity: float  # 解的多样性
    success_rate: float  # 成功率

class AdaptivePSO(PSO):
    """自适应粒子群优化算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable,
                 max_iterations: int = 100, population_size: int = 50,
                 w_min: float = 0.4, w_max: float = 0.9, 
                 c1_min: float = 1.5, c1_max: float = 2.5,
                 c2_min: float = 1.5, c2_max: float = 2.5):
        """
        初始化自适应PSO算法
        
        Args:
            bounds: 参数边界
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            population_size: 种群大小
            w_min, w_max: 惯性权重范围
            c1_min, c1_max: 个体学习因子范围
            c2_min, c2_max: 社会学习因子范围
        """
        super().__init__(bounds, objective_func, max_iterations, population_size)
        self.w_min = w_min
        self.w_max = w_max
        self.c1_min = c1_min
        self.c1_max = c1_max
        self.c2_min = c2_min
        self.c2_max = c2_max
        
        # 自适应参数
        self.stagnation_count = 0
        self.last_best_fitness = float('inf')
        self.diversity_history = []
        
    def calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.particles) < 2:
            return 0.0
        
        # 计算粒子间的平均距离
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.particles)):
            for j in range(i + 1, len(self.particles)):
                distance = np.linalg.norm(self.particles[i] - self.particles[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def adaptive_parameters(self, iteration: int):
        """自适应调整参数"""
        # 计算种群多样性
        diversity = self.calculate_diversity()
        self.diversity_history.append(diversity)
        
        # 检测停滞
        if abs(self.best_fitness - self.last_best_fitness) < 1e-8:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        
        self.last_best_fitness = self.best_fitness
        
        # 自适应调整惯性权重
        if self.stagnation_count > 5:
            # 停滞时增加多样性
            self.w = min(self.w_max, self.w + 0.1)
            self.c1 = max(self.c1_min, self.c1 - 0.1)
            self.c2 = max(self.c2_min, self.c2 - 0.1)
        else:
            # 正常收敛时减少惯性权重
            self.w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iterations
        
        # 根据多样性调整学习因子
        if diversity < 0.01:  # 多样性过低
            self.c1 = min(self.c1_max, self.c1 + 0.1)
            self.c2 = min(self.c2_max, self.c2 + 0.1)
        elif diversity > 0.1:  # 多样性过高
            self.c1 = max(self.c1_min, self.c1 - 0.1)
            self.c2 = max(self.c2_min, self.c2 - 0.1)
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行自适应PSO优化"""
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # 自适应参数调整
            self.adaptive_parameters(iteration)
            
            for i in range(self.population_size):
                # 更新速度
                r1 = np.random.random()
                r2 = np.random.random()
                
                cognitive = self.c1 * r1 * (self.personal_best[i] - self.particles[i])
                social = self.c2 * r2 * (self.best_solution - self.particles[i])
                
                self.velocities[i] = (self.w * self.velocities[i] + 
                                    cognitive + social)
                
                # 更新位置
                self.particles[i] += self.velocities[i]
                
                # 边界处理
                for j in range(self.dimension):
                    if self.particles[i][j] < self.bounds[j][0]:
                        self.particles[i][j] = self.bounds[j][0]
                        self.velocities[i][j] = 0
                    elif self.particles[i][j] > self.bounds[j][1]:
                        self.particles[i][j] = self.bounds[j][1]
                        self.velocities[i][j] = 0
                
                # 评估适应度
                fitness = self.evaluate_fitness(self.particles[i])
                
                # 更新个体最优
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()
                    
                    # 更新全局最优
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = self.particles[i].copy()
            
            # 记录适应度历史
            self.fitness_history.append(self.best_fitness)
            
            # 早停机制
            if len(self.fitness_history) > 20:
                recent_improvement = self.fitness_history[-20] - self.fitness_history[-1]
                if recent_improvement < 1e-10:
                    break
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness

class AdaptiveDE(DE):
    """自适应差分进化算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable,
                 max_iterations: int = 100, population_size: int = 50,
                 F_min: float = 0.1, F_max: float = 1.0,
                 CR_min: float = 0.1, CR_max: float = 0.9):
        """
        初始化自适应DE算法
        
        Args:
            bounds: 参数边界
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            population_size: 种群大小
            F_min, F_max: 缩放因子范围
            CR_min, CR_max: 交叉概率范围
        """
        super().__init__(bounds, objective_func, max_iterations, population_size)
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        
        # 自适应参数
        self.F_history = []
        self.CR_history = []
        self.success_count = 0
        self.total_trials = 0
        
    def adaptive_parameters(self, iteration: int):
        """自适应调整参数"""
        # 计算成功率
        success_rate = self.success_count / max(self.total_trials, 1)
        
        # 根据成功率调整F
        if success_rate > 0.3:
            self.F = min(self.F_max, self.F + 0.1)
        elif success_rate < 0.1:
            self.F = max(self.F_min, self.F - 0.1)
        
        # 根据迭代进度调整CR
        self.CR = self.CR_min + (self.CR_max - self.CR_min) * (1 - iteration / self.max_iterations)
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行自适应DE优化"""
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # 自适应参数调整
            self.adaptive_parameters(iteration)
            
            new_population = []
            new_fitness = []
            
            for i in range(self.population_size):
                # 选择三个不同的个体
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # 变异
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                
                # 交叉
                trial = self.population[i].copy()
                for j in range(self.dimension):
                    if np.random.random() < self.CR:
                        trial[j] = mutant[j]
                
                # 边界处理
                for j in range(self.dimension):
                    if trial[j] < self.bounds[j][0]:
                        trial[j] = self.bounds[j][0]
                    elif trial[j] > self.bounds[j][1]:
                        trial[j] = self.bounds[j][1]
                
                # 选择
                trial_fitness = self.evaluate_fitness(trial)
                self.total_trials += 1
                
                if trial_fitness < self.fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                    self.success_count += 1
                else:
                    new_population.append(self.population[i])
                    new_fitness.append(self.fitness[i])
                
                # 更新最优解
                if new_fitness[-1] < self.best_fitness:
                    self.best_fitness = new_fitness[-1]
                    self.best_solution = new_population[-1].copy()
            
            self.population = new_population
            self.fitness = new_fitness
            self.fitness_history.append(self.best_fitness)
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness

class AdaptiveGA(GA):
    """自适应遗传算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable,
                 max_iterations: int = 100, population_size: int = 50,
                 mutation_rate_min: float = 0.01, mutation_rate_max: float = 0.3,
                 crossover_rate_min: float = 0.5, crossover_rate_max: float = 0.9):
        """
        初始化自适应GA算法
        
        Args:
            bounds: 参数边界
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            population_size: 种群大小
            mutation_rate_min, mutation_rate_max: 变异率范围
            crossover_rate_min, crossover_rate_max: 交叉率范围
        """
        super().__init__(bounds, objective_func, max_iterations, population_size)
        self.mutation_rate_min = mutation_rate_min
        self.mutation_rate_max = mutation_rate_max
        self.crossover_rate_min = crossover_rate_min
        self.crossover_rate_max = crossover_rate_max
        
        # 自适应参数
        self.fitness_variance_history = []
        
    def calculate_fitness_variance(self) -> float:
        """计算适应度方差"""
        if len(self.fitness) < 2:
            return 0.0
        return np.var(self.fitness)
    
    def adaptive_parameters(self, iteration: int):
        """自适应调整参数"""
        # 计算适应度方差
        fitness_variance = self.calculate_fitness_variance()
        self.fitness_variance_history.append(fitness_variance)
        
        # 根据适应度方差调整变异率
        if fitness_variance < 1e-6:  # 收敛过早
            self.mutation_rate = min(self.mutation_rate_max, self.mutation_rate + 0.05)
        elif fitness_variance > 1e-3:  # 多样性过高
            self.mutation_rate = max(self.mutation_rate_min, self.mutation_rate - 0.05)
        
        # 根据迭代进度调整交叉率
        self.crossover_rate = self.crossover_rate_max - (self.crossover_rate_max - self.crossover_rate_min) * iteration / self.max_iterations
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行自适应GA优化"""
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # 自适应参数调整
            self.adaptive_parameters(iteration)
            
            new_population = []
            new_fitness = []
            
            # 精英保留策略
            best_idx = min(range(self.population_size), key=lambda x: self.fitness[x])
            new_population.append(self.population[best_idx].copy())
            new_fitness.append(self.fitness[best_idx])
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择
                parent1_idx = self.tournament_selection()
                parent2_idx = self.tournament_selection()
                
                # 交叉
                child1, child2 = self.crossover(self.population[parent1_idx], 
                                               self.population[parent2_idx])
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # 评估适应度
                fitness1 = self.evaluate_fitness(child1)
                fitness2 = self.evaluate_fitness(child2)
                
                new_population.extend([child1, child2])
                new_fitness.extend([fitness1, fitness2])
            
            # 保持种群大小
            if len(new_population) > self.population_size:
                new_population = new_population[:self.population_size]
                new_fitness = new_fitness[:self.population_size]
            
            self.population = new_population
            self.fitness = new_fitness
            
            # 更新最优解
            best_idx = min(range(self.population_size), key=lambda x: self.fitness[x])
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_solution = self.population[best_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness

class AlgorithmFusionOptimizer:
    """算法融合优化器"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable,
                 max_iterations: int = 100, population_size: int = 50,
                 fusion_strategy: str = "adaptive"):
        """
        初始化算法融合优化器
        
        Args:
            bounds: 参数边界
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            population_size: 种群大小
            fusion_strategy: 融合策略 ("adaptive", "parallel", "sequential", "hybrid")
        """
        self.bounds = bounds
        self.objective_func = objective_func
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.fusion_strategy = fusion_strategy
        
        # 初始化算法
        self.algorithms = {
            AlgorithmType.PSO: AdaptivePSO(bounds, objective_func, max_iterations, population_size),
            AlgorithmType.DE: AdaptiveDE(bounds, objective_func, max_iterations, population_size),
            AlgorithmType.GA: AdaptiveGA(bounds, objective_func, max_iterations, population_size)
        }
        
        # 性能记录
        self.performance_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.execution_time = 0
        
    def evaluate_algorithm_performance(self, algorithm: OptimizationAlgorithm, 
                                     algorithm_type: AlgorithmType) -> AlgorithmPerformance:
        """评估算法性能"""
        # 计算解的稳定性（基于适应度历史）
        if len(algorithm.fitness_history) > 10:
            recent_fitness = algorithm.fitness_history[-10:]
            stability = 1.0 / (1.0 + np.std(recent_fitness))
        else:
            stability = 0.0
        
        # 计算解的多样性（基于最终种群）
        if hasattr(algorithm, 'particles') and len(algorithm.particles) > 1:
            # PSO算法
            positions = np.array(algorithm.particles)
            diversity = np.mean(np.std(positions, axis=0))
        elif hasattr(algorithm, 'population') and len(algorithm.population) > 1:
            # DE/GA算法
            positions = np.array(algorithm.population)
            diversity = np.mean(np.std(positions, axis=0))
        else:
            diversity = 0.0
        
        # 计算成功率（基于收敛情况）
        if len(algorithm.fitness_history) > 5:
            improvement = algorithm.fitness_history[0] - algorithm.fitness_history[-1]
            success_rate = min(1.0, improvement / max(abs(algorithm.fitness_history[0]), 1e-10))
        else:
            success_rate = 0.0
        
        return AlgorithmPerformance(
            algorithm_type=algorithm_type,
            best_fitness=algorithm.best_fitness,
            execution_time=algorithm.execution_time,
            convergence_iterations=len(algorithm.fitness_history),
            stability=stability,
            diversity=diversity,
            success_rate=success_rate
        )
    
    def select_best_algorithm(self, performances: List[AlgorithmPerformance]) -> AlgorithmType:
        """选择最佳算法"""
        # 综合评分
        scores = []
        for perf in performances:
            # 归一化评分（越小越好）
            fitness_score = 1.0 / (1.0 + perf.best_fitness)
            time_score = 1.0 / (1.0 + perf.execution_time)
            stability_score = perf.stability
            diversity_score = min(1.0, perf.diversity)
            success_score = perf.success_rate
            
            # 加权综合评分
            total_score = (0.4 * fitness_score + 0.2 * time_score + 
                          0.2 * stability_score + 0.1 * diversity_score + 0.1 * success_score)
            scores.append(total_score)
        
        best_idx = np.argmax(scores)
        return performances[best_idx].algorithm_type
    
    def adaptive_fusion_optimize(self) -> Tuple[np.ndarray, float]:
        """自适应融合优化"""
        start_time = time.time()
        
        # 第一阶段：并行运行所有算法
        print("第一阶段：并行运行所有算法...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for algo_type, algorithm in self.algorithms.items():
                future = executor.submit(algorithm.optimize)
                futures[future] = algo_type
        
            # 收集结果
            performances = []
            for future, algo_type in futures.items():
                try:
                    solution, fitness = future.result(timeout=300)  # 5分钟超时
                    performance = self.evaluate_algorithm_performance(
                        self.algorithms[algo_type], algo_type)
                    performances.append(performance)
                    
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = solution
                        
                except Exception as e:
                    print(f"算法 {algo_type.value} 执行失败: {e}")
        
        # 第二阶段：选择最佳算法进行精细优化
        if performances:
            best_algo_type = self.select_best_algorithm(performances)
            print(f"第二阶段：使用最佳算法 {best_algo_type.value} 进行精细优化...")
            
            # 使用最佳算法的结果作为初始解
            best_algorithm = self.algorithms[best_algo_type]
            
            # 精细优化（减少迭代次数，提高精度）
            fine_tuned_algo = type(best_algorithm)(
                self.bounds, self.objective_func, 
                max_iterations=self.max_iterations // 2, 
                population_size=self.population_size
            )
            
            # 使用最佳解初始化种群
            if hasattr(fine_tuned_algo, 'particles'):
                # PSO算法
                for i in range(len(fine_tuned_algo.particles)):
                    # 在最佳解附近添加噪声
                    noise = np.random.normal(0, 0.01, len(self.best_solution))
                    fine_tuned_algo.particles[i] = self.best_solution + noise
                    # 边界处理
                    for j in range(len(self.bounds)):
                        fine_tuned_algo.particles[i][j] = np.clip(
                            fine_tuned_algo.particles[i][j], 
                            self.bounds[j][0], self.bounds[j][1])
            
            solution, fitness = fine_tuned_algo.optimize()
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness
    
    def parallel_fusion_optimize(self) -> Tuple[np.ndarray, float]:
        """并行融合优化"""
        start_time = time.time()
        
        print("并行运行所有算法...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for algo_type, algorithm in self.algorithms.items():
                future = executor.submit(algorithm.optimize)
                futures[future] = algo_type
            
            # 收集结果
            for future, algo_type in futures.items():
                try:
                    solution, fitness = future.result(timeout=300)
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = solution
                        
                except Exception as e:
                    print(f"算法 {algo_type.value} 执行失败: {e}")
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness
    
    def sequential_fusion_optimize(self) -> Tuple[np.ndarray, float]:
        """顺序融合优化"""
        start_time = time.time()
        
        # 按顺序运行算法，每个算法的结果作为下一个算法的初始解
        current_solution = None
        current_fitness = float('inf')
        
        for algo_type, algorithm in self.algorithms.items():
            print(f"运行算法: {algo_type.value}")
            
            # 如果有当前最佳解，用它初始化算法
            if current_solution is not None:
                if hasattr(algorithm, 'particles'):
                    # PSO算法
                    for i in range(len(algorithm.particles)):
                        noise = np.random.normal(0, 0.01, len(current_solution))
                        algorithm.particles[i] = current_solution + noise
                        # 边界处理
                        for j in range(len(self.bounds)):
                            algorithm.particles[i][j] = np.clip(
                                algorithm.particles[i][j], 
                                self.bounds[j][0], self.bounds[j][1])
            
            solution, fitness = algorithm.optimize()
            
            if fitness < current_fitness:
                current_fitness = fitness
                current_solution = solution
        
        self.best_solution = current_solution
        self.best_fitness = current_fitness
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness
    
    def hybrid_fusion_optimize(self) -> Tuple[np.ndarray, float]:
        """混合融合优化"""
        start_time = time.time()
        
        # 第一阶段：快速探索（并行运行，较少迭代）
        print("第一阶段：快速探索...")
        exploration_iterations = self.max_iterations // 3
        
        exploration_algorithms = {
            AlgorithmType.PSO: AdaptivePSO(self.bounds, self.objective_func, 
                                         exploration_iterations, self.population_size),
            AlgorithmType.DE: AdaptiveDE(self.bounds, self.objective_func, 
                                       exploration_iterations, self.population_size),
            AlgorithmType.GA: AdaptiveGA(self.bounds, self.objective_func, 
                                       exploration_iterations, self.population_size)
        }
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for algo_type, algorithm in exploration_algorithms.items():
                future = executor.submit(algorithm.optimize)
                futures[future] = algo_type
            
            # 收集探索结果
            exploration_results = {}
            for future, algo_type in futures.items():
                try:
                    solution, fitness = future.result(timeout=300)
                    exploration_results[algo_type] = (solution, fitness)
                    
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = solution
                        
                except Exception as e:
                    print(f"探索算法 {algo_type.value} 执行失败: {e}")
        
        # 第二阶段：精细优化（选择最佳算法进行深度优化）
        if exploration_results:
            best_algo_type = min(exploration_results.keys(), 
                               key=lambda x: exploration_results[x][1])
            print(f"第二阶段：使用 {best_algo_type.value} 进行精细优化...")
            
            # 精细优化
            fine_tuned_algo = type(exploration_algorithms[best_algo_type])(
                self.bounds, self.objective_func, 
                max_iterations=self.max_iterations - exploration_iterations, 
                population_size=self.population_size
            )
            
            # 使用最佳解初始化
            if hasattr(fine_tuned_algo, 'particles'):
                for i in range(len(fine_tuned_algo.particles)):
                    noise = np.random.normal(0, 0.005, len(self.best_solution))
                    fine_tuned_algo.particles[i] = self.best_solution + noise
                    # 边界处理
                    for j in range(len(self.bounds)):
                        fine_tuned_algo.particles[i][j] = np.clip(
                            fine_tuned_algo.particles[i][j], 
                            self.bounds[j][0], self.bounds[j][1])
            
            solution, fitness = fine_tuned_algo.optimize()
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行融合优化"""
        if self.fusion_strategy == "adaptive":
            return self.adaptive_fusion_optimize()
        elif self.fusion_strategy == "parallel":
            return self.parallel_fusion_optimize()
        elif self.fusion_strategy == "sequential":
            return self.sequential_fusion_optimize()
        elif self.fusion_strategy == "hybrid":
            return self.hybrid_fusion_optimize()
        else:
            raise ValueError(f"未知的融合策略: {self.fusion_strategy}")
    
    def get_performance_summary(self) -> Dict:
        """获取性能总结"""
        return {
            'best_fitness': self.best_fitness,
            'execution_time': self.execution_time,
            'fusion_strategy': self.fusion_strategy,
            'algorithms_used': list(self.algorithms.keys()),
            'best_solution': self.best_solution.tolist() if self.best_solution is not None else None
        }

def compare_fusion_strategies(bounds: List[Tuple[float, float]], 
                             objective_func: Callable,
                             max_iterations: int = 100, 
                             population_size: int = 50) -> Dict:
    """比较不同融合策略的性能"""
    strategies = ["adaptive", "parallel", "sequential", "hybrid"]
    results = {}
    
    for strategy in strategies:
        print(f"\n测试融合策略: {strategy}")
        optimizer = AlgorithmFusionOptimizer(
            bounds, objective_func, max_iterations, population_size, strategy)
        
        start_time = time.time()
        solution, fitness = optimizer.optimize()
        execution_time = time.time() - start_time
        
        results[strategy] = {
            'best_fitness': fitness,
            'execution_time': execution_time,
            'solution': solution.tolist() if solution is not None else None
        }
        
        print(f"策略 {strategy}: 适应度={fitness:.6f}, 时间={execution_time:.2f}s")
    
    return results

def plot_fusion_comparison(results: Dict):
    """绘制融合策略比较图"""
    strategies = list(results.keys())
    fitnesses = [results[s]['best_fitness'] for s in strategies]
    times = [results[s]['execution_time'] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 适应度比较
    ax1.bar(strategies, fitnesses)
    ax1.set_title('融合策略适应度比较')
    ax1.set_ylabel('适应度')
    ax1.tick_params(axis='x', rotation=45)
    
    # 执行时间比较
    ax2.bar(strategies, times)
    ax2.set_title('融合策略执行时间比较')
    ax2.set_ylabel('执行时间 (s)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试代码
    def test_objective(x):
        """测试目标函数"""
        return np.sum(x**2) + np.sin(np.sum(x))
    
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    # 测试融合优化器
    optimizer = AlgorithmFusionOptimizer(bounds, test_objective, 50, 30, "adaptive")
    solution, fitness = optimizer.optimize()
    
    print(f"融合优化结果:")
    print(f"最优解: {solution}")
    print(f"最优适应度: {fitness:.6f}")
    print(f"执行时间: {optimizer.execution_time:.2f}s")
    
    # 比较不同策略
    results = compare_fusion_strategies(bounds, test_objective, 30, 20)
    plot_fusion_comparison(results)

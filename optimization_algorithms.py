"""
优化算法模块
实现PSO、DE、GA、NSGA-II等优化算法用于亥姆霍兹线圈设计
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Callable, Optional
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class OptimizationAlgorithm(ABC):
    """优化算法基类"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable, 
                 max_iterations: int = 100, population_size: int = 50):
        """
        初始化优化算法
        
        Args:
            bounds: 参数边界 [(min1, max1), (min2, max2), ...]
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            population_size: 种群大小
        """
        self.bounds = bounds
        self.objective_func = objective_func
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.dimension = len(bounds)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.execution_time = 0
    
    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行优化"""
        pass
    
    def evaluate_fitness(self, individual: np.ndarray) -> float:
        """评估个体适应度"""
        try:
            # 边界检查
            for i, (min_val, max_val) in enumerate(self.bounds):
                if individual[i] < min_val or individual[i] > max_val:
                    return float('inf')
            
            result = self.objective_func(individual)
            
            # 检查结果有效性
            if np.isnan(result) or np.isinf(result):
                return float('inf')
                
            return result
        except Exception as e:
            print(f"适应度评估错误: {e}")
            return float('inf')
    
    def parallel_evaluate_fitness(self, individuals: List[np.ndarray], 
                                 max_workers: int = None) -> List[float]:
        """并行评估多个个体的适应度"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(individuals))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.evaluate_fitness, ind) for ind in individuals]
            results = [future.result() for future in futures]
        
        return results

class PSO(OptimizationAlgorithm):
    """粒子群优化算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable,
                 max_iterations: int = 100, population_size: int = 50,
                 w: float = 0.9, c1: float = 2.0, c2: float = 2.0):
        """
        初始化PSO算法
        
        Args:
            bounds: 参数边界
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            population_size: 种群大小
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
        """
        super().__init__(bounds, objective_func, max_iterations, population_size)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # 初始化粒子群
        self.particles = []
        self.velocities = []
        self.personal_best = []
        self.personal_best_fitness = []
        
        for i in range(population_size):
            # 随机初始化粒子位置
            particle = np.array([random.uniform(bounds[j][0], bounds[j][1]) 
                                for j in range(self.dimension)])
            self.particles.append(particle)
            
            # 初始化速度
            velocity = np.array([random.uniform(-0.1, 0.1) for _ in range(self.dimension)])
            self.velocities.append(velocity)
            
            # 初始化个体最优
            self.personal_best.append(particle.copy())
            fitness = self.evaluate_fitness(particle)
            self.personal_best_fitness.append(fitness)
            
            # 更新全局最优
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = particle.copy()
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行PSO优化"""
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            for i in range(self.population_size):
                # 更新速度
                r1 = random.random()
                r2 = random.random()
                
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
            
            # 动态调整惯性权重
            self.w = 0.9 - 0.5 * iteration / self.max_iterations
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness

class DE(OptimizationAlgorithm):
    """差分进化算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable,
                 max_iterations: int = 100, population_size: int = 50,
                 F: float = 0.8, CR: float = 0.9):
        """
        初始化DE算法
        
        Args:
            bounds: 参数边界
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            population_size: 种群大小
            F: 缩放因子
            CR: 交叉概率
        """
        super().__init__(bounds, objective_func, max_iterations, population_size)
        self.F = F
        self.CR = CR
        
        # 初始化种群
        self.population = []
        self.fitness = []
        
        for i in range(population_size):
            individual = np.array([random.uniform(bounds[j][0], bounds[j][1]) 
                                 for j in range(self.dimension)])
            self.population.append(individual)
            fitness = self.evaluate_fitness(individual)
            self.fitness.append(fitness)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = individual.copy()
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行DE优化"""
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            new_population = []
            new_fitness = []
            
            for i in range(self.population_size):
                # 选择三个不同的个体
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = random.sample(candidates, 3)
                
                # 变异
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                
                # 交叉
                trial = self.population[i].copy()
                for j in range(self.dimension):
                    if random.random() < self.CR:
                        trial[j] = mutant[j]
                
                # 边界处理
                for j in range(self.dimension):
                    if trial[j] < self.bounds[j][0]:
                        trial[j] = self.bounds[j][0]
                    elif trial[j] > self.bounds[j][1]:
                        trial[j] = self.bounds[j][1]
                
                # 选择
                trial_fitness = self.evaluate_fitness(trial)
                if trial_fitness < self.fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
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

class GA(OptimizationAlgorithm):
    """遗传算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable,
                 max_iterations: int = 100, population_size: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 selection_pressure: float = 2.0):
        """
        初始化GA算法
        
        Args:
            bounds: 参数边界
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            selection_pressure: 选择压力
        """
        super().__init__(bounds, objective_func, max_iterations, population_size)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        
        # 初始化种群
        self.population = []
        self.fitness = []
        
        for i in range(population_size):
            individual = np.array([random.uniform(bounds[j][0], bounds[j][1]) 
                                 for j in range(self.dimension)])
            self.population.append(individual)
            fitness = self.evaluate_fitness(individual)
            self.fitness.append(fitness)
            
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = individual.copy()
    
    def tournament_selection(self, tournament_size: int = 3) -> int:
        """锦标赛选择"""
        candidates = random.sample(range(self.population_size), tournament_size)
        winner = min(candidates, key=lambda x: self.fitness[x])
        return winner
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 算术交叉
        alpha = random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = individual.copy()
        
        for i in range(self.dimension):
            if random.random() < self.mutation_rate:
                # 高斯变异
                sigma = (self.bounds[i][1] - self.bounds[i][0]) * 0.1
                mutated[i] += random.gauss(0, sigma)
                
                # 边界处理
                mutated[i] = max(self.bounds[i][0], min(self.bounds[i][1], mutated[i]))
        
        return mutated
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行GA优化"""
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            new_population = []
            new_fitness = []
            
            # 保留最优个体
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

class NSGA2(OptimizationAlgorithm):
    """NSGA-II多目标优化算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_funcs: List[Callable],
                 max_iterations: int = 100, population_size: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        """
        初始化NSGA-II算法
        
        Args:
            bounds: 参数边界
            objective_funcs: 多目标函数列表
            max_iterations: 最大迭代次数
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
        """
        super().__init__(bounds, objective_funcs[0], max_iterations, population_size)
        self.objective_funcs = objective_funcs
        self.num_objectives = len(objective_funcs)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # 初始化种群
        self.population = []
        self.objectives = []
        
        for i in range(population_size):
            individual = np.array([random.uniform(bounds[j][0], bounds[j][1]) 
                                 for j in range(self.dimension)])
            self.population.append(individual)
            objectives = [func(individual) for func in objective_funcs]
            self.objectives.append(objectives)
    
    def evaluate_objectives(self, individual: np.ndarray) -> List[float]:
        """评估多目标函数"""
        try:
            return [func(individual) for func in self.objective_funcs]
        except:
            return [float('inf')] * self.num_objectives
    
    def dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """判断obj1是否支配obj2"""
        better_in_one = False
        for i in range(self.num_objectives):
            if obj1[i] > obj2[i]:  # 假设都是最小化问题
                return False
            elif obj1[i] < obj2[i]:
                better_in_one = True
        return better_in_one
    
    def fast_non_dominated_sort(self) -> List[List[int]]:
        """快速非支配排序"""
        fronts = []
        dominated_count = [0] * self.population_size
        dominated_solutions = [[] for _ in range(self.population_size)]
        
        # 计算支配关系
        for i in range(self.population_size):
            for j in range(self.population_size):
                if i != j:
                    if self.dominates(self.objectives[i], self.objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(self.objectives[j], self.objectives[i]):
                        dominated_count[i] += 1
        
        # 构建第一前沿
        front_0 = []
        for i in range(self.population_size):
            if dominated_count[i] == 0:
                front_0.append(i)
        
        fronts.append(front_0)
        
        # 构建后续前沿
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return fronts
    
    def crowding_distance(self, front: List[int]) -> List[float]:
        """计算拥挤距离"""
        distances = [0.0] * len(front)
        
        for m in range(self.num_objectives):
            # 按第m个目标函数排序
            sorted_front = sorted(front, key=lambda x: self.objectives[x][m])
            
            # 边界点距离设为无穷大
            distances[0] = float('inf')
            distances[-1] = float('inf')
            
            # 计算其他点的拥挤距离
            obj_min = self.objectives[sorted_front[0]][m]
            obj_max = self.objectives[sorted_front[-1]][m]
            
            if obj_max - obj_min > 0:
                for i in range(1, len(front) - 1):
                    distances[i] += (self.objectives[sorted_front[i+1]][m] - 
                                   self.objectives[sorted_front[i-1]][m]) / (obj_max - obj_min)
        
        return distances
    
    def tournament_selection(self, tournament_size: int = 2) -> int:
        """锦标赛选择"""
        candidates = random.sample(range(self.population_size), tournament_size)
        
        # 比较前沿等级和拥挤距离
        best = candidates[0]
        for candidate in candidates[1:]:
            if (self.front_rank[candidate] < self.front_rank[best] or
                (self.front_rank[candidate] == self.front_rank[best] and 
                 self.crowding_dist[candidate] > self.crowding_dist[best])):
                best = candidate
        
        return best
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 模拟二进制交叉
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(self.dimension):
            if random.random() < 0.5:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = individual.copy()
        
        for i in range(self.dimension):
            if random.random() < self.mutation_rate:
                # 多项式变异
                sigma = (self.bounds[i][1] - self.bounds[i][0]) * 0.1
                mutated[i] += random.gauss(0, sigma)
                
                # 边界处理
                mutated[i] = max(self.bounds[i][0], min(self.bounds[i][1], mutated[i]))
        
        return mutated
    
    def optimize(self) -> Tuple[List[np.ndarray], List[List[float]]]:
        """执行NSGA-II优化"""
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # 非支配排序
            fronts = self.fast_non_dominated_sort()
            
            # 计算前沿等级和拥挤距离
            self.front_rank = [0] * self.population_size
            self.crowding_dist = [0.0] * self.population_size
            
            for rank, front in enumerate(fronts):
                for idx in front:
                    self.front_rank[idx] = rank
                
                if len(front) > 1:
                    distances = self.crowding_distance(front)
                    for i, idx in enumerate(front):
                        self.crowding_dist[idx] = distances[i]
            
            # 生成新种群
            new_population = []
            new_objectives = []
            
            # 选择父代
            for _ in range(self.population_size // 2):
                parent1_idx = self.tournament_selection()
                parent2_idx = self.tournament_selection()
                
                # 交叉
                child1, child2 = self.crossover(self.population[parent1_idx], 
                                               self.population[parent2_idx])
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # 评估目标函数
                obj1 = self.evaluate_objectives(child1)
                obj2 = self.evaluate_objectives(child2)
                
                new_population.extend([child1, child2])
                new_objectives.extend([obj1, obj2])
            
            # 合并父代和子代
            combined_population = self.population + new_population
            combined_objectives = self.objectives + new_objectives
            
            # 环境选择
            fronts = self.fast_non_dominated_sort()
            
            self.population = []
            self.objectives = []
            
            for front in fronts:
                if len(self.population) + len(front) <= self.population_size:
                    self.population.extend([combined_population[i] for i in front])
                    self.objectives.extend([combined_objectives[i] for i in front])
                else:
                    # 需要部分选择
                    remaining = self.population_size - len(self.population)
                    distances = self.crowding_distance(front)
                    sorted_front = sorted(front, key=lambda x: distances[x], reverse=True)
                    
                    for i in range(remaining):
                        idx = sorted_front[i]
                        self.population.append(combined_population[idx])
                        self.objectives.append(combined_objectives[idx])
                    break
        
        self.execution_time = time.time() - start_time
        
        # 返回帕累托前沿
        pareto_front = self.fast_non_dominated_sort()[0]
        pareto_solutions = [self.population[i] for i in pareto_front]
        pareto_objectives = [self.objectives[i] for i in pareto_front]
        
        return pareto_solutions, pareto_objectives

def plot_optimization_results(algorithms: List[OptimizationAlgorithm], title: str = "优化算法比较"):
    """绘制优化结果比较图"""
    plt.figure(figsize=(12, 8))
    
    for i, algorithm in enumerate(algorithms):
        plt.subplot(2, 2, i + 1)
        plt.plot(algorithm.fitness_history)
        plt.title(f"{algorithm.__class__.__name__} - 最优适应度: {algorithm.best_fitness:.6f}")
        plt.xlabel("迭代次数")
        plt.ylabel("适应度")
        plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def compare_algorithms_performance(algorithms: List[OptimizationAlgorithm]) -> Dict:
    """比较算法性能"""
    results = {}
    
    for algorithm in algorithms:
        results[algorithm.__class__.__name__] = {
            'best_fitness': algorithm.best_fitness,
            'execution_time': algorithm.execution_time,
            'convergence_iterations': len(algorithm.fitness_history),
            'final_improvement': algorithm.fitness_history[0] - algorithm.fitness_history[-1] if algorithm.fitness_history else 0
        }
    
    return results

class SimulatedAnnealing(OptimizationAlgorithm):
    """模拟退火算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable,
                 max_iterations: int = 100, initial_temp: float = 100.0,
                 cooling_rate: float = 0.95, min_temp: float = 0.01):
        """
        初始化模拟退火算法
        
        Args:
            bounds: 参数边界
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            initial_temp: 初始温度
            cooling_rate: 冷却率
            min_temp: 最小温度
        """
        super().__init__(bounds, objective_func, max_iterations, 1)  # SA是单个体算法
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        
        # 初始化当前解
        self.current_solution = np.array([random.uniform(bounds[i][0], bounds[i][1]) 
                                        for i in range(self.dimension)])
        self.current_fitness = self.evaluate_fitness(self.current_solution)
        
        if self.current_fitness < self.best_fitness:
            self.best_fitness = self.current_fitness
            self.best_solution = self.current_solution.copy()
    
    def generate_neighbor(self, solution: np.ndarray, temperature: float) -> np.ndarray:
        """生成邻域解"""
        neighbor = solution.copy()
        
        # 根据温度调整扰动幅度
        perturbation_scale = temperature / self.initial_temp
        
        for i in range(self.dimension):
            # 高斯扰动
            sigma = (self.bounds[i][1] - self.bounds[i][0]) * 0.1 * perturbation_scale
            neighbor[i] += random.gauss(0, sigma)
            
            # 边界处理
            neighbor[i] = max(self.bounds[i][0], min(self.bounds[i][1], neighbor[i]))
        
        return neighbor
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行模拟退火优化"""
        start_time = time.time()
        temperature = self.initial_temp
        
        for iteration in range(self.max_iterations):
            # 生成邻域解
            neighbor = self.generate_neighbor(self.current_solution, temperature)
            neighbor_fitness = self.evaluate_fitness(neighbor)
            
            # 接受准则
            delta = neighbor_fitness - self.current_fitness
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                self.current_solution = neighbor
                self.current_fitness = neighbor_fitness
                
                # 更新最优解
                if neighbor_fitness < self.best_fitness:
                    self.best_fitness = neighbor_fitness
                    self.best_solution = neighbor.copy()
            
            # 记录适应度历史
            self.fitness_history.append(self.best_fitness)
            
            # 降温
            temperature *= self.cooling_rate
            
            # 早停条件
            if temperature < self.min_temp:
                break
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness

class ParticleSwarmOptimizationEnhanced(PSO):
    """增强型粒子群优化算法"""
    
    def __init__(self, bounds: List[Tuple[float, float]], objective_func: Callable,
                 max_iterations: int = 100, population_size: int = 50,
                 w: float = 0.9, c1: float = 2.0, c2: float = 2.0,
                 c3: float = 1.0, use_parallel: bool = True):
        """
        初始化增强型PSO算法
        
        Args:
            bounds: 参数边界
            objective_func: 目标函数
            max_iterations: 最大迭代次数
            population_size: 种群大小
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            c3: 全局最优学习因子
            use_parallel: 是否使用并行计算
        """
        super().__init__(bounds, objective_func, max_iterations, population_size, w, c1, c2)
        self.c3 = c3
        self.use_parallel = use_parallel
        
        # 添加全局最优历史
        self.global_best_history = []
        self.diversity_threshold = 0.01
    
    def calculate_swarm_diversity(self) -> float:
        """计算群体多样性"""
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
    
    def adaptive_velocity_update(self, i: int, iteration: int):
        """自适应速度更新"""
        r1 = random.random()
        r2 = random.random()
        r3 = random.random()
        
        # 计算多样性
        diversity = self.calculate_swarm_diversity()
        
        # 自适应调整学习因子
        if diversity < self.diversity_threshold:
            # 多样性不足，增加探索
            adaptive_c1 = self.c1 * 1.2
            adaptive_c2 = self.c2 * 0.8
        else:
            # 多样性充足，增加开发
            adaptive_c1 = self.c1 * 0.8
            adaptive_c2 = self.c2 * 1.2
        
        # 速度更新公式
        cognitive = adaptive_c1 * r1 * (self.personal_best[i] - self.particles[i])
        social = adaptive_c2 * r2 * (self.best_solution - self.particles[i])
        
        # 添加全局最优历史影响
        if len(self.global_best_history) > 0:
            global_influence = self.c3 * r3 * (self.global_best_history[-1] - self.particles[i])
        else:
            global_influence = np.zeros(self.dimension)
        
        self.velocities[i] = (self.w * self.velocities[i] + 
                            cognitive + social + global_influence)
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行增强型PSO优化"""
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # 记录全局最优历史
            self.global_best_history.append(self.best_solution.copy())
            
            # 并行或串行更新粒子
            if self.use_parallel and self.population_size > 10:
                # 并行更新
                with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), self.population_size)) as executor:
                    futures = []
                    for i in range(self.population_size):
                        future = executor.submit(self._update_particle, i, iteration)
                        futures.append(future)
                    
                    # 收集结果
                    for future in futures:
                        future.result()
            else:
                # 串行更新
                for i in range(self.population_size):
                    self._update_particle(i, iteration)
            
            # 记录适应度历史
            self.fitness_history.append(self.best_fitness)
            
            # 动态调整惯性权重
            self.w = 0.9 - 0.5 * iteration / self.max_iterations
            
            # 早停机制
            if len(self.fitness_history) > 20:
                recent_improvement = self.fitness_history[-20] - self.fitness_history[-1]
                if recent_improvement < 1e-10:
                    break
        
        self.execution_time = time.time() - start_time
        return self.best_solution, self.best_fitness
    
    def _update_particle(self, i: int, iteration: int):
        """更新单个粒子"""
        # 自适应速度更新
        self.adaptive_velocity_update(i, iteration)
        
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

def create_optimization_suite(bounds: List[Tuple[float, float]], 
                             objective_func: Callable,
                             max_iterations: int = 100, 
                             population_size: int = 50) -> Dict[str, OptimizationAlgorithm]:
    """创建优化算法套件"""
    algorithms = {
        'PSO': PSO(bounds, objective_func, max_iterations, population_size),
        'PSO_Enhanced': ParticleSwarmOptimizationEnhanced(bounds, objective_func, max_iterations, population_size),
        'DE': DE(bounds, objective_func, max_iterations, population_size),
        'GA': GA(bounds, objective_func, max_iterations, population_size),
        'SA': SimulatedAnnealing(bounds, objective_func, max_iterations)
    }
    
    return algorithms

def run_optimization_comparison(bounds: List[Tuple[float, float]], 
                               objective_func: Callable,
                               max_iterations: int = 100, 
                               population_size: int = 50,
                               num_runs: int = 5) -> Dict:
    """运行优化算法比较"""
    algorithms = create_optimization_suite(bounds, objective_func, max_iterations, population_size)
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"运行算法: {name}")
        
        run_results = []
        for run in range(num_runs):
            # 重新初始化算法
            if name == 'PSO':
                algo = PSO(bounds, objective_func, max_iterations, population_size)
            elif name == 'PSO_Enhanced':
                algo = ParticleSwarmOptimizationEnhanced(bounds, objective_func, max_iterations, population_size)
            elif name == 'DE':
                algo = DE(bounds, objective_func, max_iterations, population_size)
            elif name == 'GA':
                algo = GA(bounds, objective_func, max_iterations, population_size)
            elif name == 'SA':
                algo = SimulatedAnnealing(bounds, objective_func, max_iterations)
            
            solution, fitness = algo.optimize()
            run_results.append({
                'solution': solution,
                'fitness': fitness,
                'execution_time': algo.execution_time,
                'convergence_iterations': len(algo.fitness_history)
            })
        
        # 统计结果
        fitnesses = [r['fitness'] for r in run_results]
        times = [r['execution_time'] for r in run_results]
        
        results[name] = {
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'best_fitness': np.min(fitnesses),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'success_rate': sum(1 for f in fitnesses if f < float('inf')) / len(fitnesses),
            'runs': run_results
        }
    
    return results

def plot_optimization_comparison(results: Dict):
    """绘制优化算法比较图"""
    algorithms = list(results.keys())
    
    # 准备数据
    mean_fitness = [results[alg]['mean_fitness'] for alg in algorithms]
    std_fitness = [results[alg]['std_fitness'] for alg in algorithms]
    mean_time = [results[alg]['mean_time'] for alg in algorithms]
    success_rate = [results[alg]['success_rate'] for alg in algorithms]
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 平均适应度
    ax1.bar(algorithms, mean_fitness, yerr=std_fitness, capsize=5)
    ax1.set_title('平均适应度比较')
    ax1.set_ylabel('适应度')
    ax1.tick_params(axis='x', rotation=45)
    
    # 平均执行时间
    ax2.bar(algorithms, mean_time)
    ax2.set_title('平均执行时间比较')
    ax2.set_ylabel('时间 (s)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 成功率
    ax3.bar(algorithms, success_rate)
    ax3.set_title('成功率比较')
    ax3.set_ylabel('成功率')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    
    # 综合评分
    scores = []
    for alg in algorithms:
        # 归一化评分
        fitness_score = 1.0 / (1.0 + results[alg]['mean_fitness'])
        time_score = 1.0 / (1.0 + results[alg]['mean_time'])
        success_score = results[alg]['success_rate']
        
        # 综合评分
        total_score = 0.5 * fitness_score + 0.3 * time_score + 0.2 * success_score
        scores.append(total_score)
    
    ax4.bar(algorithms, scores)
    ax4.set_title('综合评分比较')
    ax4.set_ylabel('综合评分')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return scores

"""
亥姆霍兹线圈优化目标函数模块
实现多种优化目标函数，用于不同场景的线圈设计优化
"""

import numpy as np
from typing import List, Tuple, Dict, Callable
from helmholtz_coil import MultiHelmholtzSystem, create_optimized_helmholtz_system

class HelmholtzObjectiveFunction:
    """亥姆霍兹线圈目标函数基类"""
    
    def __init__(self, bounds: List[Tuple[float, float]], 
                 region_size: float = 0.02, resolution: int = 20):
        """
        初始化目标函数
        
        Args:
            bounds: 参数边界 [(min1, max1), (min2, max2), ...]
            region_size: 匀场区域大小 (m)
            resolution: 计算分辨率
        """
        self.bounds = bounds
        self.region_size = region_size
        self.resolution = resolution
        
    def create_system(self, params: np.ndarray) -> MultiHelmholtzSystem:
        """根据参数创建线圈系统"""
        return create_optimized_helmholtz_system(params.tolist())
    
    def evaluate_uniformity(self, params: np.ndarray) -> Dict:
        """评估匀场性能"""
        try:
            system = self.create_system(params)
            return system.field_uniformity(self.region_size, self.resolution)
        except Exception as e:
            print(f"评估匀场性能时出错: {e}")
            return {
                'mean_field': 0,
                'std_field': float('inf'),
                'uniformity': 0,
                'direction_consistency': float('inf')
            }

class FieldUniformityObjective(HelmholtzObjectiveFunction):
    """磁场均匀性目标函数"""
    
    def __init__(self, bounds: List[Tuple[float, float]], 
                 target_field: float = 0.1, 
                 region_size: float = 0.02, resolution: int = 20):
        """
        初始化磁场均匀性目标函数
        
        Args:
            bounds: 参数边界
            target_field: 目标磁场强度 (T)
            region_size: 匀场区域大小 (m)
            resolution: 计算分辨率
        """
        super().__init__(bounds, region_size, resolution)
        self.target_field = target_field
    
    def __call__(self, params: np.ndarray) -> float:
        """计算目标函数值"""
        uniformity_result = self.evaluate_uniformity(params)
        
        # 目标函数：最小化磁场不均匀性
        uniformity_penalty = 1 - uniformity_result['uniformity']
        field_deviation = abs(uniformity_result['mean_field'] - self.target_field) / self.target_field
        direction_penalty = uniformity_result['direction_consistency']
        
        # 综合目标函数
        fitness = uniformity_penalty + field_deviation + direction_penalty
        
        return fitness

class FieldStrengthObjective(HelmholtzObjectiveFunction):
    """磁场强度目标函数"""
    
    def __init__(self, bounds: List[Tuple[float, float]], 
                 target_field: float = 0.1,
                 region_size: float = 0.02, resolution: int = 20):
        """
        初始化磁场强度目标函数
        
        Args:
            bounds: 参数边界
            target_field: 目标磁场强度 (T)
            region_size: 匀场区域大小 (m)
            resolution: 计算分辨率
        """
        super().__init__(bounds, region_size, resolution)
        self.target_field = target_field
    
    def __call__(self, params: np.ndarray) -> float:
        """计算目标函数值"""
        uniformity_result = self.evaluate_uniformity(params)
        
        # 目标函数：最小化磁场强度偏差
        field_deviation = abs(uniformity_result['mean_field'] - self.target_field) / self.target_field
        uniformity_penalty = 1 - uniformity_result['uniformity']
        
        # 综合目标函数
        fitness = field_deviation + 0.1 * uniformity_penalty
        
        return fitness

class MultiObjectiveHelmholtz(HelmholtzObjectiveFunction):
    """多目标亥姆霍兹线圈优化"""
    
    def __init__(self, bounds: List[Tuple[float, float]], 
                 target_field: float = 0.1,
                 region_size: float = 0.02, resolution: int = 20):
        """
        初始化多目标函数
        
        Args:
            bounds: 参数边界
            target_field: 目标磁场强度 (T)
            region_size: 匀场区域大小 (m)
            resolution: 计算分辨率
        """
        super().__init__(bounds, region_size, resolution)
        self.target_field = target_field
    
    def __call__(self, params: np.ndarray) -> List[float]:
        """计算多目标函数值"""
        uniformity_result = self.evaluate_uniformity(params)
        
        # 目标1：磁场强度偏差
        field_deviation = abs(uniformity_result['mean_field'] - self.target_field) / self.target_field
        
        # 目标2：磁场不均匀性
        uniformity_penalty = 1 - uniformity_result['uniformity']
        
        # 目标3：方向一致性
        direction_penalty = uniformity_result['direction_consistency']
        
        return [field_deviation, uniformity_penalty, direction_penalty]

class EfficiencyObjective(HelmholtzObjectiveFunction):
    """效率优化目标函数"""
    
    def __init__(self, bounds: List[Tuple[float, float]], 
                 target_field: float = 0.1,
                 region_size: float = 0.02, resolution: int = 20):
        """
        初始化效率目标函数
        
        Args:
            bounds: 参数边界
            target_field: 目标磁场强度 (T)
            region_size: 匀场区域大小 (m)
            resolution: 计算分辨率
        """
        super().__init__(bounds, region_size, resolution)
        self.target_field = target_field
    
    def calculate_power_consumption(self, params: np.ndarray) -> float:
        """计算功耗"""
        total_power = 0
        for i in range(0, len(params), 4):
            if i + 1 < len(params):
                current = params[i + 1]
                turns = int(params[i + 2])
                # 假设线圈电阻与匝数成正比
                resistance = turns * 0.01  # 每匝0.01欧姆
                power = current**2 * resistance
                total_power += power
        return total_power
    
    def __call__(self, params: np.ndarray) -> float:
        """计算目标函数值"""
        uniformity_result = self.evaluate_uniformity(params)
        
        # 目标函数：最小化功耗，同时保证磁场质量
        field_deviation = abs(uniformity_result['mean_field'] - self.target_field) / self.target_field
        uniformity_penalty = 1 - uniformity_result['uniformity']
        power_consumption = self.calculate_power_consumption(params)
        
        # 综合目标函数（权重平衡）
        fitness = 0.4 * field_deviation + 0.3 * uniformity_penalty + 0.3 * power_consumption / 1000
        
        return fitness

class RobustObjective(HelmholtzObjectiveFunction):
    """鲁棒性目标函数"""
    
    def __init__(self, bounds: List[Tuple[float, float]], 
                 target_field: float = 0.1,
                 region_size: float = 0.02, resolution: int = 20,
                 noise_level: float = 0.05):
        """
        初始化鲁棒性目标函数
        
        Args:
            bounds: 参数边界
            target_field: 目标磁场强度 (T)
            region_size: 匀场区域大小 (m)
            resolution: 计算分辨率
            noise_level: 噪声水平
        """
        super().__init__(bounds, region_size, resolution)
        self.target_field = target_field
        self.noise_level = noise_level
    
    def __call__(self, params: np.ndarray) -> float:
        """计算鲁棒性目标函数值"""
        # 添加噪声测试鲁棒性
        noisy_params = params + np.random.normal(0, self.noise_level, len(params))
        
        # 边界处理
        for i, (min_val, max_val) in enumerate(self.bounds):
            noisy_params[i] = np.clip(noisy_params[i], min_val, max_val)
        
        uniformity_result = self.evaluate_uniformity(noisy_params)
        
        # 目标函数：在噪声条件下仍能保持良好性能
        field_deviation = abs(uniformity_result['mean_field'] - self.target_field) / self.target_field
        uniformity_penalty = 1 - uniformity_result['uniformity']
        
        # 鲁棒性惩罚
        robustness_penalty = np.std(noisy_params) / np.mean(np.abs(params))
        
        fitness = field_deviation + uniformity_penalty + 0.1 * robustness_penalty
        
        return fitness

def create_helmholtz_bounds(num_pairs: int = 2) -> List[Tuple[float, float]]:
    """
    创建亥姆霍兹线圈参数边界
    
    Args:
        num_pairs: 线圈对数量
        
    Returns:
        参数边界列表
    """
    bounds = []
    
    for _ in range(num_pairs):
        # 半径边界 (m)
        bounds.append((0.05, 0.5))  # 5cm - 50cm
        # 电流边界 (A)
        bounds.append((0.1, 10.0))   # 0.1A - 10A
        # 匝数边界
        bounds.append((10, 1000))   # 10 - 1000匝
        # 间距边界 (m)
        bounds.append((0.05, 0.5))  # 5cm - 50cm
    
    return bounds

def benchmark_objective_functions(bounds: List[Tuple[float, float]], 
                                 test_params: np.ndarray) -> Dict:
    """基准测试不同目标函数"""
    objectives = {
        'uniformity': FieldUniformityObjective(bounds),
        'strength': FieldStrengthObjective(bounds),
        'efficiency': EfficiencyObjective(bounds),
        'robust': RobustObjective(bounds)
    }
    
    results = {}
    for name, obj_func in objectives.items():
        try:
            start_time = time.time()
            fitness = obj_func(test_params)
            eval_time = time.time() - start_time
            
            results[name] = {
                'fitness': fitness,
                'evaluation_time': eval_time
            }
        except Exception as e:
            results[name] = {
                'fitness': float('inf'),
                'evaluation_time': 0,
                'error': str(e)
            }
    
    return results

if __name__ == "__main__":
    import time
    
    # 测试代码
    bounds = create_helmholtz_bounds(1)  # 1对线圈
    test_params = np.array([0.1, 1.0, 100, 0.1])  # 测试参数
    
    print("亥姆霍兹线圈目标函数测试:")
    print(f"测试参数: {test_params}")
    
    # 测试不同目标函数
    results = benchmark_objective_functions(bounds, test_params)
    
    for name, result in results.items():
        print(f"{name}: 适应度={result['fitness']:.6f}, 时间={result['evaluation_time']:.4f}s")

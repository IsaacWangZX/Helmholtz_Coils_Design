"""
亥姆霍兹线圈磁场计算模块
实现多对亥姆霍兹线圈的磁场计算和匀场性能评估
"""

import numpy as np
from scipy.integrate import quad
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HelmholtzCoil:
    """亥姆霍兹线圈类"""
    
    def __init__(self, radius: float, current: float, turns: int, position: float):
        """
        初始化亥姆霍兹线圈
        
        Args:
            radius: 线圈半径 (m)
            current: 电流 (A)
            turns: 匝数
            position: 线圈位置 (m)
        """
        self.radius = radius
        self.current = current
        self.turns = turns
        self.position = position
        self.mu0 = 4 * np.pi * 1e-7  # 真空磁导率
    
    def magnetic_field_single_coil(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        计算单个线圈在点(x,y,z)处的磁场
        
        Args:
            x, y, z: 空间坐标 (m)
            
        Returns:
            (Bx, By, Bz): 磁场分量 (T)
        """
        # 线圈中心在z轴上，位置为self.position
        z_coil = self.position
        
        # 计算到线圈中心的距离
        rho = np.sqrt(x**2 + y**2)
        z_rel = z - z_coil
        
        # 使用椭圆积分计算磁场
        if rho == 0 and z_rel == 0:
            # 在线圈中心处
            Bx = By = 0
            Bz = self.mu0 * self.current * self.turns / (2 * self.radius)
        else:
            # 使用Biot-Savart定律的解析解
            k_squared = 4 * self.radius * rho / ((self.radius + rho)**2 + z_rel**2)
            
            if k_squared >= 1:
                k_squared = 0.999999  # 避免数值问题
            
            k = np.sqrt(k_squared)
            
            # 计算椭圆积分
            from scipy.special import ellipk, ellipe
            
            K = ellipk(k_squared)
            E = ellipe(k_squared)
            
            # 磁场分量
            factor = self.mu0 * self.current * self.turns / (2 * np.pi * np.sqrt((self.radius + rho)**2 + z_rel**2))
            
            B_rho = factor * z_rel / rho * ((self.radius**2 + rho**2 + z_rel**2) / ((self.radius - rho)**2 + z_rel**2) * E - K)
            B_z = factor * ((self.radius**2 - rho**2 - z_rel**2) / ((self.radius - rho)**2 + z_rel**2) * E + K)
            
            # 转换到笛卡尔坐标系
            if rho > 0:
                Bx = B_rho * x / rho
                By = B_rho * y / rho
            else:
                Bx = By = 0
            
            Bz = B_z
        
        return Bx, By, Bz

class MultiHelmholtzSystem:
    """多对亥姆霍兹线圈系统"""
    
    def __init__(self, coil_pairs: List[List[HelmholtzCoil]]):
        """
        初始化多对亥姆霍兹线圈系统
        
        Args:
            coil_pairs: 线圈对列表，每个元素是一对线圈
        """
        self.coil_pairs = coil_pairs
        self.total_coils = sum(len(pair) for pair in coil_pairs)
    
    def magnetic_field(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        计算系统在点(x,y,z)处的总磁场
        
        Args:
            x, y, z: 空间坐标 (m)
            
        Returns:
            (Bx, By, Bz): 总磁场分量 (T)
        """
        Bx_total = By_total = Bz_total = 0
        
        for pair in self.coil_pairs:
            for coil in pair:
                Bx, By, Bz = coil.magnetic_field_single_coil(x, y, z)
                Bx_total += Bx
                By_total += By
                Bz_total += Bz
        
        return Bx_total, By_total, Bz_total
    
    def field_magnitude(self, x: float, y: float, z: float) -> float:
        """计算磁场强度"""
        Bx, By, Bz = self.magnetic_field(x, y, z)
        return np.sqrt(Bx**2 + By**2 + Bz**2)
    
    def field_uniformity(self, region_size: float = 0.02, resolution: int = 20) -> Dict:
        """
        计算匀场性能
        
        Args:
            region_size: 匀场区域大小 (m)，默认2cm
            resolution: 计算分辨率
            
        Returns:
            包含匀场性能指标的字典
        """
        # 定义匀场区域
        x_range = np.linspace(-region_size/2, region_size/2, resolution)
        y_range = np.linspace(-region_size/2, region_size/2, resolution)
        z_range = np.linspace(-region_size/2, region_size/2, resolution)
        
        # 计算磁场分布
        field_values = []
        field_vectors = []
        
        for z in z_range:
            for y in y_range:
                for x in x_range:
                    Bx, By, Bz = self.magnetic_field(x, y, z)
                    field_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
                    field_values.append(field_mag)
                    field_vectors.append([Bx, By, Bz])
        
        field_values = np.array(field_values)
        field_vectors = np.array(field_vectors)
        
        # 计算匀场性能指标
        mean_field = np.mean(field_values)
        std_field = np.std(field_values)
        uniformity = 1 - std_field / mean_field if mean_field > 0 else 0
        
        # 计算磁场方向一致性
        field_directions = field_vectors / (field_values[:, np.newaxis] + 1e-10)
        direction_consistency = np.mean(np.linalg.norm(field_directions - np.mean(field_directions, axis=0), axis=1))
        
        return {
            'mean_field': mean_field,
            'std_field': std_field,
            'uniformity': uniformity,
            'direction_consistency': direction_consistency,
            'field_values': field_values,
            'field_vectors': field_vectors
        }
    
    def visualize_field(self, region_size: float = 0.02, resolution: int = 20):
        """可视化磁场分布"""
        x_range = np.linspace(-region_size/2, region_size/2, resolution)
        y_range = np.linspace(-region_size/2, region_size/2, resolution)
        z_range = np.linspace(-region_size/2, region_size/2, resolution)
        
        # 创建网格
        X, Y, Z = np.meshgrid(x_range, y_range, z_range)
        
        # 计算磁场
        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)
        Bz = np.zeros_like(Z)
        
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    bx, by, bz = self.magnetic_field(X[i,j,k], Y[i,j,k], Z[i,j,k])
                    Bx[i,j,k] = bx
                    By[i,j,k] = by
                    Bz[i,j,k] = bz
        
        # 绘制磁场分布
        fig = plt.figure(figsize=(15, 5))
        
        # 磁场强度分布
        ax1 = fig.add_subplot(131, projection='3d')
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        ax1.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=B_mag.flatten(), cmap='viridis')
        ax1.set_title('磁场强度分布')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 磁场矢量分布
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.quiver(X[::2,::2,::2], Y[::2,::2,::2], Z[::2,::2,::2], 
                  Bx[::2,::2,::2], By[::2,::2,::2], Bz[::2,::2,::2], 
                  length=0.001, normalize=True)
        ax2.set_title('磁场矢量分布')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        
        # 匀场区域截面
        ax3 = fig.add_subplot(133)
        z_mid = resolution // 2
        im = ax3.imshow(B_mag[:, :, z_mid], extent=[-region_size/2, region_size/2, -region_size/2, region_size/2], 
                       cmap='viridis', origin='lower')
        ax3.set_title('Z=0截面磁场强度')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        plt.show()

def create_helmholtz_pair(radius: float, current: float, turns: int, separation: float) -> List[HelmholtzCoil]:
    """
    创建一对亥姆霍兹线圈
    
    Args:
        radius: 线圈半径 (m)
        current: 电流 (A)
        turns: 匝数
        separation: 线圈间距 (m)
        
    Returns:
        包含两个线圈的列表
    """
    coil1 = HelmholtzCoil(radius, current, turns, -separation/2)
    coil2 = HelmholtzCoil(radius, current, turns, separation/2)
    return [coil1, coil2]

def create_optimized_helmholtz_system(params: List[float]) -> MultiHelmholtzSystem:
    """
    根据参数创建优化的亥姆霍兹线圈系统
    
    Args:
        params: 优化参数 [radius1, current1, turns1, separation1, radius2, current2, turns2, separation2, ...]
        
    Returns:
        多对亥姆霍兹线圈系统
    """
    coil_pairs = []
    
    # 假设每4个参数定义一对线圈
    for i in range(0, len(params), 4):
        if i + 3 < len(params):
            radius = params[i]
            current = params[i + 1]
            turns = int(params[i + 2])
            separation = params[i + 3]
            
            pair = create_helmholtz_pair(radius, current, turns, separation)
            coil_pairs.append(pair)
    
    return MultiHelmholtzSystem(coil_pairs)

if __name__ == "__main__":
    # 测试代码
    # 创建一对标准亥姆霍兹线圈
    radius = 0.1  # 10cm半径
    current = 1.0  # 1A电流
    turns = 100   # 100匝
    separation = 0.1  # 10cm间距
    
    pair1 = create_helmholtz_pair(radius, current, turns, separation)
    system = MultiHelmholtzSystem([pair1])
    
    # 计算匀场性能
    uniformity_result = system.field_uniformity()
    print(f"匀场性能指标:")
    print(f"平均磁场: {uniformity_result['mean_field']:.6f} T")
    print(f"磁场标准差: {uniformity_result['std_field']:.6f} T")
    print(f"匀场度: {uniformity_result['uniformity']:.4f}")
    print(f"方向一致性: {uniformity_result['direction_consistency']:.4f}")
    
    # 可视化磁场分布
    system.visualize_field()

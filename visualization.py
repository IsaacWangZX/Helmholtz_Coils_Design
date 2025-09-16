"""
亥姆霍兹线圈优化可视化模块
实现磁场分布、算法对比、优化结果等多种可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import pandas as pd
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HelmholtzVisualizer:
    """亥姆霍兹线圈可视化类"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def plot_field_distribution(self, system, region_size: float = 0.02, 
                               resolution: int = 20, title: str = "磁场分布"):
        """
        绘制磁场分布图
        
        Args:
            system: 亥姆霍兹线圈系统
            region_size: 区域大小 (m)
            resolution: 分辨率
            title: 图标题
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 创建网格
        x_range = np.linspace(-region_size/2, region_size/2, resolution)
        y_range = np.linspace(-region_size/2, region_size/2, resolution)
        z_range = np.linspace(-region_size/2, region_size/2, resolution)
        
        # 计算磁场分布
        field_data = self._calculate_field_data(system, x_range, y_range, z_range)
        
        # 1. XY平面磁场强度分布 (Z=0)
        ax1 = fig.add_subplot(221)
        self._plot_2d_field(ax1, field_data['xy'], x_range, y_range, 
                          "XY平面磁场强度分布 (Z=0)", "X (m)", "Y (m)")
        
        # 2. XZ平面磁场强度分布 (Y=0)
        ax2 = fig.add_subplot(222)
        self._plot_2d_field(ax2, field_data['xz'], x_range, z_range,
                          "XZ平面磁场强度分布 (Y=0)", "X (m)", "Z (m)")
        
        # 3. YZ平面磁场强度分布 (X=0)
        ax3 = fig.add_subplot(223)
        self._plot_2d_field(ax3, field_data['yz'], y_range, z_range,
                          "YZ平面磁场强度分布 (X=0)", "Y (m)", "Z (m)")
        
        # 4. 3D磁场强度分布
        ax4 = fig.add_subplot(224, projection='3d')
        self._plot_3d_field(ax4, field_data['xyz'], x_range, y_range, z_range,
                          "3D磁场强度分布")
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _calculate_field_data(self, system, x_range, y_range, z_range):
        """计算磁场数据"""
        field_data = {
            'xy': np.zeros((len(y_range), len(x_range))),
            'xz': np.zeros((len(z_range), len(x_range))),
            'yz': np.zeros((len(z_range), len(y_range))),
            'xyz': np.zeros((len(z_range), len(y_range), len(x_range)))
        }
        
        for i, z in enumerate(z_range):
            for j, y in enumerate(y_range):
                for k, x in enumerate(x_range):
                    Bx, By, Bz = system.magnetic_field(x, y, z)
                    field_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
                    
                    field_data['xyz'][i, j, k] = field_mag
                    
                    # XY平面 (Z=0)
                    if i == len(z_range) // 2:
                        field_data['xy'][j, k] = field_mag
                    
                    # XZ平面 (Y=0)
                    if j == len(y_range) // 2:
                        field_data['xz'][i, k] = field_mag
                    
                    # YZ平面 (X=0)
                    if k == len(x_range) // 2:
                        field_data['yz'][i, j] = field_mag
        
        return field_data
    
    def _plot_2d_field(self, ax, field_data, x_range, y_range, title, xlabel, ylabel):
        """绘制2D磁场分布"""
        im = ax.imshow(field_data, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], 
                      cmap='viridis', origin='lower', aspect='auto')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('磁场强度 (T)', fontsize=10)
        
        # 添加等高线
        X, Y = np.meshgrid(x_range, y_range)
        contours = ax.contour(X, Y, field_data, levels=8, colors='white', alpha=0.6, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8)
    
    def _plot_3d_field(self, ax, field_data, x_range, y_range, z_range, title):
        """绘制3D磁场分布"""
        # 降采样以提高性能
        step = max(1, len(x_range) // 8)
        x_sub = x_range[::step]
        y_sub = y_range[::step]
        z_sub = z_range[::step]
        
        X, Y, Z = np.meshgrid(x_sub, y_sub, z_sub)
        field_sub = field_data[::step, ::step, ::step]
        
        # 散点图显示磁场强度
        scatter = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
                          c=field_sub.flatten(), cmap='viridis', s=20, alpha=0.6)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label('磁场强度 (T)', fontsize=10)
    
    def plot_field_uniformity(self, system, region_size: float = 0.02, 
                            resolution: int = 20, title: str = "磁场均匀性分析"):
        """
        绘制磁场均匀性分析图
        
        Args:
            system: 亥姆霍兹线圈系统
            region_size: 区域大小 (m)
            resolution: 分辨率
            title: 图标题
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 计算匀场性能
        uniformity_result = system.field_uniformity(region_size, resolution)
        
        # 1. 磁场强度分布直方图
        ax1.hist(uniformity_result['field_values'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(uniformity_result['mean_field'], color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {uniformity_result["mean_field"]:.6f} T')
        ax1.set_title('磁场强度分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('磁场强度 (T)', fontsize=10)
        ax1.set_ylabel('频次', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 磁场强度空间分布
        x_range = np.linspace(-region_size/2, region_size/2, resolution)
        y_range = np.linspace(-region_size/2, region_size/2, resolution)
        z_mid = resolution // 2
        
        field_xy = np.zeros((len(y_range), len(x_range)))
        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                Bx, By, Bz = system.magnetic_field(x, y, 0)
                field_xy[i, j] = np.sqrt(Bx**2 + By**2 + Bz**2)
        
        im2 = ax2.imshow(field_xy, extent=[-region_size/2, region_size/2, -region_size/2, region_size/2],
                        cmap='viridis', origin='lower')
        ax2.set_title('XY平面磁场强度分布', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        plt.colorbar(im2, ax=ax2, label='磁场强度 (T)')
        
        # 3. 磁场方向一致性
        field_vectors = uniformity_result['field_vectors']
        field_directions = field_vectors / (uniformity_result['field_values'][:, np.newaxis] + 1e-10)
        
        # 计算方向角
        angles = np.arctan2(field_directions[:, 1], field_directions[:, 0])
        ax3.hist(angles, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('磁场方向分布', fontsize=12, fontweight='bold')
        ax3.set_xlabel('方向角 (弧度)', fontsize=10)
        ax3.set_ylabel('频次', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能指标总结
        ax4.axis('off')
        metrics_text = f"""
        磁场均匀性分析结果
        
        平均磁场强度: {uniformity_result['mean_field']:.6f} T
        磁场标准差: {uniformity_result['std_field']:.6f} T
        均匀性指标: {uniformity_result['uniformity']:.4f}
        方向一致性: {uniformity_result['direction_consistency']:.4f}
        
        匀场区域大小: {region_size*1000:.1f} mm
        计算分辨率: {resolution}×{resolution}×{resolution}
        """
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_algorithm_comparison(self, results: Dict, title: str = "算法性能对比"):
        """
        绘制算法性能对比图
        
        Args:
            results: 算法结果字典
            title: 图标题
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        algorithms = list(results.keys())
        
        # 1. 适应度对比
        fitnesses = [results[alg]['mean_fitness'] for alg in algorithms]
        std_fitnesses = [results[alg]['std_fitness'] for alg in algorithms]
        
        bars1 = ax1.bar(algorithms, fitnesses, yerr=std_fitnesses, capsize=5, 
                       color=self.colors[:len(algorithms)], alpha=0.7, edgecolor='black')
        ax1.set_title('平均适应度对比', fontsize=12, fontweight='bold')
        ax1.set_ylabel('适应度', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, fitness in zip(bars1, fitnesses):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(std_fitnesses)*0.1,
                    f'{fitness:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 执行时间对比
        times = [results[alg]['mean_time'] for alg in algorithms]
        std_times = [results[alg]['std_time'] for alg in algorithms]
        
        bars2 = ax2.bar(algorithms, times, yerr=std_times, capsize=5,
                       color=self.colors[:len(algorithms)], alpha=0.7, edgecolor='black')
        ax2.set_title('平均执行时间对比', fontsize=12, fontweight='bold')
        ax2.set_ylabel('时间 (s)', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, time in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(std_times)*0.1,
                    f'{time:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 成功率对比
        success_rates = [results[alg]['success_rate'] for alg in algorithms]
        
        bars3 = ax3.bar(algorithms, success_rates, color=self.colors[:len(algorithms)], 
                       alpha=0.7, edgecolor='black')
        ax3.set_title('成功率对比', fontsize=12, fontweight='bold')
        ax3.set_ylabel('成功率', fontsize=10)
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars3, success_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.2%}', ha='center', va='bottom', fontsize=9)
        
        # 4. 综合评分雷达图
        self._plot_radar_chart(ax4, results, algorithms)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _plot_radar_chart(self, ax, results, algorithms):
        """绘制雷达图"""
        # 计算综合评分
        metrics = ['fitness', 'time', 'success_rate']
        metric_labels = ['适应度', '时间效率', '成功率']
        
        # 归一化数据
        normalized_data = {}
        for alg in algorithms:
            normalized_data[alg] = []
            
            # 适应度 (越小越好)
            fitness_score = 1.0 / (1.0 + results[alg]['mean_fitness'])
            normalized_data[alg].append(fitness_score)
            
            # 时间效率 (越小越好)
            time_score = 1.0 / (1.0 + results[alg]['mean_time'])
            normalized_data[alg].append(time_score)
            
            # 成功率 (越大越好)
            success_score = results[alg]['success_rate']
            normalized_data[alg].append(success_score)
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels)
        ax.set_ylim(0, 1)
        ax.grid(True)
        
        for i, alg in enumerate(algorithms):
            values = normalized_data[alg] + normalized_data[alg][:1]  # 闭合数据
            ax.plot(angles, values, 'o-', linewidth=2, label=alg, color=self.colors[i])
            ax.fill(angles, values, alpha=0.25, color=self.colors[i])
        
        ax.set_title('综合性能雷达图', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    def plot_convergence_curves(self, algorithms: List, title: str = "收敛曲线对比"):
        """
        绘制收敛曲线对比图
        
        Args:
            algorithms: 算法列表
            title: 图标题
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 线性坐标
        for i, algorithm in enumerate(algorithms):
            if hasattr(algorithm, 'fitness_history') and algorithm.fitness_history:
                ax1.plot(algorithm.fitness_history, label=algorithm.__class__.__name__,
                        color=self.colors[i % len(self.colors)], linewidth=2, alpha=0.8)
        
        ax1.set_title('收敛曲线对比', fontsize=12, fontweight='bold')
        ax1.set_xlabel('迭代次数', fontsize=10)
        ax1.set_ylabel('适应度', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 对数坐标
        for i, algorithm in enumerate(algorithms):
            if hasattr(algorithm, 'fitness_history') and algorithm.fitness_history:
                ax2.semilogy(algorithm.fitness_history, label=algorithm.__class__.__name__,
                            color=self.colors[i % len(self.colors)], linewidth=2, alpha=0.8)
        
        ax2.set_title('收敛曲线对比 (对数坐标)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('迭代次数', fontsize=10)
        ax2.set_ylabel('适应度 (对数)', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_pareto_front(self, pareto_solutions: List, pareto_objectives: List, 
                         title: str = "帕累托前沿"):
        """
        绘制帕累托前沿
        
        Args:
            pareto_solutions: 帕累托解
            pareto_objectives: 帕累托目标值
            title: 图标题
        """
        if len(pareto_objectives[0]) < 2:
            print("需要至少2个目标函数才能绘制帕累托前沿")
            return None
        
        fig = plt.figure(figsize=(12, 8))
        
        if len(pareto_objectives[0]) == 2:
            # 2D帕累托前沿
            ax = fig.add_subplot(111)
            objectives_array = np.array(pareto_objectives)
            
            ax.scatter(objectives_array[:, 0], objectives_array[:, 1], 
                      c='red', s=50, alpha=0.7, edgecolors='black')
            
            # 连接帕累托前沿点
            sorted_indices = np.argsort(objectives_array[:, 0])
            sorted_objectives = objectives_array[sorted_indices]
            ax.plot(sorted_objectives[:, 0], sorted_objectives[:, 1], 
                   'r--', alpha=0.5, linewidth=1)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('目标函数1', fontsize=12)
            ax.set_ylabel('目标函数2', fontsize=12)
            ax.grid(True, alpha=0.3)
            
        elif len(pareto_objectives[0]) == 3:
            # 3D帕累托前沿
            ax = fig.add_subplot(111, projection='3d')
            objectives_array = np.array(pareto_objectives)
            
            ax.scatter(objectives_array[:, 0], objectives_array[:, 1], objectives_array[:, 2],
                      c='red', s=50, alpha=0.7)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('目标函数1', fontsize=12)
            ax.set_ylabel('目标函数2', fontsize=12)
            ax.set_zlabel('目标函数3', fontsize=12)
            
        else:
            # 多目标平行坐标图
            ax = fig.add_subplot(111)
            objectives_array = np.array(pareto_objectives)
            
            # 归一化数据
            normalized_data = (objectives_array - objectives_array.min(axis=0)) / \
                            (objectives_array.max(axis=0) - objectives_array.min(axis=0))
            
            # 绘制平行坐标图
            for i in range(len(normalized_data)):
                ax.plot(range(len(normalized_data[i])), normalized_data[i], 
                       alpha=0.3, linewidth=1)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('目标函数', fontsize=12)
            ax.set_ylabel('归一化值', fontsize=12)
            ax.set_xticks(range(len(normalized_data[0])))
            ax.set_xticklabels([f'目标{i+1}' for i in range(len(normalized_data[0]))])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_optimization_summary(self, results: Dict, title: str = "优化结果总结"):
        """
        绘制优化结果总结图
        
        Args:
            results: 优化结果
            title: 图标题
        """
        fig = plt.figure(figsize=(16, 10))
        
        # 创建子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 最佳适应度对比
        ax1 = fig.add_subplot(gs[0, 0])
        algorithms = list(results.keys())
        best_fitnesses = [results[alg]['best_fitness'] for alg in algorithms]
        
        bars1 = ax1.bar(algorithms, best_fitnesses, color=self.colors[:len(algorithms)], 
                       alpha=0.7, edgecolor='black')
        ax1.set_title('最佳适应度', fontsize=12, fontweight='bold')
        ax1.set_ylabel('适应度', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 平均执行时间
        ax2 = fig.add_subplot(gs[0, 1])
        mean_times = [results[alg]['mean_time'] for alg in algorithms]
        
        bars2 = ax2.bar(algorithms, mean_times, color=self.colors[:len(algorithms)], 
                       alpha=0.7, edgecolor='black')
        ax2.set_title('平均执行时间', fontsize=12, fontweight='bold')
        ax2.set_ylabel('时间 (s)', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 成功率
        ax3 = fig.add_subplot(gs[0, 2])
        success_rates = [results[alg]['success_rate'] for alg in algorithms]
        
        bars3 = ax3.bar(algorithms, success_rates, color=self.colors[:len(algorithms)], 
                       alpha=0.7, edgecolor='black')
        ax3.set_title('成功率', fontsize=12, fontweight='bold')
        ax3.set_ylabel('成功率', fontsize=10)
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 综合评分
        ax4 = fig.add_subplot(gs[1, :])
        scores = []
        for alg in algorithms:
            fitness_score = 1.0 / (1.0 + results[alg]['mean_fitness'])
            time_score = 1.0 / (1.0 + results[alg]['mean_time'])
            success_score = results[alg]['success_rate']
            total_score = 0.4 * fitness_score + 0.3 * time_score + 0.3 * success_score
            scores.append(total_score)
        
        bars4 = ax4.bar(algorithms, scores, color=self.colors[:len(algorithms)], 
                       alpha=0.7, edgecolor='black')
        ax4.set_title('综合评分 (适应度40% + 时间30% + 成功率30%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('综合评分', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars4, scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 5. 性能指标表格
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # 创建性能指标表格
        table_data = []
        for alg in algorithms:
            row = [
                alg,
                f"{results[alg]['mean_fitness']:.6f}",
                f"{results[alg]['std_fitness']:.6f}",
                f"{results[alg]['best_fitness']:.6f}",
                f"{results[alg]['mean_time']:.3f}",
                f"{results[alg]['std_time']:.3f}",
                f"{results[alg]['success_rate']:.2%}"
            ]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data,
                         colLabels=['算法', '平均适应度', '适应度标准差', '最佳适应度', 
                                  '平均时间(s)', '时间标准差(s)', '成功率'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(table_data) + 1):
            for j in range(7):
                if i == 0:  # 表头
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:  # 数据行
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
                    else:
                        table[(i, j)].set_facecolor('white')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.show()
        
        return fig

def create_comparison_report(results: Dict, save_path: str = "optimization_report.html"):
    """
    创建HTML格式的对比报告
    
    Args:
        results: 优化结果
        save_path: 保存路径
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>亥姆霍兹线圈优化报告</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; text-align: center; }
            h2 { color: #34495e; border-bottom: 2px solid #3498db; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
            th { background-color: #3498db; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .metric { background-color: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .best { background-color: #d5f4e6; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>亥姆霍兹线圈优化算法对比报告</h1>
    """
    
    # 添加算法对比表格
    html_content += "<h2>算法性能对比</h2><table>"
    html_content += "<tr><th>算法</th><th>平均适应度</th><th>最佳适应度</th><th>平均时间(s)</th><th>成功率</th></tr>"
    
    for alg, result in results.items():
        html_content += f"<tr><td>{alg}</td><td>{result['mean_fitness']:.6f}</td>"
        html_content += f"<td>{result['best_fitness']:.6f}</td><td>{result['mean_time']:.3f}</td>"
        html_content += f"<td>{result['success_rate']:.2%}</td></tr>"
    
    html_content += "</table>"
    
    # 找出最佳算法
    best_fitness_alg = min(results.keys(), key=lambda x: results[x]['best_fitness'])
    best_time_alg = min(results.keys(), key=lambda x: results[x]['mean_time'])
    best_success_alg = max(results.keys(), key=lambda x: results[x]['success_rate'])
    
    html_content += f"""
    <h2>最佳算法</h2>
    <div class="metric best">最佳适应度: {best_fitness_alg}</div>
    <div class="metric best">最快执行: {best_time_alg}</div>
    <div class="metric best">最高成功率: {best_success_alg}</div>
    """
    
    html_content += """
    <h2>建议</h2>
    <div class="metric">
    <p><strong>选择建议:</strong></p>
    <ul>
        <li>如果追求最优解质量，推荐使用 {best_fitness_alg}</li>
        <li>如果追求计算速度，推荐使用 {best_time_alg}</li>
        <li>如果追求稳定性，推荐使用 {best_success_alg}</li>
    </ul>
    </div>
    </body>
    </html>
    """.format(best_fitness_alg=best_fitness_alg, best_time_alg=best_time_alg, best_success_alg=best_success_alg)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已保存到: {save_path}")

if __name__ == "__main__":
    # 测试可视化功能
    print("亥姆霍兹线圈可视化模块测试")
    print("=" * 50)
    
    # 创建测试数据
    test_results = {
        'PSO': {'mean_fitness': 0.123, 'std_fitness': 0.045, 'best_fitness': 0.089, 
                'mean_time': 1.23, 'std_time': 0.12, 'success_rate': 0.95},
        'DE': {'mean_fitness': 0.156, 'std_fitness': 0.067, 'best_fitness': 0.098, 
               'mean_time': 0.98, 'std_time': 0.08, 'success_rate': 0.88},
        'GA': {'mean_fitness': 0.134, 'std_fitness': 0.052, 'best_fitness': 0.092, 
               'mean_time': 1.45, 'std_time': 0.15, 'success_rate': 0.92}
    }
    
    # 创建可视化器
    visualizer = HelmholtzVisualizer()
    
    # 测试算法对比图
    visualizer.plot_algorithm_comparison(test_results)
    
    # 测试优化总结图
    visualizer.plot_optimization_summary(test_results)
    
    # 创建HTML报告
    create_comparison_report(test_results)
    
    print("可视化测试完成！")

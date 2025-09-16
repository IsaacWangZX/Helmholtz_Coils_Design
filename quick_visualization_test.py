"""
快速可视化功能测试
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

def test_matplotlib():
    """测试matplotlib基本功能"""
    print("测试matplotlib基本功能...")
    
    try:
        # 创建简单图形
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
        ax.set_title('测试图形')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
        
        # 保存图形
        plt.savefig('test_matplotlib.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print("✓ matplotlib基本功能正常")
        return True
        
    except Exception as e:
        print(f"✗ matplotlib测试失败: {e}")
        return False

def test_visualization_import():
    """测试可视化模块导入"""
    print("测试可视化模块导入...")
    
    try:
        from visualization import HelmholtzVisualizer, create_comparison_report
        print("✓ 可视化模块导入成功")
        return True
        
    except Exception as e:
        print(f"✗ 可视化模块导入失败: {e}")
        return False

def test_helmholtz_modules():
    """测试亥姆霍兹模块"""
    print("测试亥姆霍兹模块...")
    
    try:
        from helmholtz_objectives import create_helmholtz_bounds, FieldUniformityObjective
        from helmholtz_coil import create_optimized_helmholtz_system
        
        # 创建测试数据
        bounds = create_helmholtz_bounds(1)
        test_params = np.array([0.1, 1.0, 100, 0.1])
        
        # 测试目标函数
        objective = FieldUniformityObjective(bounds, target_field=0.1)
        fitness = objective(test_params)
        
        # 测试线圈系统
        system = create_optimized_helmholtz_system(test_params.tolist())
        
        print("✓ 亥姆霍兹模块功能正常")
        return True
        
    except Exception as e:
        print(f"✗ 亥姆霍兹模块测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("快速可视化功能测试")
    print("=" * 40)
    
    tests = [
        test_matplotlib,
        test_visualization_import,
        test_helmholtz_modules
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 40)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有基础测试通过！")
        print("可视化功能已准备就绪")
    else:
        print("✗ 部分测试失败")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    print(f"\n测试{'成功' if success else '失败'}")

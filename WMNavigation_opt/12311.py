# test_habitat_simple.py
import sys
import os

# 临时屏蔽有问题的导入
try:
    # 尝试直接导入核心模块
    from habitat_sim._ext.habitat_sim_bindings import Simulator, SimulatorConfiguration, AgentConfiguration, Configuration
    print("✅ 直接导入成功")
    
    # 简单配置测试
    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = ""
    sim_cfg.create_renderer = False
    sim_cfg.enable_physics = False
    
    agent_cfg = AgentConfiguration()
    cfg = Configuration(sim_cfg, [agent_cfg])
    
    sim = Simulator(cfg)
    print("✅ Simulator 创建成功!")
    
    sim.close()
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")
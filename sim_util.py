import math
import os
import random
import pygame
import git
import imageio
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt
import json

from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
class sim_setup():
    def __init__(self,cfg):
        self.cfg=cfg
        self.agent=None
        self.sim=None

    def display_sample(self,rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):#设计展示样例
        from habitat_sim.utils.common import d3_40_colors_rgb

        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

        arr = [rgb_img]
        titles = ["rgb"]
        if semantic_obs.size != 0:
            semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
            semantic_img.putpalette(d3_40_colors_rgb.flatten())
            semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
            semantic_img = semantic_img.convert("RGBA")
            arr.append(semantic_img)
            titles.append("semantic")

        if depth_obs.size != 0:
            depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
            arr.append(depth_img)
            titles.append("depth")

        plt.figure(figsize=(12, 8))
        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i + 1)
            ax.axis("off")
            ax.set_title(titles[i])
            plt.imshow(data)
        plt.show(block=False)

    def make_simple_cfg(self,settings,rgb_sensor=None,depth_sensor=None,semantic_sensor=None):#创建配置
    # 仿真器后端
        sim_cfg = habitat_sim.SimulatorConfiguration()
        # 根据是否需要语义信息选择场景文件
        if semantic_sensor and "semantic_scene" in settings:
            sim_cfg.scene_id = settings["semantic_scene"]  # 使用语义场景文件
        else:
            sim_cfg.scene_id = settings["scene"]  # 使用基础场景文件
        sim_cfg.gpu_device_id = 0
        # 智能体
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        rgb_sensor=settings["rgb_sensor"]
        depth_sensor=settings["depth_sensor"]
        semantic_sensor=settings["semantic_sensor"]

        agent_cfg.sensor_specifications = []
        if rgb_sensor:
        # 传感器
            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = "color_sensor"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
            rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            agent_cfg.sensor_specifications.append(rgb_sensor_spec)
        if depth_sensor:
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [settings["height"], settings["width"]]
            depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            agent_cfg.sensor_specifications.append(depth_sensor_spec)
        if semantic_sensor:
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
            semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            agent_cfg.sensor_specifications.append(semantic_sensor_spec)
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(settings['move_length'])
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(settings['rotate_angle'])
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(settings['rotate_angle'])
            ),
        }#amount中展示的是转动的角度以及前进的距离
            
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def agent_state_set(self,settings):#设置初始智能体状态
        self.agent = self.sim.initialize_agent(settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        position_1=np.array(settings["agent_state"]['position'], dtype=np.float32)
        agent_state.position = mn.Vector3(float(position_1[0]), float(position_1[1]), float(position_1[2]))
        self.agent.set_state(agent_state)
        self.sim.config.sim_cfg.allow_sliding = True

if __name__ == "__main__":
    path_to_config = "/home/sjs/test_file/config/config.json"
    with open(path_to_config) as f:
        cfg_first = json.load(f)  
    sim_cls=sim_setup(cfg_first)
    cfg=sim_cls.make_simple_cfg(cfg_first)
    sim_cls.sim = habitat_sim.Simulator(cfg)
    random_point = sim_cls.sim.pathfinder.get_random_navigable_point()
    print(f"随机可达点: {random_point}")
    cfg_first["agent_state"]['position']=np.array(random_point)
    sim_cls.agent_state_set(cfg_first)
    agent_state=sim_cls.agent.get_state()
    print(f'xini{agent_state}')
    action="move_forward"
    observations = sim_cls.sim.step(action)
    sim_cls.display_sample(observations["color_sensor"])
    print('okok')















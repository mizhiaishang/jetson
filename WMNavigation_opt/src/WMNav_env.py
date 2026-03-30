import gzip
import json
import logging
import math
import os
import random
import requests
import traceback
import habitat_sim
import time
import multiprocessing as mp
import tqdm

import pandas as pd
import numpy as np

from multiprocessing import Process, Value,Manager
from PIL import Image
from simWrapper import PolarAction, SimWrapper
from WMNav_agent import *
from custom_agent import *
from utils import *
from car_control import *
from display import *
from display1 import *

class Env:
    """
    Base class for creating an environment for embodied navigation tasks.
    This class defines the setup, logging, running, and evaluation of episodes.
    """

    task = 'Not defined'

    def __init__(self, cfg: dict):
        """
        Initializes the environment with the provided configuration.

        Args:
            cfg (dict): Configuration dictionary containing environment, simulation, and agent settings.
        """
        self.cfg = cfg['env_cfg']
        self.sim_cfg = cfg['sim_cfg']
        if self.cfg['name'] == 'default':
            self.cfg['name'] = f'default_{random.randint(0, 1000)}'
        self._initialize_logging(cfg)
        self._initialize_agent(cfg)
        self.outer_run_name = self.task + '_' + self.cfg['name']
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
       # self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper = None  # 修改self.simWrapper: SimWrapper = None
        self.num_episodes = 0
        print('初始化基础环境')
        self._initialize_experiment()

    def _initialize_agent(self, cfg: dict):
        """Initializes the agent for the environment."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        agent_cls = globals()[cfg['agent_cls']]
        self.agent: Agent = agent_cls(cfg['agent_cfg'])
        self.agent_cls = cfg['agent_cls']

    def _initialize_logging(self, cfg: dict):
        """
        Initializes logging for the environment.

        Args:
            cfg (dict): Configuration dictionary containing logging settings.
        """
        self.log_file = os.path.join(os.environ.get("LOG_DIR"), f'{cfg["task"]}_{self.cfg["name"]}/{self.cfg["instance"]}_of_{self.cfg["instances"]}.txt')
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if self.cfg['parallel']:
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )

    def _initialize_experiment(self):
        """
        Abstract method for setting up the environment and initializing all required variables.
        Should be implemented in derived classes.
        """
        raise NotImplementedError

    def run_experiment(self):
        """
        Runs the experiment by iterating over episodes.
        """
        instance_size = math.ceil(self.num_episodes / self.cfg['instances'])  # 1000
        start_ndx = self.cfg['instance'] * instance_size
        end_ndx = self.num_episodes

        for episode_ndx in range(start_ndx, min(start_ndx + self.cfg['num_episodes'], end_ndx)):

            self.wandb_log_data = {
                'episode_ndx': episode_ndx,  # 0
                'instance': self.inner_run_name,  # 0_of_1
                'total_episodes': self.cfg['instances'] * self.cfg['num_episodes'],  # 1
                'task': self.task,  # ObjectNav
                'task_data': {},
                'spl': 0,
                'goal_reached': False
            }

            try:
                self._run_episode(episode_ndx)
            except Exception as e:
                log_exception(e)
                self.simWrapper.reset()

    # def _timelog_create(self):
    #     time_pass1=self.time['ini_time']-self.time['start_time']
    #     time_pass2=self.time['ini_finish_time']-self.time['ini_time']
    #     time_pass3=self.time['explore_time']-self.time['ini_finish_time']
    #     time_pass4=self.time['score_time']-self.time['explore_time']
    #     time_pass5=self.time['pic_slc_time']-self.time['score_time']
    #     time_pass6=self.time['plan_time']-self.time['pic_slc_time']
    #     time_pass7=self.time['dic_slc_time']-self.time['plan_time']
    #     time_pass8=self.time['act_time']-self.time['dic_slc_time']
    #     time_pass9=self.time['act_finish_time']-self.time['act_time']
    #     log1 = (
    #             f"初始化阶段耗时: {time_pass1:.3f} 秒（从程序启动到初始化开始）\n"
    #             f"初始化完成耗时: {time_pass2:.3f} 秒（从初始化开始到初始化结束）\n"
    #             f"探索阶段耗时: {time_pass3:.3f} 秒（从初始化结束到探索阶段结束）\n"
    #             f"评分阶段耗时: {time_pass4:.3f} 秒（从探索结束到评分结束）\n"
    #             f"图片选择阶段耗时: {time_pass5:.3f} 秒（从评分结束到图片选择结束）\n"
    #             f"规划阶段耗时: {time_pass6:.3f} 秒（从图片选择结束到规划结束）\n"
    #             f"字典选择阶段耗时: {time_pass7:.3f} 秒（从规划结束到字典选择结束）\n"
    #             f"执行阶段耗时: {time_pass8:.3f} 秒（从字典选择结束到执行开始）\n"
    #             f"执行完成耗时: {time_pass9:.3f} 秒（从执行开始到执行完成）\n"
    #         )
    #     with open("/home/wheeltec/WMNavigation/tiaoshi/log.txt", "w", encoding="utf-8") as f:
    #         f.write(log1)

    def _run_episode(self, episode_ndx: int):
        """
        Runs a single episode.p

        Args:
            episode_ndx (int): The index of the episode to run.
        """
        obs = self._initialize_episode(episode_ndx)  # color_sensor(1080, 1920, 4) depth_sensor(1080, 1920) agent_state[position rotation sensor_states]
        print('初始化日志系统')
        print(f'\n===================开始运行======================\n')
        for _ in range(self.cfg['max_steps']):
            try:
                print('初始化实验环境')
                agent_action=self._step_env(obs)               
                if agent_action is None:
                    break
                agent_action.r=agent_action.r
                agent_action.theta=-agent_action.theta
                obs = self.simWrapper.step(agent_action)  # 执行操作，更新agent的状态和观察
            except Exception as e:
                log_exception(e)

            finally:
                self.step += 1
        self._post_episode()

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode. This method should be implemented in derived classes.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        self.step = 0
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0
        self.prev_agent_position = None

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment. This method should be implemented in derived classes.

        Args:
            obs (dict): The current observation. Contains agent state and sensor observations.

        Returns:
            PolarAction: The next action to be taken by the agent.
        """
        logging.info(f'Step {self.step}')
        agent_state: habitat_sim.AgentState = AgentState(obs['agent_state'])
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position

        return None

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        self.df.to_pickle(os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'))
        self.simWrapper.reset()
        self.agent.reset()
        if self.cfg['parallel']:
            try:
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')
        if self.cfg['log_freq'] == 1:
            create_gif(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'), self.agent.cfg['sensor_cfg']['img_height'], self.agent.cfg['sensor_cfg']['img_width'], agent_cls=self.agent_cls
            )
            create_gif_voxel(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        """
        Appends the step metadata to the dataframe, and saves the images and general metadata to disk.

        Args:
            images (dict): Images generated during the step.
            step_metadata (dict): Metadata for the current step.
            logging_data (dict): General logging data.
        """
        self.df = pd.concat([self.df, pd.DataFrame([step_metadata])], ignore_index=True)

        if self.step % self.cfg['log_freq'] == 0 or step_metadata['success'] == 0:
            path = os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}')
            if not step_metadata['success']:
                path += '_ERROR'
            os.makedirs(path, exist_ok=True)
            for name, im in images.items():
                if im is not None:
                    im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                    im.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w') as file:
                if step_metadata['success']:
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')

    def _calculate_metrics(self, agent_state: habitat_sim.AgentState, agent_action: PolarAction, max_steps: int):
        """
        Calculates the navigation metrics at a given step.

        Args:
            agent_state: The state of the agent.
            agent_action: The action taken by the agent.
            geodesic_path: The shortest path to the goal.
            max_steps (int): Maximum steps allowed for the episode.

        Returns:
            dict: A dictionary containing calculated metrics.
        """
        metrics = {}
        #self.path_calculator.requested_start = agent_state.position
        #metrics['distance_to_goal'] = self.simWrapper.get_path(self.path_calculator)
        metrics['spl'] = 0
        metrics['goal_reached'] = False
        metrics['done'] = False
        metrics['finish_status'] = 'running'

        if agent_action is PolarAction.stop or self.step + 1 == max_steps:
            metrics['done'] = True

            if self.agent.success:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': True
                })
            else:
                if agent_action is PolarAction.stop:
                    metrics['finish_status'] = 'fp'
                else:
                    metrics['finish_status'] = 'max_steps'

        return metrics

class WMNavEnv(Env):

    task = 'ObjectNav'

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.all_episodes = {}
        print('获取初始化照片')
        car_1=CarController(port='/dev/ttyUSB0')
        try:
            rgb,dep=car_1.get_images()
            if car_1.connect():
                time.sleep(0.2)
                start_position,start_rotation=car_1.get_agent_state()
        finally:
            car_1.disconnect()
        self.all_episodes['start_position']=start_position
        self.all_episodes['start_rotation']=start_rotation
        print('获取导航目标')
        input_task_obj='I want to find a fire extinguisher.'#input('please input what do you want to search')
        objection=self.agent._taskselect_module(input_task_obj,rgb)

        self.all_episodes['object_category']=objection['object']        
        self.num_episodes = 1

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the BASE task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes
        view_positions = []
        print('初始化智能体状态')
        self.current_episode = {
            'object': episode['object_category'],
            'view_positions': view_positions
        }
        self.init_pos = np.array(episode['start_position'])
        self.simWrapper = SimWrapper(self.sim_cfg)
        car = CarController(port='/dev/ttyUSB0')
        try:
            if car.connect():
                car.initialize_car()
                print('初始化成功')
        finally:
            car.disconnect()
        print('获取原始智能体状态')
        obs = self.simWrapper.step(PolarAction.null)
        obs_s=obs['agent_state']['position']
        print(f'智能体初始位置为{obs_s}')
        self.previous_subtask = '{}'  # Initialize the last subtask with an empty dictionary
        return obs

    def _core_code(self, obs1, episode_images):
        for _ in range(6):
            self.agent.navigability(obs1[_], _+1,self.count_1)
            self.count_1+=1
        img_dir = '/home/wheeltec/WMNavigation_opt/imgs'
        files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        # 读取并resize所有图片
        imgss = []
        for f in files:
            img = Image.open(os.path.join(img_dir, f))
            img = img.resize((1000, 1000))   # 统一大小
            imgss.append(img)
        # 处理 voxel_map (numpy -> PIL.Image)
        voxel = cv2.resize(self.agent.voxel_map, (1000, 1000), interpolation=cv2.INTER_NEAREST)
        voxel_img = Image.fromarray(voxel.astype("uint8"))  # 转为Image
        imgss.append(voxel_img)
        print('finish')
        obs=obs1[5]
        nav_map = self.agent.generate_voxel(obs['agent_state'])
        panoramic_image, explorable_value, reason = self.agent.make_curiosity_value(episode_images, self.current_episode['object'])
        img3 = Image.fromarray(panoramic_image, mode='RGB')
        img3.save('/home/wheeltec/WMNavigation_opt/tiaoshi/panoramic_image.png')
        img3 = Image.fromarray(voxel_img, mode='RGB')
        img3.save('/home/wheeltec/WMNavigation_opt/tiaoshi/voxel_img.png')
        goal_rotate, goal_reason = self.agent.update_curiosity_value(explorable_value, reason)

        direction_image = episode_images[goal_rotate]
        goal_flag, subtask = self.agent.make_plan(direction_image, self.previous_subtask, goal_reason, self.current_episode['object'])

        self.previous_subtask = subtask # update last subtask


        
        return goal_rotate,explorable_value,reason,goal_flag,subtask,panoramic_image,nav_map




    def _step_env(self , obs: dict):
        """
        Takes a step in the environment for the BASEV1 task.

        Args:
            obs (dict): The current observation.

        Returns:
            list: The next action to be taken by the agent.
        """
        #obs color sensor是一个480*640*4的列表，复制，取前3通道(去掉alpha通道)
        episode_images = [(obs['color_sensor'].copy())[:, :, :3]]
        #color_origin代表最初的小车面朝方向的图片480*640*3
        color_origin = episode_images[0]
        #动作 顺时针旋转60°
        loop_action_clockwise = PolarAction(0, -0.167 *2* np.pi)
        #  确定目标方向
        print('感知模块启动')
        obs1,episode_images= self.simWrapper.step1(loop_action_clockwise,episode_images)
        self.count_1=0
        
        print('体素地图信息获取过程中')
        goal_rotate,explorable_value,reason,goal_flag,subtask,panoramic_image,nav_map=self._core_code(obs1, episode_images)

        print('开始动作规划')

        # 整个模型前向运行一次，返回动作和结果        #  转向目标方向
        print('转向选择方向')
        loop_action_clockwise_whole=PolarAction(0, -0.167 *2*goal_rotate* np.pi)
        obs = self.simWrapper.step(loop_action_clockwise_whole)
                
        super()._step_env(obs)
        obs['goal'] = self.current_episode['object']  # 目标的类别，最短距离，目标位置，所有可到点
        obs['subtask'] = subtask  # 子目标
        obs['goal_flag'] = goal_flag  # 是否发现目标
        agent_state: habitat_sim.AgentState = AgentState(obs['agent_state'])
        self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position


        agent_action,metadata=self.agent.step(obs)
        step_metadata = metadata['step_metadata']
        metadata['logging_data']['EVALUATOR_RESPONSE'] = str({'goal_rotate':goal_rotate*30, 'explorable_value': explorable_value, 'reason': reason})
        metadata['logging_data']['PLANNING_RESPONSE'] = str({'goal_flag': goal_flag, 'subtask': subtask})
        logging_data = metadata['logging_data']

        images = metadata['images']

        if metadata['step'] is not None:
            step_text = f"step {metadata['step']}"
            color_origin = np.ascontiguousarray(color_origin)
            color_origin = cv2.putText(color_origin, step_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if obs['goal'] is not None:
            scale_factor = color_origin.shape[0] / 1080
            padding = 20
            text_size = 2.5 * scale_factor
            text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(f"goal:{obs['goal']}", cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
            text_position = (color_origin.shape[1] - text_width - padding, padding + text_height)
            cv2.putText(color_origin, f"goal:{obs['goal']}", text_position, cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 0, 0), text_thickness,
                        cv2.LINE_AA)
            
        planner_images = {'panoramic': panoramic_image,
                          'color_origin': color_origin,
                          'nav_map': nav_map}
        images.update(planner_images)  # 保存规划过程的图片

        metrics = self._calculate_metrics(agent_state, agent_action, self.cfg['max_steps'])
        step_metadata.update(metrics)

        self._log(images, step_metadata, logging_data)

        if metrics['done']:
            agent_action = None
        return agent_action

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        # self.df.to_pickle(os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl'))
        # self.simWrapper.reset()
        # self.agent.reset()
        # if self.cfg['parallel']:
        #     try:
        #         self.wandb_log_data['spend'] = self.agent.get_spend()
        #         self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
        #         response = requests.post(f'http://localhost:{self.cfg["port"]}/log', json=self.wandb_log_data)
        #         if response.status_code != 200:
        #             logging.error(f"Failed to send metrics: {response.text}")
        #     except Exception as e:
        #         tb = traceback.extract_tb(e.__traceback__)
        #         for frame in tb:
        #             logging.error(f"Frame {frame.filename} line {frame.lineno}")
        #         logging.error(e)

        logging.info(f"成功完成任务")
        logging.info('\n===================运行结束===================\n')
        if self.cfg['log_freq'] == 1:
            create_gif(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'), self.agent.cfg['sensor_cfg']['img_height'], self.agent.cfg['sensor_cfg']['img_width'], agent_cls=self.agent_cls
            )
            create_gif_nav(
                    os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                    1800, 1800
            )
            create_gif_cvalue(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )

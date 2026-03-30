import logging
import numpy as np
import magnum as mn
import time
from car_control import *

# from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
from camare_test import *
from utils import *


class PolarAction:
    
    default = None #Default action if the VLM response is not parsed
    stop = None #Stop action
    null = None #Null action to just get current observation

    def __init__(self, r, theta, type=None):
        self.theta = theta
        self.r = r
        self.type = type

PolarAction.stop = PolarAction(0, 0, 'stop')
PolarAction.null = PolarAction(0, 0, 'null')
    

class SimWrapper:
    """
    A wrapper for Habitat-Sim that initializes agents, sensors, and manages movement and sensor observations.
    """
    def __init__(self, cfg):
        """
        Initialize the simulator with the specified configurations.

        :param cfg: Dictionary with configurations for the simulator, agents, and sensors.
        """
        self.use_goal_image_agent = cfg['use_goal_image_agent']
        self.allow_slide = cfg['allow_slide']
        self.sensor_pitch = cfg['sensor_cfg']['pitch']
        self.fov = cfg['sensor_cfg']['fov']
        self.sensor_height = cfg['sensor_cfg']['height']
        self.resolution = (
            cfg['sensor_cfg']['img_height'],
            cfg['sensor_cfg']['img_width']
        )
    
    def step1(self, action: PolarAction,episode_images:list):
        """
        Move the agent based on the specified action and magnitude.

        :param action: The action to perform.
        """
        observation=[]
        car = CarController(port='/dev/ttyUSB0')
        try:
            if car.connect():
                for i in range(6):
                    rotate=car.rotate(-action.theta*180/3.1415)
                    time.sleep(0.2)
                    state_position,state_rotation=car.get_agent_state()
                    print(state_position,state_rotation)
                    rgb_image,dep_image=car.get_images()
                    observation1=obs_solve(state_position,state_rotation,rgb_image,dep_image)
                    observation.append(observation1)
                    if i < 5:
                        episode_images.append((observation1['color_sensor'].copy())[:, :, :3])
        finally:
            car.disconnect()
        return observation,episode_images


    def step(self, action: PolarAction):
        """
        Move the agent based on the specified action and magnitude.

        :param action: The action to perform.
        """
        car = CarController(port='/dev/ttyUSB0')
        try:
            if car.connect():
                time.sleep(0.2)
                rotate=car.rotate(-action.theta*180/3.1415)
                time.sleep(0.2)
                move=car.move(action.r)
                time.sleep(0.2)
                state_position,state_rotation=car.get_agent_state()
                rgb_image,dep_image=car.get_images()
        finally:
            car.disconnect()
        observation=obs_solve(state_position,state_rotation,rgb_image,dep_image)
        return observation

    def _move_forward(self, agent_state, curr_position, curr_rotation, magnitude):
        """
        Move the agent_state forward by a specified magnitude.

        :param agent_state: The state of the agent to be updated.
        :param curr_position: Current position of the agent.
        :param curr_rotation: Current rotation of the agent.
        :param magnitude: Distance to move forward.
        """
        local_point = np.array([0, 0, -magnitude])
        global_point = local_to_global(curr_position, curr_rotation, local_point)
        delta = (global_point - curr_position) / 10
        new_position = np.copy(curr_position)

        for _ in range(10):
            if self.allow_slide:
                new_position = self.sim.pathfinder.try_step(new_position, new_position + delta)
            else:
                new_position = self.sim.pathfinder.try_step_no_sliding(new_position, new_position + delta)

        agent_state.position = new_position

    def _rotate_yaw(self, agent_state, curr_rotation, magnitude):
        """
        Rotate the agent_state by a specified angle.

        :param agent_state: The state of the agent to be updated.
        :param curr_rotation: Current rotation of the agent.
        :param magnitude: Angle in radians to rotate counterclockwise.
        """
        theta, axis = quat_to_angle_axis(curr_rotation)
        if axis[1] < 0:  # Ensure consistent rotation direction
            theta = 2 * np.pi - theta
        new_theta = theta + magnitude
        agent_state.rotation = quat_from_angle_axis(new_theta, np.array([0, 1, 0]))

    def get_goal_image(self, goal_position, goal_rotation):
        """
        Capture an image from the goal agent's perspective.

        :param goal_position: Position of the goal agent.
        :param goal_rotation: Rotation of the goal agent.
        :return: The captured image from the goal sensor.
        """
        assert self.use_goal_image_agent, "Goal image agent is not enabled."

        goal_agent = self.sim.get_agent(1)
        new_agent_state = habitat_sim.AgentState()
        new_agent_state.position = goal_position
        new_agent_state.rotation = goal_rotation
        goal_agent.set_state(new_agent_state)

        observations = self.sim.get_sensor_observations(1)
        return observations['goal_sensor']

    def set_state(self, pos, quat):
        """
        Set the agent's state to the specified position and orientation.

        :param pos: The position to set.
        :param quat: The quaternion orientation to set.
        """
        agent = self.sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = pos
        agent_state.rotation = quat
        agent.set_state(agent_state)

    def get_path(self, path):
        """
        Find a path to the specified target position.

        :param path: Target position for pathfinding.
        :return: The path found by the pathfinder.
        """
        if self.sim.pathfinder.find_path(path):
            return path.geodesic_distance
        else:
            logging.info('NO PATH FOUND')
            return 1000

    def reset(self):
        """
        Close the simulator to clean up memory.
        """
        try:
            self.sim.close()
        except:
            pass
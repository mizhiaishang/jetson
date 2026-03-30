import asyncio
import time
import logging
import random
import threading
import sys
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from car_control import *
sys.path.append("/home/wheeltec/test_file/pyorbbecsdk")
sys.path.append("/home/wheeltec/test_file/datanavi")
from testnavi import *
from util import *
from camera import OrbbecCamera
from dongtai import *


class AgentState(Enum):
    NORMAL = "normal"
    EMERGENCY = "emergency"
    STOPPED = "stopped"

class Camera():
    def __init__(self,pipeline,config,hole_filling_filter,temporal_filter):
        self.pipeline=pipeline
        self.config=config
        self.hole_filling_filter=hole_filling_filter
        self.temporal_filter=temporal_filter


@dataclass
class Command:
    type: str
    level: int = 0
    timestamp: float = 0

class AsyncAgentSystem:
    def __init__(self,action):
        self.state = AgentState.NORMAL
        self.command_queue = asyncio.Queue()
        self.action_complete = False
        self.hov=91
        self.vov=65
        self.resolution=(720,1280)
        self.init_camera()
        self.init_car()
        self.action=[]
        self.action.append(action)
        self.mubiaodian_pos=[0,0,-8]
        self.if_action=False
        self.stop=False
        self.wait=False
    def init_car(self):
        self.car = CarController(port='/dev/ttyUSB0')
        if self.car.connect():
            print('car init completed')
            self.car.initialize_car()
    
    def init_camera(self):
        pipeline,config,hole_filling_filter,temporal_filter=set_config()
        self.camera=Camera(pipeline,config,hole_filling_filter,temporal_filter)
        print("start pipeline")
        self.camera.pipeline.start(config) 
        self.camera_model, self.camera_device = init_model(TRAINED_WEIGHTS, DEVICE)
        time.sleep(2)

    def get_image(self):
        pipeline=self.camera.pipeline
        hole_filling_filter=self.camera.hole_filling_filter
        temporal_filter=self.camera.temporal_filter
        rgb_1,dep_1=image_type(pipeline,hole_filling_filter,temporal_filter)
        return rgb_1,dep_1

    def get_init_information(self):
        pos, rot = car.get_agent_state()
    
    def action_dec(self,start):
        while True:
            if time.time()-start>0.1:
                if self.stop:
                    continue
                if len(self.action) != 0:
                    self.action_complete=False
                    self.action_done()
                    if self.stop:
                        continue
                elif len(self.action) == 0 and self.action_complete:
                    print('jieshu')
                    break

                start=time.time()


    def action_done(self):
        print('start')
        action_1=self.action.copy()
        action=action_1[0]
        self.action.pop(0)
        angle=action[1]/3.14*180
        self.car.rotate(-angle)
        self.car.move(action[0])
        self.action_complete=True
        print('okokokokokokokokokokokokokokoko')


    def monitor_agent_work(self):
        """监控智能体工作循环"""
        rgb_image,dep_image=self.get_image()
        pos, rot = self.car.get_agent_state()
        obs=obs_solve(pos,rot,rgb_image,dep_image)
        pos=obs['agent_state']['position']
        rot=obs['agent_state']['rotation']
        obt,zhangaiweizhi,zhangaidaxiao=get_obt_position(rgb_image,dep_image,pos,rot)
        print(obt)
        if obt:
            self.car.stop()
            self.action=[]
            self.stop = True
            print('stop')
            strat_time=time.time()
            img,mask_ini=process_image(self.camera_model,self.car,self.camera_device,rgb_image,dep_image)
            a=time.time()-strat_time
            converted_matrix = mask_ini.copy()
            converted_matrix = (converted_matrix == 255)
            mask = converted_matrix.copy()
            
            action_2_1,action_2_2=action_make(zhangaiweizhi,zhangaidaxiao,pos,rot,
                                              dep_image,self.mubiaodian_pos,self.resolution,
                                              self.hov,self.vov,obs,mask)
            b=time.time()-strat_time
            self.action.append(action_2_1)
            self.action.append(action_2_2)
            self.stop=False
            self.wait=True
    def monitor(self,start):
        while True:
            if self.wait:
                time.sleep(5)
                self.wait=False
                continue
            # if time.time()-start>0.1:
            self.monitor_agent_work()
            start=time.time()
            if self.action_complete:
                if len(self.action)==0:
                    break
            
    async def run_system(self):
        """运行系统"""
        start=time.time()
        main_thread = threading.Thread(target=self.action_dec, args=(start,))
        monitor_thread = threading.Thread(target=self.monitor, args=(start,))
        monitor_thread.start()
        main_thread.start()

        # 等待 monitor 线程结束
        monitor_thread.join()
        main_thread.join()

# 运行系统
if __name__ == "__main__":
    action=(8,0)
    system = AsyncAgentSystem(action)
    asyncio.run(system.run_system())
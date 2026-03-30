import time
import multiprocessing as mp
from tqdm import tqdm
import subprocess

def display1(progress_dict):
    time.sleep(2)
    total_steps = 10
    bar = tqdm(total=total_steps, desc="动作规划进度", dynamic_ncols=True)
    count = 0
    while count < total_steps:
        time.sleep(1.5)
        bar.update(1)
        count += 1
    bar.close()
    time.sleep(2)
    agent_action=progress_dict['agent_action']
    theta=agent_action.theta*180/3.14
    print(f'向{theta}方向前进{agent_action.r}m')
    print('动作执行模块启动')
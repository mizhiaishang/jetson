import time
import multiprocessing as mp
from tqdm import tqdm
import subprocess
import psutil

def show_gif_and_close(gif_path, display_time=5):
    """
    在Ubuntu上显示GIF文件并在指定时间后关闭
    
    Args:
        gif_path: GIF文件路径
        display_time: 显示时间（秒）
    """
    try:
        # 使用eog打开GIF文件
        process = subprocess.Popen(['eog', '--fullscreen', gif_path])
        # 等待GIF播放完成或指定时间
        time.sleep(display_time)
        # 关闭eog进程
        process.terminate()
        process.wait()
    except Exception as e:
        print(f"错误: {e}")
def display(progress_dict):
    time.sleep(2)
    total_steps = 10
    bar = tqdm(total=total_steps, desc="提速地图信息获取进度", dynamic_ncols=True)
    count = 0
    while count < total_steps:
        time.sleep(3)
        bar.update(1)
        count += 1
    bar.close()
    # show_gif_and_close("/home/wheeltec/WMNavigation/imgs/asa.gif", 4) 
    # print('图像合并中')
    # show_gif_and_close("/home/wheeltec/WMNavigation/tiaoshi/panoramic_image.png", 4)
    print('图像评分模型运行中')
    time.sleep(10)
    explorable_value=progress_dict['explorable_value']
    reason=progress_dict['reason']
    print(f'图像的评分为{explorable_value}')
    print(f'原因为{reason}')
    time.sleep(10)
    print('图片选择模型运行中')
    goal_rotate=progress_dict['goal_rotate']
    goal_reason=progress_dict['goal_reason']
    print(f'选择的方向为{goal_rotate}')
    print(f'选择的原因为{goal_reason}')
    time.sleep(10)
    print('规划模型运行中')
    subtask=progress_dict['subtask']
    print(f'下一步的任务为{subtask}')
    time.sleep(5)
    print('转向方向进行中') 


import os
import time
import cv2
import numpy as np
from pyorbbecsdk import *
from examples.utils import frame_to_bgr_image
from threading import Thread, Event

class OrbbecCamera:
    def __init__(self, save_dir="./capture_results"):
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = os.path.expanduser(save_dir)
        self._stop_event = Event()
        self._color_thread = Thread(target=self._capture_color, daemon=True)
        self._depth_thread = Thread(target=self._capture_depth, daemon=True)
        self._pipeline = Pipeline()
        self._config = Config()
        self._color_frame = None
        self._depth_data = None
    

    def initialize(self):
        color_profile = self._pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
        depth_profile = self._pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
        self._config.enable_stream(color_profile)
        self._config.enable_stream(depth_profile)
        self._pipeline.start(self._config)
        self._color_thread.start()
        self._depth_thread.start()

    def _capture_color(self):
        while not self._stop_event.is_set():
            frames = self._pipeline.wait_for_frames(200)
            if frames and frames.get_color_frame():
                self._color_frame = frame_to_bgr_image(frames.get_color_frame())
                self.color_data = cv2.cvtColor(self._color_frame, cv2.COLOR_BGR2RGB)
    def _capture_depth(self):
        while not self._stop_event.is_set():
            frames = self._pipeline.wait_for_frames(200)
            if frames and frames.get_depth_frame():
                depth_frame = frames.get_depth_frame()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                
                # 使用 get_width() 和 get_height() 获取分辨率
                height, width = depth_frame.get_height(), depth_frame.get_width()
                depth_2d = depth_data.reshape((height, width))
                
                # 转换为 float32 并处理无效值
                self._depth_data = depth_2d.astype(np.float32)
                self._depth_data[depth_2d == 0] = np.nan  # 可选：0.0 或 np.nan

    def get_current_frames(self):
        """
        获取当前的RGB图像和深度图像
        
        返回:
            tuple: (rgb_image, depth_image) 
                - rgb_image: numpy数组 (H,W,3), RGB格式, uint8类型
                - depth_image: numpy数组 (H,W), float32类型 (无效点为np.nan)
            如果数据不可用，返回 (None, None)
        """
        # 使用实际存在的属性名 _color_frame 和 _depth_data
        if self._color_frame is not None and self._depth_data is not None:
            # 确保_color_frame是RGB格式
            if len(self._color_frame.shape) == 3:  # 确认是彩色图像
                # 检查是否是BGR格式（OpenCV默认）
                if self._color_frame[0,0,0] == self._color_frame[0,0,2]:  # 简单BGR检查
                    rgb_image = cv2.cvtColor(self._color_frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = self._color_frame
                
                return rgb_image, self._depth_data
        
        return None, None

    def shutdown(self):
        self._stop_event.set()
        self._color_thread.join()
        self._depth_thread.join()
        self._pipeline.stop()

import os
import cv2
import numpy as np
import time
from pyorbbecsdk import *
from examples.utils import frame_to_bgr_image
from threading import Thread, Event


class OrbbecCamera:
    def __init__(self, save_dir="./capture_results"):
        os.makedirs(save_dir, exist_ok=True)
        self._save_dir = os.path.expanduser(save_dir)
        self._stop_event = Event()

        # Pipeline 和 Config 不在 __init__ 创建
        self._pipeline = None
        self._config = None

        # 数据缓冲
        self._color_frame = None
        self._depth_data = None

        # 线程对象
        self._color_thread = None
        self._depth_thread = None

    def initialize(self):
        """初始化相机并启动采集线程"""
        # 每次初始化都新建 pipeline 和 config
        self._pipeline = Pipeline()
        self._config = Config()

        color_profile = self._pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        ).get_default_video_stream_profile()
        depth_profile = self._pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        ).get_default_video_stream_profile()

        self._config.enable_stream(color_profile)
        self._config.enable_stream(depth_profile)

        # 启动 pipeline
        self._pipeline.start(self._config)

        # 重置停止标志
        self._stop_event.clear()

        # 新建线程
        self._color_thread = Thread(target=self._capture_color, daemon=True)
        self._depth_thread = Thread(target=self._capture_depth, daemon=True)

        self._color_thread.start()
        self._depth_thread.start()

    def _capture_color(self):
        """采集彩色图像"""
        while not self._stop_event.is_set():
            frames = self._pipeline.wait_for_frames(200)
            if frames and frames.get_color_frame():
                self._color_frame = frame_to_bgr_image(frames.get_color_frame())
                self._color_frame = cv2.cvtColor(
                    self._color_frame, cv2.COLOR_BGR2RGB
                )

    def _capture_depth(self):
        """采集深度图像"""
        while not self._stop_event.is_set():
            frames = self._pipeline.wait_for_frames(200)
            if frames and frames.get_depth_frame():
                depth_frame = frames.get_depth_frame()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)

                height, width = depth_frame.get_height(), depth_frame.get_width()
                depth_2d = depth_data.reshape((height, width))

                self._depth_data = depth_2d.astype(np.float32)
                self._depth_data[depth_2d == 0] = np.nan  # 可选：屏蔽无效值

    def save_current_frame(self, prefix="frame"):
        """返回当前一帧彩色图像和深度图"""
        if self._color_frame is not None and self._depth_data is not None:
            return self._color_frame, self._depth_data
        return None, None

    def shutdown(self):
        """关闭相机并释放资源"""
        self._stop_event.set()

        if self._color_thread and self._color_thread.is_alive():
            self._color_thread.join()
        if self._depth_thread and self._depth_thread.is_alive():
            self._depth_thread.join()

        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            self._config = None

        self._color_thread = None
        self._depth_thread = None
import os
import cv2
import numpy as np
import time
from pyorbbecsdk import *
from examples.utils import frame_to_bgr_image
from threading import Thread, Event


class OrbbecCamera:
    def __init__(self):
        self._save_dir = "/home/wheeltec/WMNavigation/pyorbbecsdk"
        self._stop_event = Event()

        # Pipeline 和 Config 不在 __init__ 创建
        self._pipeline = None
        self._config = None

        # 数据缓冲
        self._color_frame = None
        self._depth_data = None

        # 视频录制相关
        self._rgb_writer = None
        self._depth_writer = None
        self._is_recording = False
        self._recording_start_time = None
        
        # 线程对象
        self._color_thread = None
        self._depth_thread = None
        self._recording_thread = None

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

    def _process_depth_for_video(self, depth_data):
        """将深度数据转换为适合视频录制的8位图像"""
        # 移除NaN值，设置为0
        depth_processed = np.nan_to_num(depth_data, nan=0)
        
        # 归一化到0-255范围
        if np.max(depth_processed) > 0:
            depth_normalized = cv2.normalize(depth_processed, None, 0, 255, cv2.NORM_MINMAX)
        else:
            depth_normalized = np.zeros_like(depth_processed)
        
        # 转换为8位
        depth_8bit = depth_normalized.astype(np.uint8)
        
        # 应用颜色映射以便可视化
        depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        
        return depth_colored

    def start_recording(self, filename_prefix="recording", fps=30):
        """开始录制视频"""
        if self._is_recording:
            print("已经在录制中")
            return False
        
        try:
            # 获取第一帧来确定视频尺寸
            max_wait_time = 5  # 最大等待时间（秒）
            start_time = time.time()
            
            while self._color_frame is None or self._depth_data is None:
                if time.time() - start_time > max_wait_time:
                    print("等待帧数据超时")
                    return False
                time.sleep(0.1)
            
            # 确定视频尺寸
            color_height, color_width = self._color_frame.shape[:2]
            depth_height, depth_width = self._depth_data.shape[:2]
            
            # 创建视频写入器
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # RGB视频写入器
            rgb_filename = os.path.join(self._save_dir, f"{filename_prefix}_rgb_{timestamp}.avi")
            fourcc_rgb = cv2.VideoWriter_fourcc(*'XVID')
            self._rgb_writer = cv2.VideoWriter(rgb_filename, fourcc_rgb, fps, (color_width, color_height))
            
            # 深度视频写入器（使用处理后的彩色深度图）
            depth_filename = os.path.join(self._save_dir, f"{filename_prefix}_depth_{timestamp}.avi")
            fourcc_depth = cv2.VideoWriter_fourcc(*'XVID')
            self._depth_writer = cv2.VideoWriter(depth_filename, fourcc_depth, fps, (depth_width, depth_height))
            
            if not self._rgb_writer.isOpened() or not self._depth_writer.isOpened():
                print("无法创建视频文件")
                self.stop_recording()
                return False
            
            self._is_recording = True
            self._recording_start_time = time.time()
            
            # 启动录制线程
            self._recording_thread = Thread(target=self._recording_loop, daemon=True)
            self._recording_thread.start()
            
            print(f"开始录制: RGB->{rgb_filename}, Depth->{depth_filename}")
            return True
            
        except Exception as e:
            print(f"开始录制失败: {e}")
            self.stop_recording()
            return False

    def _recording_loop(self):
        """录制循环"""
        frame_count = 0
        
        while self._is_recording and not self._stop_event.is_set():
            if self._color_frame is not None and self._depth_data is not None:
                try:
                    # 写入RGB帧
                    rgb_frame_bgr = cv2.cvtColor(self._color_frame, cv2.COLOR_RGB2BGR)
                    self._rgb_writer.write(rgb_frame_bgr)
                    
                    # 处理并写入深度帧
                    depth_visualization = self._process_depth_for_video(self._depth_data)
                    self._depth_writer.write(depth_visualization)
                    
                    frame_count += 1
                    
                    # 显示录制状态（可选）
                    if frame_count % 30 == 0:  # 每30帧显示一次
                        recording_time = time.time() - self._recording_start_time
                        print(f"录制中... 帧数: {frame_count}, 时长: {recording_time:.1f}秒")
                        
                except Exception as e:
                    print(f"录制帧时出错: {e}")
            
            time.sleep(0.01)  # 短暂休眠以避免过度占用CPU

    def stop_recording(self):
        """停止录制视频"""
        if not self._is_recording:
            return
        
        self._is_recording = False
        
        # 等待录制线程结束
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=2.0)
        
        # 释放视频写入器
        if self._rgb_writer:
            self._rgb_writer.release()
            self._rgb_writer = None
        
        if self._depth_writer:
            self._depth_writer.release()
            self._depth_writer = None
        
        recording_time = time.time() - self._recording_start_time if self._recording_start_time else 0
        print(f"录制已停止，总时长: {recording_time:.1f}秒")

    def save_current_frame(self, prefix="frame"):
        """保存当前一帧彩色图像和深度图"""
        if self._color_frame is not None and self._depth_data is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # 保存RGB图像
            rgb_filename = os.path.join(self._save_dir, f"{prefix}_rgb_{timestamp}.png")
            rgb_frame_bgr = cv2.cvtColor(self._color_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_filename, rgb_frame_bgr)
            
            # 保存深度图像（原始数据）
            depth_filename = os.path.join(self._save_dir, f"{prefix}_depth_{timestamp}.npy")
            np.save(depth_filename, self._depth_data)
            
            # 保存深度可视化图像
            depth_visual = self._process_depth_for_video(self._depth_data)
            depth_visual_filename = os.path.join(self._save_dir, f"{prefix}_depth_visual_{timestamp}.png")
            cv2.imwrite(depth_visual_filename, depth_visual)
            
            print(f"帧已保存: {rgb_filename}, {depth_filename}")
            return rgb_filename, depth_filename
        return None, None

    def is_recording(self):
        """返回录制状态"""
        return self._is_recording

    def shutdown(self):
        """关闭相机并释放资源"""
        # 先停止录制
        if self._is_recording:
            self.stop_recording()
        
        # 设置停止事件
        self._stop_event.set()

        # 等待线程结束
        if self._color_thread and self._color_thread.is_alive():
            self._color_thread.join(timeout=2.0)
        if self._depth_thread and self._depth_thread.is_alive():
            self._depth_thread.join(timeout=2.0)

        # 停止pipeline
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            self._config = None

        # 清理线程引用
        self._color_thread = None
        self._depth_thread = None
        self._recording_thread = None


# 使用示例
if __name__ == "__main__":
    camera = OrbbecCamera()
    
    try:
        camera.initialize()
        print("相机初始化完成，等待3秒后开始录制...")
        time.sleep(3)
        
        # 开始录制
        if camera.start_recording("test_recording", fps=30):
            print("录制开始，按Enter键停止录制...")

            input()  # 等待用户输入
            
        # 停止录制


        camera.stop_recording()
        
        # 保存一帧作为示例
        camera.save_current_frame("sample")
        
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        camera.shutdown()
        print("相机已关闭")
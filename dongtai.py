import numpy as np
import sys
sys.path.append('/home/nvidia/mf/test_file/pyorbbecsdk/examples')
import pyorbbecsdk as ob
import time
import cv2
from utils import frame_to_bgr_image
from depth import *

MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

def depth_data_get(depth_frame,temporal_filter):
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((height, width))

    depth_data = depth_data.astype(np.float32) * scale
    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
    depth_data = depth_data.astype(np.uint16)
    # Apply temporal filtering
    depth_data = temporal_filter.process(depth_data)
    return depth_data

# def check_supported_resolutions():
#     """检查设备支持的所有分辨率"""
#     context = ob.Context()
#     device_list = context.query_devices()
#     if device_list.get_count() == 0:
#         print("No device found!")
#         return []
        
#     device = device_list.get_device_by_index(0)
#     pipeline = ob.Pipeline(device)
    
#     print("=== Supported Depth Resolutions ===")
#     profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    
#     depth_profiles = []
#     count = profile_list.get_count()
#     print(f"Found {count} depth profiles")
    
#     for i in range(count):
#         try:
#             # 使用正确的方法名
#             profile = profile_list.get_stream_profile_by_index(i)
            
#             if profile.is_video_stream_profile():
#                 video_profile = profile.as_video_stream_profile()
#                 depth_profiles.append(video_profile)
#                 print(f"  {video_profile.get_width()}x{video_profile.get_height()} @ {video_profile.get_fps()}fps - Format: {video_profile.get_format()}")
                
#         except Exception as e:
#             print(f"Error getting profile {i}: {e}")
#             continue
    
#     return depth_profiles

# # 先运行这个来查看支持的分辨率
# depth_profiles = check_supported_resolutions()

def set_config():
    config = ob.Config()  # Initialize the config for the pipeline
    pipeline = ob.Pipeline()  # Create the pipeline object
    temporal_filter = TemporalFilter(alpha=0.5)
    hole_filling_filter = ob.HoleFillingFilter()
    hole_filling_filter.set_filling_mode(OBHoleFillingMode.NEAREST)
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_video_stream_profile(1280,720,ob.OBFormat.Y16, 30)
    config.enable_stream(depth_profile)
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    return pipeline,config,hole_filling_filter,temporal_filter

def image_type(pipeline,hole_filling_filter,temporal_filter):
    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        # Get depth and color frames from the captured frames
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame is None or color_frame is None:
            continue
        color_image=frame_to_bgr_image(color_frame)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        filled_frame = hole_filling_filter.process(depth_frame)
        filled_frame=filled_frame.as_depth_frame()
        depth_image=depth_data_get(filled_frame,temporal_filter)
        depth_image=depth_image/1000
        if rgb_image is not None and depth_image is not None:
            break
    return rgb_image,depth_image

def vedio_type(pipeline,hole_filling_filter,temporal_filter):
    rgb=[]
    dep=[]
    while True:
        try:
            # Wait for frames from the pipeline (with a timeout of 100 ms)
            frames = pipeline.wait_for_frames(500)
            if frames is None:
                continue

            # Get depth and color frames from the captured frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Skip iteration if depth or color frame is not available
            if depth_frame is None or color_frame is None:
                continue
            color_image=frame_to_bgr_image(color_frame)
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            filled_frame = hole_filling_filter.process(depth_frame)
            filled_frame=filled_frame.as_depth_frame()
            depth_image=depth_data_get(filled_frame,temporal_filter)

            rgb.append(rgb_image)
            dep.append(depth_image)
            if len(rgb)>=6:
                break
        except Exception as e:
            print(e)
    return rgb,dep
def main_camera():
    pipeline,config,hole_filling_filter,temporal_filter=set_config()
    print("start pipeline")
    pipeline.start(config)  # Start the pipeline with the confi
    # rgb,dep=vedio_type(pipeline,hole_filling_filter,temporal_filter)
    rgb_1,dep_1=image_type(pipeline,hole_filling_filter,temporal_filter)
    return rgb_1,dep_1
if __name__ == "__main__":
    main_camera()
    print('ok')
import cv2
import numpy as np
import os
from typing import Tuple

class DepthSpikeRemover:
    def __init__(self, 
                 rgb_video_path: str, 
                 depth_video_path: str, 
                 output_dir: str = "./depth_denoised",
                 depth_scale: float = 1000.0,
                 # 针对小毛刺的核心参数
                 median_kernel: int = 3,  # 中值滤波核（3x3适合小毛刺）
                 morph_kernel: int = 3):  # 形态学滤波核（去除孤立噪点）
        """
        深度图小毛刺去除器：只处理孤立噪点，保留深度层次
        :param median_kernel: 中值滤波核大小（必须为奇数，3最佳）
        :param morph_kernel: 形态学开运算核大小（3x3适合小毛刺）
        """
        # 视频读取器
        self.rgb_cap = cv2.VideoCapture(rgb_video_path)
        self.depth_cap = cv2.VideoCapture(depth_video_path)
        
        if not self.rgb_cap.isOpened() or not self.depth_cap.isOpened():
            raise IOError("无法打开视频文件，请检查路径")
        
        # 视频参数
        self.fps = self.rgb_cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = min(
            int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self.depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        
        # 输出配置
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.depth_output_path = os.path.join(output_dir, "depth_denoised.avi")
        self.depth_colored_output_path = os.path.join(output_dir, "depth_colored.mp4")
        
        # 编码器设置（确保16位深度图兼容）
        self.depth_writer = cv2.VideoWriter(
            self.depth_output_path,
            cv2.VideoWriter_fourcc(*'FFV1'),
            self.fps,
            (self.width, self.height),
            isColor=False
        )
        self.depth_colored_writer = cv2.VideoWriter(
            self.depth_colored_output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )
        
        # 核心参数（针对小毛刺优化）
        self.depth_scale = depth_scale
        self.median_kernel = median_kernel if median_kernel % 2 == 1 else 3  # 确保奇数核
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)  # 椭圆核更适合圆形小毛刺
        )

    def _preprocess_depth(self, depth_frame: np.ndarray) -> np.ndarray:
        """仅做格式转换，不轻易标记无效值（避免破坏层次）"""
        if len(depth_frame.shape) == 3:
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        # 保留原始16位数据（避免精度损失）
        return depth_frame.astype(np.uint16)

    def _remove_spikes(self, depth_16bit: np.ndarray) -> np.ndarray:
        """
        核心：去除小毛刺（孤立噪点）
        步骤：中值滤波（抑制椒盐噪声）→ 形态学开运算（去除孤立小区域）
        """
        # 1. 中值滤波：专门针对小毛刺（椒盐噪声），保留边缘
        depth_median = cv2.medianBlur(depth_16bit, ksize=self.median_kernel)
        
        # 2. 形态学开运算：去除比核小的孤立噪点（小毛刺）
        # 先转换为8位处理（形态学操作对8位更友好）
        depth_8bit = cv2.normalize(
            depth_median, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_morph = cv2.morphologyEx(
            depth_8bit, 
            cv2.MORPH_OPEN,  # 开运算=腐蚀+膨胀，先去除小噪点再恢复形状
            self.morph_kernel
        )
        
        # 恢复为16位深度图
        depth_denoised = cv2.normalize(
            depth_morph, None, 
            0, 65535,  # 16位最大值
            cv2.NORM_MINMAX, dtype=cv2.CV_16U
        )
        
        # 保留原始深度的动态范围（避免亮度偏移）
        min_val = np.min(depth_16bit)
        max_val = np.max(depth_16bit)
        if max_val > min_val:  # 防止除零
            depth_denoised = cv2.normalize(
                depth_denoised, None, min_val, max_val, cv2.NORM_MINMAX, dtype=cv2.CV_16U
            )
        
        return depth_denoised

    def process_frame(self, depth_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """处理单帧深度图：只去毛刺，不破坏层次"""
        # 1. 预处理（格式转换）
        depth_16bit = self._preprocess_depth(depth_frame)
        
        # 2. 去除小毛刺
        depth_denoised = self._remove_spikes(depth_16bit)
        
        # 3. 生成彩色可视化图（便于对比）
        depth_colored = cv2.applyColorMap(
            cv2.normalize(depth_denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        return depth_denoised, depth_colored

    def run(self) -> None:
        """处理整个视频"""
        print(f"开始去除小毛刺：共 {self.total_frames} 帧\n"
              f"中值滤波核：{self.median_kernel}x{self.median_kernel}\n"
              f"形态学核：{self.morph_kernel.shape[0]}x{self.morph_kernel.shape[1]}")
        
        for frame_idx in range(self.total_frames):
            # 只需要深度帧（RGB可选，此处用于对齐但不参与滤波）
            ret_rgb, _ = self.rgb_cap.read()
            ret_depth, depth_frame = self.depth_cap.read()
            
            if not ret_rgb or not ret_depth:
                break
            
            try:
                depth_denoised, depth_colored = self.process_frame(depth_frame)
            except Exception as e:
                print(f"帧 {frame_idx} 处理出错：{str(e)}，跳过")
                continue
            
            # 保存结果
            self.depth_writer.write(depth_denoised)
            self.depth_colored_writer.write(depth_colored)
            
            if frame_idx % 10 == 0:
                print(f"已处理 {frame_idx}/{self.total_frames} 帧")
        
        # 释放资源
        self.rgb_cap.release()
        self.depth_cap.release()
        self.depth_writer.release()
        self.depth_colored_writer.release()
        
        print(f"处理完成！结果保存至：\n"
              f"- 去毛刺深度视频：{self.depth_output_path}\n"
              f"- 彩色深度视频：{self.depth_colored_output_path}")


if __name__ == "__main__":
    # 配置路径
    RGB_VIDEO = "/home/zyc/ultralytics/video/rgb2.mp4"
    DEPTH_VIDEO = "/home/zyc/ultralytics/video/depth2.mp4"
    OUTPUT_DIR = "./depth_spike_removed"
    
    # 初始化处理器（参数针对小毛刺优化）
    processor = DepthSpikeRemover(
        rgb_video_path=RGB_VIDEO,
        depth_video_path=DEPTH_VIDEO,
        output_dir=OUTPUT_DIR,
        depth_scale=1000.0,
        median_kernel=5,  # 3x3中值滤波，刚好去除1-2像素的毛刺
        morph_kernel=5    # 3x3形态学核，去除孤立小区域
    )
    
    processor.run()
    
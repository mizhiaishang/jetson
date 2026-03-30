import sys
import time
import numpy as np
import cv2
import math
sys.path.append("/home/wheeltec/test_file/pyorbbecsdk")

from camera import OrbbecCamera
from PIL import Image

# 简单的深度值查询类
class SimpleDepthInspector:
    def __init__(self):
        self.current_depth_map = None
        self.local_point=None
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，点击显示深度值"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_depth_map is not None:
                depth_value = self.current_depth_map[y, x]
                x_1=local_points[y,x][0]
                y_1=local_points[y,x][1]
                print(f"坐标: [{x_1}, {y_1}], 深度值: {depth_value:.4f}mm")
    
    def inspect(self, depth_map, window_name="Depth Inspector"):
        """简单的深度值查询"""
        self.current_depth_map = depth_map
        
        # 创建彩色深度图用于显示
        display_img = draw_depth(depth_map)
        
        # 添加使用说明
        cv2.putText(display_img, "Click anywhere to see depth value", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_img, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示窗口并设置鼠标回调
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        cv2.imshow(window_name, display_img)
        
        print("深度查询模式：点击图像查看深度值，按'q'退出")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyWindow(window_name)

def get_images():
    time.sleep(2)
    rgb_image, dep_image = camera.save_current_frame()
    return rgb_image, dep_image

def resize_image_opencv(image, target_width=1280, target_height=720):
    original_height, original_width = image.shape[:2]
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

def rgb_guided_depth_inpainting(depth_img, rgb_img, radius=5, eps=0.01):
    mask = np.isnan(depth_img).astype(np.uint8) * 255
    depth_img_no_nan = np.nan_to_num(depth_img)
    
    rgb_normalized = rgb_img.astype(np.float32) / 255.0
    
    guided_filter = cv2.ximgproc.createGuidedFilter(
        guide=rgb_normalized,
        radius=radius,
        eps=eps
    )
    
    inpainted_depth = guided_filter.filter(
        src=depth_img_no_nan.astype(np.float32),
        dst=np.zeros_like(depth_img_no_nan))
    
    result = depth_img.copy()
    result[np.isnan(result)] = inpainted_depth[np.isnan(result)]
    
    return result

def draw_depth(depth_map, min_depth=None, max_depth=None):
    depth = depth_map.copy()
    if depth.dtype != np.float32 and depth.dtype != np.float64:
        depth = depth.astype(np.float32)
    depth[np.isnan(depth)] = 0
    
    valid_depths = depth[depth > 0]
    if len(valid_depths) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    
    if min_depth is None:
        min_depth = np.min(valid_depths)
    if max_depth is None:
        max_depth = np.max(valid_depths)
    
    depth_normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    depth_uint8 = np.uint8(depth_normalized * 255)
    
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_color[depth == 0] = 0
    
    return depth_color

def depth_to_height(depth_image, hfov, camera_position, camera_orientation, vfov):
    img_height, img_width = depth_image.shape
    focal_length_px = img_width / (2 * np.tan(np.radians(hfov / 2)))

    i_idx, j_idx = np.indices((img_height, img_width))
    x_prime = (j_idx - img_width / 2)
    y_prime = (i_idx - img_height / 2)

    h_focal_length_px, v_focal_length_px = calculate_focal_length_from_hfov_vfov(hfov, vfov, img_width, img_height)

    x_local = x_prime * depth_image / h_focal_length_px
    y_local = y_prime * depth_image / v_focal_length_px
    z_local = depth_image
    local_points = np.stack((x_local, -y_local, -z_local), axis=-1)
    return local_points

def calculate_focal_length_px(hfov_degrees, img_width):
    focal_length_px = img_width / (2 * np.tan(np.radians(hfov_degrees / 2)))
    return focal_length_px

def calculate_focal_length_from_hfov_vfov(hfov, vfov, img_width, img_height):
    h_focal_px = calculate_focal_length_px(hfov, img_width)
    v_focal_px = calculate_focal_length_px(vfov, img_height)
    return h_focal_px, v_focal_px

if __name__ == "__main__":
    camera = OrbbecCamera()
    camera.initialize()
    
    # 创建深度检查器
    depth_inspector = SimpleDepthInspector()

    # 处理第一组图像
    a, b = get_images()
    b = resize_image_opencv(b)
    f=draw_depth(b)
    img4 = Image.fromarray(a, mode='RGB')
    img4.save(f'/home/wheeltec/test_file/2.png')
    local_points=depth_to_height(b,91,1,1,65)
    depth_inspector.local_point=local_points
    B = rgb_guided_depth_inpainting(b, a)
    depth1=B[620,866]
    depth2=b[620,866]
    print(depth1,depth2)
    b=draw_depth(B)
    img4 = Image.fromarray(b, mode='RGB')
    img4.save(f'/home/wheeltec/test_file/1.png')
    target_point = (866, 620)  # 替换为你的目标点坐标
    cv2.circle(b, target_point, 5, (0, 0, 255), -1)
    cv2.imwrite('/home/wheeltec/test_file/datanavi/3.png', b)
    # 使用深度查询功能
    depth_inspector.inspect(b, "Processed Depth Map 1")
    
    # 处理第二组图像
    c, d = get_images()
    d = resize_image_opencv(d)
    D = rgb_guided_depth_inpainting(d, c)
    
    # 使用深度查询功能
    depth_inspector.inspect(D, "Processed Depth Map 2")
    
    camera.shutdown()
    cv2.destroyAllWindows()
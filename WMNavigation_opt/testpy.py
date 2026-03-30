import sys
import time
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

sys.path.append("/home/wheeltec/WMNavigation_opt/pyorbbecsdk")
sys.path.append("/home/wheeltec/WMNavigation_opt/src")

from camera import OrbbecCamera
from camare_test import *

def get_images(camera):
    time.sleep(2)
    rgb_image, dep_image = camera.save_current_frame()

    # 转换颜色格式 BGR -> RGB（YOLO 需要 RGB）
    if rgb_image.shape[-1] == 3:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    return rgb_image, dep_image

# 初始化摄像头和模型
camera = OrbbecCamera()
camera.initialize()

model = YOLO("yolo11m.pt")  # 模型加载一次即可

time_start = time.time()
time_last = time.time()
i = 0
xx=depth_to_height(B,94,1,1,68)

# 运行 30 秒内持续拍摄
try:
    while time.time() - time_start < 30:
        a, b = get_images(camera)
        b = resize_image_opencv(b)
        B = rgb_guided_depth_inpainting(b, a)

        results = model.predict(source=a, conf=0.2, show=False)
        r = results[0]
        r.save(f"/home/wheeltec/WMNavigation_opt/{i}.png")

        i += 1
        print(f'保存成功: 第{i}张图片')

        # 控制间隔，例如每隔 3 秒拍一张
        time.sleep(3)
        time_last = time.time()
except Exception as e:
    print(e)
finally:
    camera.shutdown()
    print("✅ 程序安全退出，释放相机资源")


print(f'获取完成，共{i}张图片')


import time
import sys
import cv2
import quaternion
import faiss
sys.path.append("/home/nvidia/mf/test_file/SU1")
sys.path.append("/home/nvidia/mf/test_file")
from airsim_explorer_1 import *
from videotest import *#SU1
from car_control import *
from dongtai import *
from Main_visual import *

class agent_state1():
    def __init__(self,pos,rot):
        self.position=pos
        self.rotation=rot

def image_information(r1,obs,car):
        start = time.time()
        detection_objects = []
        img_width, img_height = obs['color_sensor'].shape[1], obs['color_sensor'].shape[0]
        focal_length_x, focal_length_y = calculate_focal_length_from_hfov_vfov(91, 65, img_width, img_height)
        
        img = r1.plot()
        img1=obs['color_sensor']
        
        pos,rot= car.get_agent_state()
        print(pos,rot)
        rot=[rot[3],rot[0],rot[1],rot[2]]
        rot = np.quaternion(*rot)
        agent_state=agent_state1(pos,rot)

        for result in r1:
            boxes = result.boxes
            names = result.names
            
            for box in boxes:
                # 安全地提取数据
                class_id = int(box.cls[0].cpu().item()) if box.cls[0].is_cuda else int(box.cls[0].item())
                class_name = names[class_id]
                confidence = float(box.conf[0].cpu().item()) if box.conf[0].is_cuda else float(box.conf[0].item())
                
                # 安全地提取边界框坐标
                if box.xyxy[0].is_cuda:
                    bbox = box.xyxy[0].cpu().tolist()
                    x1, y1, x2, y2 = box.xyxy[0].cpu()
                else:
                    bbox = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = box.xyxy[0]
                
                # 计算中心点
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                ex = [x, y]
                
                # 确保坐标是整数类型
                x_int = int(round(x.item() if hasattr(x, 'item') else x))
                y_int = int(round(y.item() if hasattr(y, 'item') else y))
                
                # 安全地获取深度值
                try:
                    depth = obs['depth_sensor'][y_int, x_int]  # 注意：numpy数组是 [height, width] 顺序
                except IndexError:
                    print(f"坐标超出范围: ({y_int}, {x_int})，使用默认深度值")
                    depth = 1.0  # 默认深度值
                
                xy = image_to_local(x_int, y_int, depth, (img_height, img_width), focal_length_x, focal_length_y)
                xy = local_to_global(agent_state.position, agent_state.rotation, xy)
                
                detection = [
                    class_name,
                    bbox,
                    confidence,
                    xy.tolist() if hasattr(xy, 'tolist') else xy  # 确保xy是列表格式
                ]
                detection_objects.append(detection)
        
        inf = detection_objects
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"/home/nvidia/mf/langChain_Test/semantic_graphs/detection_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)

        # 保存图片
        cv2.imwrite(f"{folder_name}/result.jpg", img)
        cv2.imwrite(f"{folder_name}/result_orin.jpg", img1)

        # 保存JSON信息（包含位置信息）
        json_data = {
            "timestamp": timestamp,
            "detections": [
                {
                    "class_name": det[0],
                    "bbox": {"x1": det[1][0], "y1": det[1][1], "x2": det[1][2], "y2": det[1][3]},
                    "confidence": det[2],
                    "global_position": det[3]  # 添加全局位置信息
                } for det in detection_objects
            ]
        }

        with open(f"{folder_name}/info.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"保存完成: {folder_name}")
        return json_data,folder_name

if __name__ == "__main__":
    start=time.time()
    car = CarController(port='/dev/ttyUSB0')
    if car.connect():
        car.initialize_car()
        while True:
            rgb,dep=main_camera()
            rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            obs={}
            obs['color_sensor']=rgb
            obs['depth_sensor']=dep
            r1=process_image(rgb)
            json_data,folder_name=image_information(r1[0],obs,car)
            main_1(json_data,folder_name)



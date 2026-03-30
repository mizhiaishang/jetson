from dongtai import *
import time
import sys
import cv2
import gc
import quaternion
from airsim_explorer_1 import *
sys.path.append("/home/nvidia/mf/test_file/SU1")
from videotest import *
from car_control import *
# from Main_visual import *

class agent_state1():
    def __init__(self,pos,rot):
        self.position=pos
        self.rotation=rot

def get_avg_depth_in_box(depth_map, x1, y1, x2, y2, edge_ratio=0.1):
    """
    框内采样 5 个点深度并取平均，按比例采样
    depth_map: HxW numpy array
    x1,y1,x2,y2: 框坐标
    edge_ratio: 左右边距占比，0~0.5
    """
    h_map, w_map = depth_map.shape[:2]

    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return None

    # 5 个采样点（中心 + 四角按比例）
    sample_points = [
        (x1 + 0.5*w, y1 + 0.5*h),           # 中心
        (x1 + edge_ratio*w, y1 + edge_ratio*h),   # 左上
        (x1 + (1-edge_ratio)*w, y1 + edge_ratio*h), # 右上
        (x1 + edge_ratio*w, y1 + (1-edge_ratio)*h), # 左下
        (x1 + (1-edge_ratio)*w, y1 + (1-edge_ratio)*h) # 右下
    ]

    depths = []

    for x, y in sample_points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w_map and 0 <= yi < h_map:
            d = depth_map[yi, xi]
            if np.isfinite(d) and d > 0:
                depths.append(float(d))

    if not depths:
        return None

    return float(np.mean(depths))

def remove_boxes_near_border(detection_objects, img_width, img_height, margin_ratio=0.05):
    """
    删除靠近图像边界的检测框
    detection_objects: list of [class_name, bbox, confidence, xy, depth]
    img_width, img_height: 图像宽高
    margin_ratio: 边界阈值，框离边界小于该比例宽高就删除
    返回: 过滤后的 detection_objects
    """
    margin_x = img_width * margin_ratio
    margin_y = img_height * margin_ratio

    filtered_objects = []

    for det in detection_objects:
        bbox = det[1]  # bbox = [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

        # 如果框在安全区域内，则保留
        if x1 >= margin_x and y1 >= margin_y and x2 <= img_width - margin_x and y2 <= img_height - margin_y:
            filtered_objects.append(det)
        else:
            # 可以打印一下被删除的框（调试用）
            print(f"删除靠边框: {det}")

    return filtered_objects

def draw_filtered_detections(img, filtered_objects):
    """
    img: 原始图像
    filtered_objects: [[class_name, bbox, confidence, xy, depth], ...]
    返回：绘制好的图像
    """
    imgs=img.copy()
    for det in filtered_objects:
        class_name, bbox, confidence, xy, depth = det
        x1, y1, x2, y2 = map(int, bbox)
        
        # 画矩形框
        cv2.rectangle(imgs, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签（类别+置信度+深度）
        label = f"{class_name} {confidence:.2f} D={depth:.2f}"
        cv2.putText(imgs, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    
    return imgs


def image_information(r1,obs,car,upto,step):
        start = time.time()
        detection_objects_orin = []
        img_width, img_height = obs['color_sensor'].shape[1], obs['color_sensor'].shape[0]
        focal_length_x, focal_length_y = calculate_focal_length_from_hfov_vfov(91, 65, img_width, img_height)
        
        img = r1.plot()
        img1=obs['color_sensor']
        pos,rot= car.get_agent_state()
        print(pos,rot)
        pos1=pos.tolist()
        rot1=rot.tolist()
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
                    x1, y1, x2, y2 = box.xyxy[0].cpu()
                
                # 计算中心点
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                ex = [x, y]
                
                # 确保坐标是整数类型
                x_int = int(round(x.item() if hasattr(x, 'item') else x))
                y_int = int(round(y.item() if hasattr(y, 'item') else y))
                
                # 安全地获取深度值
                try:
                    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                    depth =get_avg_depth_in_box(obs['depth_sensor'],x1, y1, x2, y2)
                    if depth==None:
                        continue   # 注意：numpy数组是 [height, width] 顺序
                except IndexError:
                    print(f"坐标超出范围: ({y_int}, {x_int})，使用默认深度值")
                    depth = 1.0  # 默认深度值
                
                xy_1 = image_to_local(x_int, y_int, depth, (img_height, img_width), focal_length_x, focal_length_y)
                xy = local_to_global(agent_state.position, agent_state.rotation, xy_1)
                
                detection = [
                    class_name,
                    bbox,
                    confidence,
                    xy.tolist() if hasattr(xy, 'tolist') else xy,  # 确保xy是列表格式
                    depth
                ]

                detection_objects_orin.append(detection)
                print(detection)
        detection_objects=remove_boxes_near_border(detection_objects_orin, img_width, img_height, margin_ratio=0.05)
        img_2=draw_filtered_detections(img1, detection_objects)
        inf = detection_objects
        timestamp = datetime.now().strftime("%Y%m%d")

        folder_name = f"/home/nvidia/mf/test_file/semantic_graphs/detection_{timestamp}_step{step}"
        folder_name1=f"{folder_name}/{upto}"
        os.makedirs(folder_name, exist_ok=True)
        os.makedirs(folder_name1, exist_ok=True)
        cv2.imwrite(f"{folder_name1}/result.jpg", img_2)
        cv2.imwrite(f"{folder_name1}/result_orin.jpg", img1)
        cv2.imwrite(f"{folder_name1}/result_orin_plot.jpg", img)

        # 保存JSON信息（包含位置信息）
        json_data = {
                "timestamp": timestamp,
                "detections": [
                    {
                        "class_name": det[0],
                        "bbox": {"x1": det[1][0], "y1": det[1][1], "x2": det[1][2], "y2": det[1][3]},
                        "confidence": det[2],
                        "global_position": det[3] , # 添加全局位置信息
                        "pos_rot":{"pos":pos1,"rot":rot1},
                        "depth":det[4]
                    } for det in detection_objects
                ]
            }

        with open(f"{folder_name1}/info.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 保存完成: {folder_name1}")


start=time.time()
car = CarController(port='/dev/ttyUSB0')
if car.connect():
    car.initialize_car()
    rgb,dep=main_camera()
    rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    r1,model=process_image(rgb)
    step=1
    upto=0
    while True:
            rgb,dep=main_camera()
            rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            obs={}
            obs['color_sensor']=rgb
            obs['depth_sensor']=dep
            r1 = model(rgb, device='0', conf=0.3, verbose=True)[0]
            image_information(r1,obs,car,upto,step)
            success=car.rotate(90)
            upto=upto+1
            if upto==4:
                time.sleep(1)
                success=car.move(0.5)
                upto=0
                step=step+1

else:
    print('false')


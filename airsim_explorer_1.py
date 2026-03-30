import time
import sys
import signal
import logging
import threading
import os
import json
# import habitat_sim
import networkx as nx
import numpy as np
from datetime import datetime
import cv2
import quaternion
import traceback
import queue
from ultralytics import YOLO
# from sim_util import sim_setup(仿真使用)
from util import *


# Hyperparameters
DETECTION_RADIUS = 5  # Distance threshold in meters

# Add environment name constant
ENVIRONMENT_NAME = "Building99"  # Change this based on your environment

def image_to_local(x_pixel, y_pixel, depth, resolution, focal_length_x, focal_length_y):#自己写的，有可能有问题
    """
    Converts image pixel coordinates back to local 3D coordinates given depth.

    Args:
        x_pixel (int): The x coordinate of the pixel.
        y_pixel (int): The y coordinate of the pixel.
        depth (float): The depth value at the pixel.
        resolution (tuple): The image resolution as (height, width).
        focal_length_x (float): The focal length in x direction (pixels).
        focal_length_y (float): The focal length in y direction (pixels).

    Returns:
        np.ndarray: The 3D point in local coordinates [x, y, z].
    """
    # 从像素坐标计算归一化坐标
    x_normalized = (x_pixel - resolution[1] / 2) / focal_length_x
    y_normalized = (y_pixel - resolution[0] / 2) / focal_length_y
    
    # 使用深度信息恢复3D坐标
    x_3d = x_normalized * depth
    y_3d = y_normalized * depth
    z_3d = depth
    
    # 应用坐标轴变换（与local_to_image中的变换相反）
    local_point = np.array([x_3d, -y_3d, -z_3d])
    
    return local_point

def calculate_focal_length_from_hfov_vfov(hfov, vfov, img_width, img_height):
    h_focal_px = calculate_focal_length_px(hfov, img_width)
    v_focal_px = calculate_focal_length_px(vfov, img_height)
    return h_focal_px, v_focal_px

class AirSimExplorer:
    def __init__(self,path):
        self.path_to_config=path
        self.habitat=None
        self.initialize_airsim_client()
        self.G = nx.Graph()  # Using NetworkX graph directly
        self.is_running = True
        self.drone_controller = None#后续需要重写
        self.detection_visualizer = None#后续需要重写
        self.visualization_thread = None
        self.model=None
        self.obj_temp=[]
        self.last_save_time = time.time()
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(os.path.dirname(__file__), "semantic_graphs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.output_dir, f"exploration_{ENVIRONMENT_NAME}_{timestamp}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Starting exploration of {ENVIRONMENT_NAME}")

        self._shutdown_event = threading.Event()
        self._threads = []
        self._update_lock = threading.Lock()
        self._visualization_lock = threading.Lock()
        self._shutdown_timeout = 5.0  # 5 seconds timeout for shutdown

        # Add node logging parameters
        self.min_distance_between_nodes = 0.5  # Minimum 0.5 meters between nodes
        self.min_time_between_nodes = 0.5      # Minimum 0.5 seconds between nodes
        self.last_node_position = None
        self.last_node_time = time.time()

    def initialize_airsim_client(self):#以完成修改（后续需要补加机器人控制以及识别相关内容）
        path_to_config =self.path_to_config
        with open(path_to_config) as f:
            cfg_first = json.load(f)  
        self.habitat=sim_setup(cfg_first)
        cfg=self.habitat.make_simple_cfg(cfg_first)
        # Set initial pose with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.habitat.sim = habitat_sim.Simulator(cfg)
                self.habitat.agent_state_set(cfg_first)
                self.model=YOLO("yolov8m.pt")
                self.queue = []
                print("Connected!")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Initialization attempt {attempt + 1} failed, retrying...")
                time.sleep(1)

    # def maintain_minimum_height(self):
    #     """Thread function to maintain minimum height"""
    #     while self.is_running:
    #         try:
    #             state = self.client.getMultirotorState()
    #             current_height = state.kinematics_estimated.position.z_val
                
    #             if current_height > self.MIN_HEIGHT + 0.1:  # If too low
    #                 self.client.moveToZAsync(self.MIN_HEIGHT, 1).join()
                
    #             time.sleep(0.1)
    #         except Exception as e:
    #             logging.error(f"Error in height maintenance: {e}")
    #             time.sleep(0.1)

    def take_off(self):
        self.client.takeoffAsync().join()

    def add_drone_node(self, position, yaw):
        node_id = f"drone_{len([n for n in self.G.nodes() if 'drone' in str(n)])}"
        self.G.add_node(node_id, 
                       position=position,
                       yaw=yaw,
                       type='drone',
                       level=0)
        return node_id

    def add_object_node(self, label, data):
        """Add object node to graph with proper position formatting"""
        if label not in self.G:
            print(f"Adding object node: {label}")
            try:
                self.G.add_node(label, 
                              position=data['position'],
                              type='object',
                              box2D=data['box2D'],
                              box3D=data['box3D'],
                              level=0)
                print(f"Successfully added object node: {label} at position {data['position']}")
            except Exception as e:
                print(f"Error adding object node {label}: {e}")

    def get_image_information(self):
        time_start=time.time()
        objects=[]
        obs=self.habitat.sim.get_sensor_observations()
        results = self.model(obs["color_sensor"][:,:,:3], show=False)
        depth=obs["depth_sensor"]
        r = results[0]
        object = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            object_name = r.names[cls_id]
            conf = float(box.conf[0])
            if conf < 0.5:
                continue
            # YOLO输出的bbox格式为 [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # 2D box 格式
            box2D = {
                "min": {"x": x1, "y": y1},
                "max": {"x": x2, "y": y2}
            }
            
            # 如果你有分割结果，可以取mask中心点估算位置（可选）
            position_dict_x=(x1+x2)/2
            position_dict_y=(y1+y2)/2
            position_dict = {"x": (x1+x2)/2, "y": (y1+y2)/2}  # 示例：2D中心点

            local_1=depth_to_height(depth,90,1,1,60)

            box_3d_position=local_1[int(position_dict_y),int(position_dict_x)]
            box_3d_local_min=local_1[int(y1),int(x1)]
            box_3d_local_max=local_1[int(y2),int(x2)]
            
            # YOLO本身不提供box3D，所以我们构造一个占位结构
            box3D = {
                "min": {"x": box_3d_local_min[0], "y": box_3d_local_min[1], "z": box_3d_local_min[2]},
                "max": {"x": box_3d_local_max[0], "y":box_3d_local_max[1], "z": box_3d_local_max[2]}
            }
            drone_obs=self.habitat.agent.get_state()
            drone_pose = drone_obs.position
            drone_rot=drone_obs.rotation

            self.obj_temp.append({
                "id": object_name,
                "confidence": conf,
                "position": position_dict,
                "box2D": box2D,
                "box3D": box3D,
                "position_3d": box_3d_position,
                "drone_pose": drone_pose,
                "drone_rot": drone_rot
            })
        if len(self.obj_temp)>10:
            label=self.label_filter(self.obj_temp)
            for lables in label:
                self.queue.append(lables)
    def label_filter(self,objects):
        count=0
        label=[]
        id_count = {}  # 用于统计每个id出现的次数
        for obj in objects:
            obj['repeated'] = False
            obj['id_r']=False
        for i in range(len(objects)):
            position = np.array(objects[i]['position_3d'])
            for j in range(i + 1, len(objects)):   # 从 i+1 开始，避免重复
                position_2 = np.array(objects[j]['position_3d'])
                dist = np.linalg.norm(position - position_2)
                if objects[i]['id']==objects[j]['id']:
                    if dist < 0.1:
                        objects[j]['repeated']=True
        for object in objects:
            if not object['repeated']:
                label.append(object)
        for obj in label:
            original_id = obj['id']
            # 如果这个id第一次出现，就从1开始计数
            id_count[original_id] = id_count.get(original_id, 0) + 1
            # 生成新的命名
            obj['id_renamed'] = f"{original_id}_{id_count[original_id]}"
        return label

    def local_to_global(self,position, orientation, local_point):
        """
        Transforms a local coordinate point to global coordinates based on position and quaternion orientation.

        Args:
            position (np.ndarray): The global position.
            orientation (quaternion.quaternion): The quaternion representing the rotation.
            local_point (np.ndarray): The point in local coordinates.

        Returns:
            np.ndarray: Transformed global coordinates.
        """
        rotated_point = quaternion.rotate_vectors(orientation, local_point)
        global_point = rotated_point + position
        return global_point

    def get_semantic_data(self):
        try:
            objects=[]
            for i in range(len(self.queue)):
                for obj in self.queue:
                    objects.append(obj)

            print(f"\nNumber of detections: {len(objects)}")

            semantic_labels = {}
            filter_words = ["floor", "ceiling"]

            for object in objects:
                try:
                    drone_pose = object['drone_pose']
                    drone_rot=object['drone_rot']
                    object_name = object['id']
                    if any(word in object_name.lower() for word in filter_words):
                        continue

                    print(f"\nProcessing detection: {object_name}")
                    
                    # Get object position directly from relative_pose
                    relative_position = object['position_3d']
                    print(f"Object relative position: {relative_position}")
                    
                    # Convert to global position
                    global_position = self.local_to_global(drone_pose, drone_rot,relative_position)
                    print(f"Object global position: {global_position}")
                    
                    # Convert position to dictionary format
                    position_dict = {
                        'x': float(global_position[0]),
                        'y': float(global_position[1]),
                        'z': float(global_position[2])
                    }
                    print(f"Position dictionary: {position_dict}")
                    
                    semantic_labels[object_name] = {
                        "id": object_name,
                        "position": position_dict,
                        "box2D": {
                            'min': {'x': object['box2D']['min']["x"], 'y': object['box2D']['min']["y"]},
                            'max': {'x': object['box2D']['max']["x"], 'y': object['box2D']['max']["y"]}
                        },
                        "box3D": {
                            'min': {'x':object['box3D']['min']["x"], 
                                   'y':object['box3D']['min']["y"],
                                   'z': object['box3D']['min']["z"]},
                            'max': {'x': object['box3D']['max']["x"],
                                   'y': object['box3D']['max']["y"],
                                   'z': object['box3D']['max']["z"]}
                        }
                    }
                    print(f"Successfully added semantic data for {object_name}")
                    
                except Exception as e:
                    print(f"Error processing detection {object_name}: {e}")
                    continue
            self.queue=[]
            self.obj_temp=[]
            print(f"\nTotal semantic labels collected: {len(semantic_labels)}")
            return semantic_labels
            
        except Exception as e:
            logging.error(f"Error in get_semantic_data: {str(e)}")
            return {}
    def array_to_dict(self,position):
        """Convert AirSim Vector3r to dictionary"""
        return {
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2])
        }
    
    def quaternion_to_yaw(self,quernion):
        """
        输入四元数 w, x, y, z
        返回偏航角 yaw，单位弧度
        """
        w=quernion.w
        x=quernion.x
        y=quernion.y
        z=quernion.z
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    def update_graph_and_semantics(self):
        """Update graph with drone positions and semantic information"""
        checkpoint_interval = 15 # Save every 1 minutes
        last_drone_node = None  # Keep track of the last drone node
        try:
            current_time = time.time()
            vehicle_pose = self.habitat.agent.get_state()
            current_position = vehicle_pose.position
            
            # Check if we should add a new node
            should_add_node = False
            should_add_obj_node=False
            time_since_last_node = current_time - self.last_node_time
            
            if self.last_node_position is not None:
                distance_moved = self._calculate_distance(
                    current_position,
                    self.last_node_position
                )
                print(f"Distance moved: {distance_moved:.2f}m, Time since last node: {time_since_last_node:.2f}s")
                
                if distance_moved >= self.min_distance_between_nodes :
                    should_add_node = True
                    if len(self.queue)>10:
                        should_add_obj_node = True
            else:
                # First node
                should_add_node = True
            
            if should_add_node:
                agent_state=self.habitat.agent.get_state()
                position = agent_state.position
                yaw = agent_state.rotation
                position=self.array_to_dict(position)
                yaw=self.quaternion_to_yaw(yaw)
                # Add new drone node
                current_drone_node = self.add_drone_node(position, yaw)
                print(f"Added drone node {current_drone_node} at position: {position}")
                
                # Connect to previous drone node if it exists
                if last_drone_node is not None:
                    distance = self._calculate_distance(
                        current_position,
                        self.last_node_position
                    )
                    self.G.add_edge(
                        last_drone_node, 
                        current_drone_node,
                        distance=distance,
                        type='drone_path'
                    )
                    print(f"Added drone path edge between {last_drone_node} and {current_drone_node}")
                
                # Update tracking variables
                last_drone_node = current_drone_node
                self.last_node_position = current_position
                self.last_node_time = current_time
                
                # Get semantic data at new node position
                if should_add_obj_node:
                    semantic_labels = self.get_semantic_data()
                    for label, data in semantic_labels.items():
                        self.add_object_node(label, data)

            # Check for checkpoint save
            if current_time - self.last_save_time >= checkpoint_interval:
                self.save_graph(final=False)
                self.last_save_time = current_time

            time.sleep(0.1)
            
        except Exception as e:
            logging.error(f"Error in update_graph_and_semantics: {str(e)}")
            traceback.print_exc()
            time.sleep(0.1)

    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def shutdown(self):#关闭智能体控制，检测器以及相关所有其他内容
        """Quick and forceful shutdown"""
        print("\nInitiating shutdown sequence...")
        
        try:
            # Set all shutdown flags immediately
            self.is_running = False
            self._shutdown_event.set()
            if hasattr(self, 'drone_controller'):
                self.drone_controller.is_running = False
            if hasattr(self, 'detection_visualizer'):
                self.detection_visualizer.is_running = False

            # Quick cleanup of visualization
            cv2.destroyAllWindows()
            
            # Save final graph
            try:
                print("Saving final graph...")
                with self._update_lock:
                    self.save_graph(final=True)
            except Exception as e:
                logging.error(f"Error saving final graph: {e}")

            # Quick landing sequence
            try:
                print("Landing drone...")
                self.client.landAsync()
                time.sleep(1)  # Brief wait
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except Exception as e:
                logging.error(f"Error during landing: {e}")
                # Emergency disarm
                try:
                    self.client.armDisarm(False)
                except:
                    pass

            print("Shutdown complete")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
        finally:
            # Force exit immediately
            os._exit(0)

    def run(self):
        try:
            # Start height maintenance thread
            self.get_image_information()
            
            # Create and start update thread
            self.update_graph_and_semantics()
            
            # Start visualization(后续重写)
            #self.detection_visualizer.start()
            
            # Give threads time to start
            time.sleep(1)
            
            # Start drone control (main thread)
            # try:
            #     self.drone_controller.keyboard_control()
            # except Exception as e:
            #     logging.error(f"Error in keyboard control: {e}")
            #     raise
            
        except Exception as e:
            logging.error(f"Error during run: {e}")
            traceback.print_exc()
            self.shutdown()

    def save_graph(self, final=False):
        """
        Save the graph with timestamp and environment name
        Args:
            final (bool): If True, indicates this is the final save on program exit
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "final" if final else "checkpoint"
        filename = f"{prefix}_semantic_graph_{ENVIRONMENT_NAME}_{timestamp}.gml"
        filepath = os.path.join(self.output_dir, filename)
        
        logging.info(f"Saving graph with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
        logging.info(f"Saving to: {filepath}")
        
        # Save graph attributes
        self.G.graph['environment'] = ENVIRONMENT_NAME
        self.G.graph['timestamp'] = timestamp
        self.G.graph['detection_radius'] = DETECTION_RADIUS
        
        nx.write_gml(self.G, filepath)
        print(f"Graph saved to {filepath}")


    def get_some_position(self, r1, obs):
        start = time.time()
        a = self.last_node_time
        detection_objects = []
        img_width, img_height = obs['color_sensor'].shape[1], obs['color_sensor'].shape[0]
        focal_length_x, focal_length_y = calculate_focal_length_from_hfov_vfov(90, 60, img_width, img_height)
        
        img = r1.plot()
        
        agent_state = self.habitat.agent.get_state()
        
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
        folder_name = f"detection_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)

        # 保存图片
        cv2.imwrite(f"{folder_name}/result.jpg", img)

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

    
    def __del__(self):
        """Cleanup method"""
        try:
            if hasattr(self, '_shutdown_event') and not self._shutdown_event.is_set():
                self.shutdown()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C more forcefully"""
    print("\nCtrl+C detected. Starting graceful shutdown...")
    try:
        if explorer:
            explorer.shutdown()
    except Exception as e:
        logging.error(f"Error during signal handling: {e}")
    finally:
        # Force exit after timeout
        time.sleep(2)  # Give a chance for logs to be written
        os._exit(0)

if __name__ == "__main__":
    
    
    print(f"Starting exploration of {ENVIRONMENT_NAME}")
    print(f"Graphs will be saved in: {os.path.abspath(os.path.join(os.path.dirname(__file__),'semantic_graphs'))}")
    
    explorer = None
    try:
        air=AirSimExplorer("/home/sjs/test_file/config/config.json")
        air.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        if explorer:
            explorer.shutdown()
    finally:
        # Force exit if we get here
        os._exit(0)

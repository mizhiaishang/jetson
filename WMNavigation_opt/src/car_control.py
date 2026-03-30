import sys
sys.path.append("/home/wheeltec/langChain_Test/pyorbbecsdk")

import serial
import struct
import time
import threading
import numpy as np
from typing import Optional, Tuple
import cv2
import asyncio
import math
from camera import OrbbecCamera


class CarController:
    """
    一个用于控制和监视机器人小车的类。
    它通过串口管理V2.1双向通讯协议，并使用一个独立的线程来持续接收和解析来自小车的状态数据，
    WMNavigation/src/car_control.py
    为上层应用提供简洁的阻塞式控制接口和实时的位姿查询。
    """

    # --- 命令ID常量 (Agent -> 小车) ---
    CMD_MOVE_FORWARD = 0x01
    CMD_MOVE_BACKWARD = 0x02
    CMD_ROTATE_CCW = 0x03  # 逆时针, 左转
    CMD_ROTATE_CW = 0x04   # 顺时针, 右转
    CMD_STOP = 0x05
    CMD_INIT = 0x06
    CMD_PATH_TRACKING = 0x07
    # 【新增】初始化命令ID

    # --- 状态码常量 (小车 -> Agent) ---
    STATUS_IDLE = 0x00
    STATUS_EXECUTING = 0x01
    STATUS_CMD_SUCCESS = 0x02
    STATUS_CMD_FAILED = 0x03
    

    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.raw_yaw_deg = 0.0
        self.ser: Optional[serial.Serial] = None
        
        # --- [新增] 状态变量，用于存储处理后给Agent的数据 ---
        self.position_z = -0.2
        self.position = np.array([0.0, 0.0, self.position_z], dtype=np.float32)
        self.rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) # 四元数 (x,y,z,w)
        # 【新增】用于存储目标点和指令速度
        # 目标位置 [x, y, yaw]
        self.debug_target = np.array([0.0, 0.0, 0.0], dtype=np.float32) 
        # 指令速度 [vx, vy, wz]
        self.debug_cmd_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.debug_motro_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.debug_cmd_vel_true = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.bezier_pos = np.array([0.0, 0.0], dtype=np.float32)

        self.current_status = self.STATUS_IDLE
        
        # 用于线程同步
        self.is_running = False
        self.command_completed_event = threading.Event()
        self.last_command_status: Optional[int] = None
        
        self.listener_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock() # 保护共享状态变量

    def connect(self) -> bool:
        """打开串口连接并启动监听线程。"""
        if self.is_running:
            #print("Controller is already connected.")
            return True
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.is_running = True
            self.listener_thread = threading.Thread(target=self._listen_for_data, daemon=True)
            self.listener_thread.start()
            #print(f"Successfully connected to car on {self.port}.")
            return True
        except serial.SerialException as e:
            #print(f"Error: Failed to connect to car on {self.port}. {e}")
            self.ser = None
            return False

    def disconnect(self):
        """关闭监听线程和串口连接。"""
        if not self.is_running:
            return
        self.is_running = False
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join()
        if self.ser and self.ser.is_open:
            self.ser.close()
        #print("Disconnected from car.")

    def _send_command(self, command_id: int, param: float = 0.0):
        """(内部方法) 构建并发送一个8字节的指令帧。"""
        if not self.ser or not self.ser.is_open:
            #print("Error: Not connected to car. Cannot send command.")
            return

        param_bytes = struct.pack('<f', param) # 把 float 参数变成 4 字节（小端）
        #print(len(param_bytes))

        command_byte = struct.pack('<B', command_id) # 把命令ID变成 1 字节
        payload = command_byte + param_bytes
        #print(len(payload))

        checksum = sum(payload) & 0xFF
        checksum_byte = struct.pack('<B', checksum)

        frame = b'\xAA' + payload + checksum_byte + b'\xDD'
        #print(frame);
        #print(len(frame))
        #print(frame.hex()) 
        self.ser.write(frame)

    def _listen_for_data(self):
            """(在独立线程中运行) 持续监听、解析并处理来自小车的18字节状态帧。"""
            buffer = bytearray()
            header = b'\xAA\xAA'
            trailer = b'\xDD'
            frame_len = 94

            while self.is_running:
                try:
                    # 从串口读取数据
                    data = self.ser.read(self.ser.in_waiting or 1)
                    if data:
                        # 【新增的调试代码】将收到的原始数据以十六进制格式打印出来
                        #print(f"--- Received Raw Data: {data.hex()} ---")
                    
                        buffer.extend(data)
                        while True:
                            start_idx = buffer.find(header)
                            if start_idx == -1:
                                if len(buffer) > frame_len: buffer.clear()
                                break
                        
                            if len(buffer) < start_idx + frame_len: break
                        
                            frame = buffer[start_idx : start_idx + frame_len]
                        
                            if frame.endswith(trailer):
                                self._parse_status_frame(bytes(frame))
                                del buffer[:start_idx + frame_len]
                            else:
                                del buffer[:start_idx + 1]
                except Exception as e:
                    print(f"An error occurred in listener thread: {e}")
                    self.disconnect()
                    break

    def _parse_status_frame(self, frame: bytes):
            """
            解析 42 字节的状态帧
            协议: Header(2) + Status(1) + CmdID(1) + CurrPos(12) + TarPos(12) + CmdVel(12) + Sum(1) + Tail(1)
            
            协议: Header(2) + Payload(90) + Sum(1) + Tail(1) = 94字节
            Payload: Status(1) + CmdID(1) + 22个float(88) = 90字节
            """
            # 1. 截取有效负载 (去掉头2字节，去掉校验和1字节+尾1字节)
            # 范围: frame[2 : 42-2] = frame[2 : 40]
            # payload = frame[2:90]
            checksum_idx = len(frame) - 2
            # 有效负载是从 index 2 到 index 92 (不含92)
            payload = frame[2 : checksum_idx]

            # 2. 校验和检查
            received_checksum = frame[checksum_idx] # 倒数第2个字节 90
            calculated_checksum = sum(payload) & 0xFF
            if received_checksum != calculated_checksum:
                # print("校验失败") # 可选调试
                return

            # 3. 解包
            # '<' : 小端模式
            # 'BB': Status, LastCmdID (2字节)
            # 'fff': 当前 x, y, yaw   (3个float)
            # 'fff': 目标 x, y, yaw   (3个float) 【新增】
            # 'fff': 指令 vx, vy, wz  (3个float) 【新增】
            try:
                data = struct.unpack('<BB' + 'f'*22, payload)
            except struct.error:
                return

            status = data[0]
            last_cmd_id = data[1]
            
            # 提取数据
            curr_x, curr_y, curr_yaw     = data[2], data[3], data[4]
            # print('位置 & 航向角')
            # print(curr_x, curr_y, curr_yaw)

            tar_x,  tar_y,  tar_yaw      = data[5], data[6], data[7]
            cmd_vx, cmd_vy, cmd_wz       = data[8], data[9], data[10]
            bezier_x = data[11]
            bezier_y = data[12]

            # 提取新增的电机数据 (索引从 13 开始)
            # data[11], [12] 是 bezier
            lf_ref = data[13]
            lf_fdb = data[14]
            rf_ref = data[15]
            rf_fdb = data[16]

            lb_ref = data[17]
            lb_fdb = data[18]
            rb_ref = data[19]
            rb_fdb = data[20]
    
            cmd_vx_true, cmd_vy_true, cmd_wz_true = data[21], data[22], data[23]

            # 4. 角度处理 (0~360) - 仅针对当前角度做此处理，目标角度通常不需要
            if curr_yaw < 0:
                curr_yaw += 360

            # 5. 更新类属性 (线程安全)
            with self.lock:
                self.current_status = status
                self.raw_yaw_deg = curr_yaw # 角度
                
                # # --- 更新当前位置 (保持你之前的坐标系转换习惯: x取反, y映射到z取反) ---
                # self.position[0] = -curr_x  
                # self.position[1] = 0.2
                # self.position[2] = -curr_y 


                # 设备系 -> 上位机旧系（旋转90°）
                self.position[0] = curr_y
                self.position[1] = 0.0
                self.position[2] = self.position_z - curr_x 


                # print(self.position[0], self.position[1], self.position[2])

                self.position[2] = self.position_z - curr_x 


                # 计算四元数
                half_yaw = (curr_yaw * 3.1415926 / 180.0) / 2.0 # 弧度制
                qx = 0.0
                qz = 0.0
                self.rotation[0] = qx
                self.rotation[1] = np.sin(half_yaw) # y
                self.rotation[2] = qz
                self.rotation[3] = np.cos(half_yaw) # w
                
                # print(self.rotation[0],self.rotation[1],self.rotation[2],self.rotation[3])
                
                # --- 【新增】更新调试数据 ---
                # 目标点 (做同样的坐标转换，以便对比)
                self.debug_target[0] = tar_x
                self.debug_target[1] = tar_y # 对应 Z
                self.debug_target[2] = tar_yaw # 保持原始弧度
                
                # 指令速度 (直接存原始值)
                self.debug_cmd_vel[0] = cmd_vx
                self.debug_cmd_vel[1] = cmd_vy
                self.debug_cmd_vel[2] = cmd_wz

                self.bezier_pos[0] = bezier_x
                self.bezier_pos[1] = bezier_y

                self.debug_motro_vel[0] = lf_ref
                self.debug_motro_vel[1] = lf_fdb
                self.debug_motro_vel[2] = rf_ref
                self.debug_motro_vel[3] = rf_fdb

                self.debug_motro_vel[4] = lb_ref
                self.debug_motro_vel[5] = lb_fdb
                self.debug_motro_vel[6] = rb_ref
                self.debug_motro_vel[7] = rb_fdb

                self.debug_cmd_vel_true[0] = cmd_vx_true
                self.debug_cmd_vel_true[1] = cmd_vy_true
                self.debug_cmd_vel_true[2] = cmd_wz_true

            # 6. 事件触发
            if status == self.STATUS_CMD_SUCCESS or status == self.STATUS_CMD_FAILED:
                self.last_command_status = status
                self.command_completed_event.set()


    def initialize_car(self):
        """
        新增】发送一个命令让小车重置状态并初始化所有参数。
        """
        #print("Sending initialize command to the car...")
        # 初始化指令是一个瞬时动作，不需要等待完成
        self._send_command(self.CMD_INIT, abs(0))
        # 短暂延时确保指令被处理
        time.sleep(1)  



    def move(self, distance: float, timeout: float = 30.0) -> bool:
        """控制小车前进或后退。这是一个阻塞函数。"""
        self.command_completed_event.clear()

        cmd_id = self.CMD_MOVE_FORWARD if distance > 0 else self.CMD_MOVE_BACKWARD

        self._send_command(cmd_id, abs(distance))
        
        completed = self.command_completed_event.wait(timeout)
        if not completed:
            #print(f"Warning: Move command timed out after {timeout} seconds.")
            self.stop()
            return False
        return self.last_command_status == self.STATUS_CMD_SUCCESS


    def rotate(self, angle: float, timeout: float = 30.0) -> bool:
    # """控制小车旋转。这是一个阻塞函数。
    # :param angle_deg: 旋转角度（度），正值为逆时针，负值为顺时针
    # :param timeout: 超时时间（秒）
    # :return: 是否成功执行
    # """
        self.command_completed_event.clear()

        #angle_180 = np.deg2rad(angle)  # 将角度转换为弧度

        cmd_id = self.CMD_ROTATE_CCW if angle > 0 else self.CMD_ROTATE_CW

        self._send_command(cmd_id, abs(angle))
        
        completed = self.command_completed_event.wait(10)
        
        if not completed:
            print(f"Warning: Rotate command timed out after {timeout} seconds.")
            self.stop()
            return False
            
        return self.last_command_status == self.STATUS_CMD_SUCCESS


    def track_path(self, timeout: float = 100.0) -> bool:
        """
        发送轨迹跟踪指令，并阻塞等待任务完成
        """
        if not self.ser or not self.ser.is_open:
            print("Error: Serial not connected.")
            return False

        print(f"Sending PATH_TRACKING command (ID: 0x{self.CMD_PATH_TRACKING:02X})...")
        
        # 1. 清除完成标志
        self.command_completed_event.clear()
        self.last_command_status = self.STATUS_EXECUTING # 重置状态

        # 2. 发送指令 (参数设为0，因为C代码里路径是静态写死的)
        self._send_command(self.CMD_PATH_TRACKING, 0.0)

        # 3. 阻塞等待
        # C代码逻辑：跑完点后发送 STATUS_CMD_SUCCESS，或者急停发送 STATUS_CMD_FAILED
        completed = self.command_completed_event.wait(timeout)

        if not completed:
            print(f"Warning: Path tracking timed out after {timeout}s.")
            self.stop() # 超时则发送急停
            return False

        if self.last_command_status == self.STATUS_CMD_SUCCESS:
            print("Path tracking SUCCESS.")
            return True
        else:
            print(f"Path tracking FAILED/STOPPED. Status: {self.last_command_status}")
            return False

    def stop(self):
        """发送紧急停止指令。"""
        print('ok')
        self._send_command(self.CMD_STOP)

    def get_agent_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        【新增】获取封装好的、可直接供Agent使用的位姿信息。
        :return: (position, rotation) -> (np.array[x,y,z], np.array[x,y,z,w])
        """
        # with self.lock:
        #     pos = self.position.copy()
        #     rot = self.rotation.copy()
            
        # pos[0] = -pos[0]
        # pos[2] = -pos[2]
        # return pos, rot
        return self.position.copy(), self.rotation.copy()

    def resize_image_opencv(image, target_width=1280, target_height=720):
        original_height, original_width = image.shape[:2]
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def get_images():
        camera=OrbbecCamera()
        camera.initialize()
        time.sleep(1)
        rgb_image,dep_image=camera.save_current_frame()
        dep_image_mm=CarController.resize_image_opencv(dep_image)
        dep_image=dep_image_mm/1000
        dep_image=CarController.rgb_guided_depth_inpainting(dep_image,rgb_image)
        camera.shutdown()
        del camera
        return rgb_image,dep_image

    def _center_crop(img, h=480, w=640, pad_mode=None, pad_val=0):
        H, W = img.shape[:2]
        y, x = H//2 - h//2, W//2 - w//2
        y_end, x_end = y + h, x + w

        if y < 0 or x < 0 or y_end > H or x_end > W:
            if pad_mode is None:
                raise ValueError("Image too small for crop")
            pad = (max(0,-y), max(0,y_end-H)), ((max(0,-x), max(0,x_end-W)))
            if img.ndim == 3: pad += ((0,0),)
            img = np.pad(img, pad, mode=pad_mode, constant_values=pad_val)
            y, x = max(y,0), max(x,0)

        return img[y:y+h, x:x+w] if img.ndim == 2 else img[y:y+h, x:x+w, :]

    def rgb_guided_depth_inpainting(depth_img, rgb_img, radius=5, eps=0.01):
        """
        使用RGB图像引导的联合双边滤波修复深度图像
        """
        # 将NaN转换为0并创建掩膜
        mask = np.isnan(depth_img).astype(np.uint8) * 255
        depth_img_no_nan = np.nan_to_num(depth_img)
        # 归一化RGB图像到0-1范围
        rgb_normalized = rgb_img.astype(np.float32) / 255.0
        # 使用导向滤波
        guided_filter = cv2.ximgproc.createGuidedFilter(
            guide=rgb_normalized,
            radius=radius,
            eps=eps
        )
        # 应用导向滤波
        inpainted_depth = guided_filter.filter(
            src=depth_img_no_nan.astype(np.float32),
            dst=np.zeros_like(depth_img_no_nan))
        # 仅替换原来NaN的区域
        result = depth_img.copy()
        result[np.isnan(result)] = inpainted_depth[np.isnan(result)]
        return result
    async def stop_test(self,main_task):
        await asyncio.sleep(0.5)
        main_task.cancel()

async def run_main(car):
            main_task = asyncio.create_task(car.move(1))
            monitor_task = asyncio.create_task(car.stop_test(main_task))
            await asyncio.gather(main_task, monitor_task, return_exceptions=True)

if __name__ == '__main__':
    car = CarController(port='/dev/ttyUSB0')
    try:
        # rgb_image, dep_image = CarController.get_images()
        if car.connect():
            # car.initialize_car()
            pos1, rot1 = car.get_agent_state()
            print('初始位置' + str(pos1), rot1)
            while True:
                # car.move(0.2)
                pos2, rot2 = car.get_agent_state()
                print('实时位置' + str(pos2), rot2)
                time.sleep(1)


        print('ok')
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        car.disconnect()

# car.initialize_car()
# pos1, rot1 = car.get_agent_state()
# print('初始位置' + str(pos1), rot1)
# # while True:
# car.move(0.2)
# pos2, rot2 = car.get_agent_state()
# print('第一个点' + str(pos2), rot2)
# time.sleep(1)
# success=car.rotate(90)
# pos2, rot2 = car.get_agent_state()      
# print(pos2,rot2)
# time.sleep(1)

# car.move(0.2)
# pos2, rot2 = car.get_agent_state()
# print('第二个点' + str(pos2), rot2)
# time.sleep(1)
# success=car.rotate(90)
# pos2, rot2 = car.get_agent_state()      
# print(pos2,rot2)
# time.sleep(1)

# car.move(0.2)
# pos2, rot2 = car.get_agent_state()
# print('第三个点' + str(pos2), rot2)
# time.sleep(1)
# success=car.rotate(90)
# pos2, rot2 = car.get_agent_state()      
# print(pos2,rot2)
# time.sleep(1)

# car.move(0.2)
# pos2, rot2 = car.get_agent_state()
# print('第四个点' + str(pos2), rot2)
# time.sleep(1)
# success=car.rotate(90)
# pos2, rot2 = car.get_agent_state()      
# print(pos2,rot2)
# time.sleep(1)


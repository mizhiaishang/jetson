import socket
import time
import os

UDP_IP = "0.0.0.0"   # 本机接收所有网口
UDP_PORT = 2368      # Velodyne 默认点云端口
SAVE_INTERVAL = 0
OUTPUT_DIR = "/home/nvidia/mf/test_file/point_cloud"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

pc_buffer = []
last_save = time.time()
count = 0

while True:
    data, addr = sock.recvfrom(1206)
    pc_buffer.append(data)

    if time.time() - last_save >= 20:
        filename = os.path.join(OUTPUT_DIR, f"pointcloud_raw_{count}.bin")
        with open(filename, "wb") as f:
            for pkt in pc_buffer:
                f.write(pkt)
        print(f"已保存 {filename}, 包数: {len(pc_buffer)}")
        pc_buffer = []
        last_save = time.time()
        count += 1

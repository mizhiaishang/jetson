import os
import json
import base64
import cv2
import numpy as np
import habitat_sim
import magnum as mn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import uvicorn
import asyncio

# ---------------------------
# EGL 无头渲染 + 禁用声音
os.environ["MAGNUM_DEFAULT_RENDERER"] = "EGL"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# ---------------------------
# 加载配置
path_to_config = "/home/sjs/test_file/config/config.json"  # 改成你的路径
with open(path_to_config) as f:
    cfg_first = json.load(f)
model = YOLO("yolov8m.pt") 
# ---------------------------
# Habitat-Sim 初始化
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = cfg_first["scene"]
sim_cfg.gpu_device_id = 1

agent_cfg = habitat_sim.agent.AgentConfiguration()
rgb_sensor_spec = habitat_sim.CameraSensorSpec()
rgb_sensor_spec.uuid = "color_sensor"
rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor_spec.resolution = [cfg_first["height"], cfg_first["width"]]
rgb_sensor_spec.position = [0.0, cfg_first["sensor_height"], 0.0]
agent_cfg.sensor_specifications = [rgb_sensor_spec]

# 动作空间
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(cfg_first['move_length'])
    ),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left", habitat_sim.agent.ActuationSpec(cfg_first['rotate_angle'])
    ),
    "turn_right": habitat_sim.agent.ActionSpec(
        "turn_right", habitat_sim.agent.ActuationSpec(cfg_first['rotate_angle'])
    ),
}

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)
agent = sim.initialize_agent(0)

# 设置智能体初始随机位置
state = habitat_sim.AgentState()
state.position = sim.pathfinder.get_random_navigable_point()
agent.set_state(state)

# ---------------------------
# FastAPI + WebSocket
app = FastAPI()

SERVER_IP = "192.168.137.101"  # 改成你的服务器 IP

html = f"""
<!DOCTYPE html>
<html>
  <body>
    <h3>Habitat-Sim Web 控制</h3>
    <canvas id="canvas" width="640" height="480" style="border:1px solid black;"></canvas>
    <canvas id="canvas1" width="640" height="480" style="border:1px solid black;"></canvas>
    <p>按 W/A/D 控制智能体移动</p>
<script>
      let ws = new WebSocket("ws://{SERVER_IP}:8000/ws");
      let canvas = document.getElementById("canvas");
      let ctx = canvas.getContext("2d");
      let canvas1 = document.getElementById("canvas1");
      let ctx1 = canvas1.getContext("2d");

      ws.onmessage = function(event) {{
        let [img1_b64, img2_b64] = event.data.split('|');  // 两张图用 | 分隔

        let img = new Image();
        img.src = 'data:image/jpeg;base64,' + img1_b64;
        img.onload = () => ctx.drawImage(img, 0, 0);

        let img2 = new Image();
        img2.src = 'data:image/jpeg;base64,' + img2_b64;
        img2.onload = () => ctx1.drawImage(img2, 0, 0);
      }};

      window.addEventListener("keydown", (e) => {{
        if(["w","a","d"].includes(e.key)) {{
          ws.send(e.key);
        }}
      }});
    </script>
  </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        count=1
        while True:
            try:
                key = await asyncio.wait_for(ws.receive_text(), timeout=0.03)
                if key == "w":
                    sim.step("move_forward")
                elif key == "a":
                    sim.step("turn_left")
                elif key == "d":
                    sim.step("turn_right")
            except asyncio.TimeoutError:
                pass  # 没有按键，继续循环

            # 获取图像
            obs = sim.get_sensor_observations()
            rgb = obs["color_sensor"]
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", rgb_bgr)
            data1 = base64.b64encode(buffer).decode("utf-8")


            results = model(rgb_bgr[:,:,:3], show=False)
            r1 = results[0]
            r=r1.plot()
            _, buffer = cv2.imencode(".jpg", r)
            data2 = base64.b64encode(buffer).decode("utf-8")

            await ws.send_text(f"{data1}|{data2}")

            await asyncio.sleep(0.03)  # 限制帧率 ~30FPS
    except Exception as e:
        print("WebSocket closed:", e)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

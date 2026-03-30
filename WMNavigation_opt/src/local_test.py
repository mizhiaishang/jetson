import sys
import time
import numpy as np
import cv2
import quaternion
sys.path.append("/home/wheeltec/WMNavigation_opt/pyorbbecsdk")

from camera import OrbbecCamera
from utils import *
from camare_test import *
from car_control import *

def agent_frame_to_image_coords(point, agent_state, sensor_state, resolution, focal_length):
    """
    Converts a point from agent frame to image coordinates.

    Args:
        point (np.ndarray): The point in agent frame coordinates.
        agent_state (6dof): The agent's state containing position and rotation.
        sensor_state (6dof): The sensor's state containing position and rotation.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple or None: The image coordinates (x_pixel, y_pixel), or None if the point is behind the camera.
    """
    global_p = local_to_global(agent_state[0], agent_state[1], point)
    camera_pt = global_to_local(sensor_state[0], sensor_state[1], global_p)
    if camera_pt[2] > 0:
        return None
    return local_to_image(camera_pt, resolution, focal_length)



a_1,b_1=get_images()
b=rgb_guided_depth_inpainting(b_1,a_1)
B=draw_depth(b)
cv2.imshow("Depth Visualization", a_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imshow("Depth Visualization", a_1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
car = CarController(port='/dev/ttyUSB0')
car.connect()
success = car.move(0)
pos,rot=car.get_agent_state()
car.disconnect()
x,y,z,w=rot[0],rot[1],rot[2],rot[3]
rot=np.array([w,x,y,z])
position_sensor=np.copy(pos)
position_sensor[2]=position_sensor[2]
position_sensor[1]=position_sensor[1]+0.15
s=np.copy(pos)
s[1]=s[1]-0.2
pos2=np.copy(s)
pos1=np.copy(position_sensor)
rot=quaternion.from_float_array(rot)
ags=[pos2,rot]
sgs=[pos1,rot]

global_point=depth_to_height(b,50,pos,rot)
print(global_point[240,320])
z=agent_frame_to_image_coords([0,0,0],ags,sgs,(480,640),686.2)


cv2.imshow("Depth Visualization", a_1)

cv2.waitKey(0)
cv2.destroyAllWindows()
print(global_point)
#85高,距离175，x轴负方向155左右
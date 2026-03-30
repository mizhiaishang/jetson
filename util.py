import sys
import time
import numpy as np
import cv2
import quaternion
import math
import ast
from scipy.spatial.transform import Rotation

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)

def theta_trans(theta,x,y):
    if x>0 and y>0:
        theta=theta
    elif x<0 and y>0:
        theta=2*np.pi-theta
    elif x<0 and y<0:
        theta=np.pi+theta
    else:
        theta=np.pi-theta
    return theta

def rotate_around_y_quaternion(original_quat, angle_radians):
    """
    将四元数绕世界Y轴旋转指定角度
    
    参数:
        original_quat: 原始四元数 [w, x, y, z] 格式
        angle_radians: 要旋转的角度（弧度）
    
    返回:
        旋转后的四元数 [w, x, y, z] 格式
    """
    # 将 [w, x, y, z] 格式转换为 scipy 的 [x, y, z, w] 格式

    w = original_quat.w
    x = original_quat.x
    y = original_quat.y
    z = original_quat.z
    original_quat_scipy = [x, y, z, w]
    
    # 创建原始旋转对象
    original_rot = Rotation.from_quat(original_quat_scipy)
    
    # 创建绕Y轴的旋转（使用弧度）
    y_rotation = Rotation.from_euler('y', angle_radians, degrees=False)
    
    # 组合旋转：先原始旋转，再绕Y轴旋转
    result_rot = original_rot * y_rotation
    
    # 将结果转换回 [w, x, y, z] 格式
    result_quat_scipy = result_rot.as_quat()  # [x, y, z, w]
    x, y, z, w = result_quat_scipy
    result_quat = [w, x, y, z]
    result_quat=np.quaternion(*result_quat) 
    return result_quat

def obs_solve(state_position,state_rotation,rgb_image,dep_image):
    w, x, y, z = state_rotation[3], state_rotation[0], state_rotation[1], state_rotation[2]
    inverse_rotation = np.quaternion(w, x, y, z) 
    state_rotation = np.quaternion.inverse(inverse_rotation)
    position_sensor=get_camera_global_position(state_position,state_rotation)
    state_rotation=state_rotation.normalized()
    color_sensor_inf={}
    sensor_states={}
    agent_state={}
    observation={}
    color_sensor_inf['position']=position_sensor
    color_sensor_inf['rotation']=state_rotation
    sensor_states['color_sensor']=color_sensor_inf
    sensor_states['depth_sensor']=color_sensor_inf
    agent_state['position']=state_position
    agent_state['rotation']=state_rotation
    agent_state['sensor_states']=sensor_states
    observation['color_sensor']=rgb_image
    observation['depth_sensor']=dep_image
    observation['agent_state']=agent_state
    return observation

def get_camera_global_position(car_position,car_orientation: quaternion.quaternion):
    camera_in_car = np.array([0.0, 0.3, 0.1])  # 相机在小车局部坐标下的固定位置
    # 将局部坐标旋转到全局，再加上小车位置
    camera_global = quaternion.rotate_vectors(car_orientation, camera_in_car) + car_position
    return camera_global


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


# def update_voxel(self, r: float, theta: float, agent_state, clip_dist: float,explored_map):
#     """Update the voxel map to mark actions as explored or unexplored"""
#     agent_coords = self._global_to_grid(agent_state.position)

#     # Mark explored regions
#     clipped = min(r, clip_dist)
#     local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
#     global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
#     point = self._global_to_grid(global_coords)
#     cv2.line(explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)


def calculate_focal_length_px(hfov_degrees, img_width):
    focal_length_px = img_width / (2 * np.tan(np.radians(hfov_degrees / 2)))
    return focal_length_px

def calculate_focal_length_from_hfov_vfov(hfov, vfov, img_width, img_height):
    h_focal_px = calculate_focal_length_px(hfov, img_width)
    v_focal_px = calculate_focal_length_px(vfov, img_height)
    return h_focal_px, v_focal_px

def agent_frame_to_image_coords(point, agent_state, sensor_state, resolution, focal_length_x,focal_length_y):
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
    global_p = local_to_global(agent_state.position, agent_state.rotation, point)
    camera_pt = global_to_local(sensor_state['position'], sensor_state['rotation'], global_p)
    return local_to_image(camera_pt, resolution, focal_length_x,focal_length_y)

def find_intersections(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int):
    """
    Find the intersections of a line defined by two points with the image boundaries.
    Args:
        x1 (int): The x-coordinate of the first point.
        y1 (int): The y-coordinate of the first point.
        x2 (int): The x-coordinate of the second point.
        y2 (int): The y-coordinate of the second point.
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        list of tuple or None: A list of two tuples representing the intersection points 
        with the image boundaries, or None if there are not exactly two intersections.
    """
    if x2 != x1:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    else:
        m = None  # Vertical line
        b = None

    intersections = []
    if m is not None and m != 0:  # Avoid division by zero for horizontal lines
        x_at_yh = int((img_height - b) / m)  # When y = img_height, x = (img_height - b) / m
        if 0 <= x_at_yh <= img_width:
            intersections.append((x_at_yh, img_height - 1))

    if m is not None:
        y_at_x0 = int(b)  # When x = 0, y = b
        if 0 <= y_at_x0 <= img_height:
            intersections.append((0, y_at_x0))

    if m is not None:
        y_at_xw = int(m * img_width + b)  # When x = img_width, y = m * img_width + b
        if 0 <= y_at_xw <= img_height:
            intersections.append((img_width - 1, y_at_xw))

    if m is not None and m != 0:  # Avoid division by zero for horizontal lines
        x_at_y0 = int(-b / m)  # When y = 0, x = -b / m
        if 0 <= x_at_y0 <= img_width:
            intersections.append((x_at_y0, 0))

    if m is None:
        intersections.append((x1, img_height - 1))  # Bottom edge
        intersections.append((x1, 0))  # Top edge

    if len(intersections) == 2:
        return intersections
    return None

def unproject_2d(x_pixel, y_pixel, depth, resolution, focal_length_x, focal_length_y):
    """
    Unprojects a 2D pixel coordinate back to 3D space given depth information.

    Args:
        x_pixel (int): The x coordinate of the pixel.
        y_pixel (int): The y coordinate of the pixel.
        depth (float): The depth value at the pixel.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple: The 3D coordinates (x, y, z).
    """
    x = (x_pixel - resolution[1] / 2) * depth / focal_length_x
    y = (y_pixel - resolution[0] / 2) * depth / focal_length_y
    return x, -y, -depth

def global_to_local(position, orientation, global_point):
    """
    Transforms a global coordinate point to local coordinates based on position and quaternion orientation.

    Args:
        position (np.ndarray): The global position.
        orientation (quaternion.quaternion): The quaternion representing the rotation.
        global_point (np.ndarray): The point in global coordinates.

    Returns:
        np.ndarray: Transformed local coordinates.
    """
    translated_point = global_point - position
    inverse_orientation = np.quaternion.conj(orientation)
    local_point = quaternion.rotate_vectors(inverse_orientation, translated_point)
    return local_point
def local_to_image(local_point, resolution, focal_length_x, focal_length_y):
    """
    Converts a local 3D point to image pixel coordinates.

    Args:
        local_point (np.ndarray): The point in local coordinates.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple: The pixel coordinates (x_pixel, y_pixel).
    """
    point_3d = [local_point[0], -local_point[1], -local_point[2]]  # Inconsistency between Habitat camera frame and classical convention
    if point_3d[2] == 0:
        point_3d[2] = 0.0001
    x = focal_length_x * point_3d[0] / point_3d[2]
    x_pixel = int(resolution[1] / 2 + x)

    y = focal_length_y * point_3d[1] / point_3d[2]
    y_pixel = int(resolution[0] / 2 + y)
    return x_pixel, y_pixel

def can_project(r_i: float, theta_i: float, agent_state, sensor_state,resolution,focal_length_x,focal_length_y):
    """
    Checks whether the specified polar action can be projected onto the image, i.e., not too close to the boundaries of the image.
    """
    agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]
    end_px = agent_frame_to_image_coords(
        agent_point, agent_state, sensor_state, 
        resolution, focal_length_x,focal_length_y
    )
    if end_px is None:
        return None

    if (
        0.04 * resolution[1] <= end_px[0] <= (1 - 0.04) * resolution[1] and
        0.04 * resolution[0] <= end_px[1] <= (1 - 0.04) * resolution[0]
    ):
        return end_px
    return None

def local_to_global(position, orientation, local_point):
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

def get_radial_distance(start_pxl: tuple, theta_i: float, navigability_mask: np.ndarray, 
                            agent_state, sensor_state, 
                            depth_image,resolution, focal_length_x, focal_length_y,pos):
    """
    Calculates the distance r_i that the agent can move in the direction theta_i, according to the navigability mask.
    """
    agent_point = [2 * np.sin(theta_i), 0, -2 * np.cos(theta_i)]
    end_pxl = agent_frame_to_image_coords(
        agent_point, agent_state, sensor_state, 
        resolution, focal_length_x, focal_length_y
    )
    if end_pxl is None or end_pxl[1] >= resolution[0]:
        return None, None,pos

    H, W = navigability_mask.shape

    # Find intersections of the theoretical line with the image boundaries
    intersections = find_intersections(start_pxl[0], start_pxl[1], end_pxl[0], end_pxl[1], W, H)
    if intersections is None:
        return None, None,pos

    (x1, y1), (x2, y2) = intersections
    num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
    if num_points < 5:
        return None, None,pos
    x_coords = np.linspace(x1, x2, num_points)
    y_coords = np.linspace(y1, y2, num_points)

    out = (int(x_coords[-1]), int(y_coords[-1]))
    if not navigability_mask[int(y_coords[20]), int(x_coords[0])]:
        return None, None,pos

    for i in range(num_points - 4):
        # Trace pixels until they are not navigable
        y = int(y_coords[i+20])
        x = int(x_coords[i+20])
        a=sum([navigability_mask[int(y_coords[j+20]), int(x_coords[j+20])] for j in range(i, i + 8)])
        if sum([navigability_mask[int(y_coords[j+20]), int(x_coords[j+20])] for j in range(i, i + 8)]) <= 5:
            out = (x, y)
            break

    if i < 5:
        return 0, theta_i,pos
    #use depth to get distance
    out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
    camera_coords = unproject_2d(
        *out, depth_image[out[1], out[0]], resolution, focal_length_x,focal_length_y
    )
    pos.append(out)
    local_coords = global_to_local(
        agent_state.position, agent_state.rotation,
        local_to_global(sensor_state['position'], sensor_state['rotation'], camera_coords)
    )
    r_i = np.linalg.norm([local_coords[0], local_coords[2]])

    return r_i, theta_i,pos

def project_onto_image(a_final: list, rgb_image: np.ndarray, agent_state, sensor_state,focal_length_x, focal_length_y,
        resolution,pos_ex):
        """
        Projects a set of actions onto a single image. Keeps track of action-to-number mapping.
        """
        scale_factor = rgb_image.shape[0] / 1080
        # if candidate_flag:
        #     scale_factor /= 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        circle_color = WHITE
        projected = {}
        start_px = agent_frame_to_image_coords(
                [0, 0, 0], agent_state, sensor_state, 
                resolution, focal_length_x,focal_length_y
            )
        for _, (r_i, theta_i) in enumerate(a_final):
            text_size = 2.4 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)

            end_px = can_project(r_i, theta_i, agent_state, sensor_state,resolution, focal_length_x,focal_length_y)
            end_px=pos_ex[_]
            print(end_px)
            if end_px is not None and r_i != 0.75:
                action_name = len(projected) + 1
                projected[(r_i, theta_i)] = action_name

                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), RED, math.ceil(5 * scale_factor), tipLength=0.0)
                text = str(action_name)
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = 0
                cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
                text_position = (circle_center[0] - text_width // 2 -20 , circle_center[1] + text_height // 2 + 20)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
        return projected


class AgentState():
    def __init__(self, agent_state):
        self.position = agent_state['position']
        self.rotation = agent_state['rotation']
        self.sensor_states=agent_state['sensor_states']

def navigability(obs,navigability_mask,resolution,hov,vov):
    agent_state = AgentState(obs['agent_state'])
    sensor_state = agent_state.sensor_states['color_sensor']
    rgb_image = obs['color_sensor']
    depth_image = obs[f'depth_sensor']
    sensor_range =  np.deg2rad(hov / 2) * 1.5

    all_thetas = np.linspace(-sensor_range, sensor_range, 10)
    focal_length_x,focal_length_y=calculate_focal_length_from_hfov_vfov(hov, vov, resolution[1],resolution[0])
    start = agent_frame_to_image_coords(
        [0, 0, 0], agent_state, sensor_state,
        resolution, focal_length_x,focal_length_y
    )  # agent的原点在图像坐标系的位置（像素）

    a_initial = []
    pos=[]
    for theta_i in all_thetas:
        r_i, theta_i,pos = get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image,resolution,focal_length_x,focal_length_y,pos)
        if r_i is not None and r_i != 0:
            # update_voxel(
            #     r_i, theta_i, agent_state,
            #     clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
            # )
            a_initial.append((r_i, theta_i))
    return a_initial,pos


# def action_proposer(a_initial, agent_state):(后续进行进一步提升修改)
#     """Refines the initial set of actions, ensuring spacing and adding a bias towards exploration."""
#     min_angle = self.fov/self.cfg['spacing_ratio']
#     explore_bias = self.cfg['explore_bias']
#     clip_frac = self.cfg['clip_frac']
#     clip_mag = self.cfg['max_action_dist']

#     explore = explore_bias > 0
#     unique = {}
#     for mag, theta in a_initial:
#         if theta in unique:
#             unique[theta].append(mag)
#         else:
#             unique[theta] = [mag]
#     arrowData = []

#     topdown_map = self.voxel_map.copy()
#     mask = np.all(self.explored_map == self.explored_color, axis=-1)
#     topdown_map[mask] = self.explored_color
#     for theta, mags in unique.items():
#         # Reference the map to classify actions as explored or unexplored
#         mag = min(mags)
#         cart = [self.e_i_scaling*mag*np.sin(theta), 0, -self.e_i_scaling*mag*np.cos(theta)]
#         global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
#         grid_coords = self._global_to_grid(global_coords)
#         score = (sum(np.all((topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color), axis=-1)) + 
#                 sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
#         arrowData.append([clip_frac*mag, theta, score<3])

#     arrowData.sort(key=lambda x: x[1])
#     thetas = set()
#     out = []
#     filter_thresh = 0.75
#     filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

#     filtered.sort(key=lambda x: x[1])
#     if filtered == []:
#         return []
#     if explore:
#         # Add unexplored actions with spacing, starting with the longest one
#         f = list(filter(lambda x: x[2], filtered))
#         if len(f) > 0:
#             longest = max(f, key=lambda x: x[0])
#             longest_theta = longest[1]
#             smallest_theta = longest[1]
#             longest_ndx = f.index(longest)
        
#             out.append([min(longest[0], clip_mag), longest[1], longest[2]])
#             thetas.add(longest[1])
#             for i in range(longest_ndx+1, len(f)):
#                 if f[i][1] - longest_theta > (min_angle*0.9):
#                     out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
#                     thetas.add(f[i][1])
#                     longest_theta = f[i][1]
#             for i in range(longest_ndx-1, -1, -1):
#                 if smallest_theta - f[i][1] > (min_angle*0.9):
                    
#                     out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
#                     thetas.add(f[i][1])
#                     smallest_theta = f[i][1]

#             for r_i, theta_i, e_i in filtered:
#                 if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle*explore_bias:
#                     out.append((min(r_i, clip_mag), theta_i, e_i))
#                     thetas.add(theta)

#     if len(out) == 0:
#         # if no explored actions or no explore bias
#         longest = max(filtered, key=lambda x: x[0])
#         longest_theta = longest[1]
#         smallest_theta = longest[1]
#         longest_ndx = filtered.index(longest)
#         out.append([min(longest[0], clip_mag), longest[1], longest[2]])
        
#         for i in range(longest_ndx+1, len(filtered)):
#             if filtered[i][1] - longest_theta > min_angle:
#                 out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
#                 longest_theta = filtered[i][1]
#         for i in range(longest_ndx-1, -1, -1):
#             if smallest_theta - filtered[i][1] > min_angle:
#                 out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
#                 smallest_theta = filtered[i][1]


#     if (out == [] or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
#         return self._get_default_arrows()
    
#     out.sort(key=lambda x: x[1])
#     return [(mag, theta) for mag, theta, _ in out]

def projection(a_final,obs,hov,vov,resolution,pos_ex):
    focal_length_x,focal_length_y=calculate_focal_length_from_hfov_vfov(hov, vov, resolution[1],resolution[0])
    images={}
    images['color_sensor']=obs['color_sensor']
    agent_state = AgentState(obs['agent_state'])
    a_final_projected = project_onto_image(
        a_final, images['color_sensor'], agent_state,
        agent_state.sensor_states['color_sensor'],
        focal_length_x, focal_length_y,
        resolution,pos_ex
    )

    # if not a_final_projected and (self.step_ndx - self.turned < self.cfg['turn_around_cooldown']) and not candidate_flag:
    #     logging.info('No actions projected and cannot turn around')
    #     a_final = self._get_default_arrows()
    #     a_final_projected = self._project_onto_image(
    #         a_final, images['color_sensor'], agent_state,
    #         agent_state.sensor_states['color_sensor'],
    #         step=self.step_ndx,
    #         goal=goal
    #     )

    return images['color_sensor'],a_final_projected

def theta_test(theta,theta_1):
    if theta_1>np.pi:
        theta=-theta
    else:
        theta=theta
    return theta

def action_make(zhangaiweizhi,zhangaidaxiao,pos,rot,dep_image,mubiaodian_pos,resolution,hov,vov,obs,mask):
    focal_length_x,focal_length_y=calculate_focal_length_from_hfov_vfov(hov, vov, resolution[1],resolution[0])
    agent_state = AgentState(obs['agent_state'])
    sensor_state = agent_state.sensor_states['color_sensor']
    zhongji_ex_x=zhangaiweizhi[0]+zhangaidaxiao+300
    zhongji_ex_y=zhangaiweizhi[1]-50
    zhongji_ex=[zhongji_ex_x,zhongji_ex_y]
    image = cv2.imread('/home/wheeltec/test_file/datanavi/1.png')  # 替换为你的图片路径
    target_point = (zhongji_ex_x, zhongji_ex_y)  # 替换为你的目标点坐标
    cv2.circle(image, target_point, 5, (0, 0, 255), -1)
    cv2.imwrite('/home/wheeltec/test_file/datanavi/3.png', image)
    ms=False
    if zhongji_ex_y>=720 or zhongji_ex_x>=1280:
        ms=True
    if ms or not mask[zhongji_ex_y,zhongji_ex_x]:
        zhongji_ex_x=zhangaiweizhi[0]+zhangaidaxiao-300
        zhongji_ex_y=zhangaiweizhi[1]-50
        zhongji_ex=[zhongji_ex_x,zhongji_ex_y]
        if not mask[zhongji_ex_y,zhongji_ex_x]:
            print('走不了')
            
    depth=dep_image[zhongji_ex[1],zhongji_ex[0]]
    zhongji_lc=image_to_local(zhongji_ex[0],zhongji_ex[1],depth,resolution,focal_length_x,focal_length_y)
    zhongji_gl=local_to_global(pos,obs['agent_state']['rotation'],zhongji_lc)
    pos_2 = np.array([pos[0], pos[2]])
    zhongji_gl_2 = np.array([zhongji_gl[0], zhongji_gl[2]])
    mubiaodian_pos_2=[mubiaodian_pos[0],mubiaodian_pos[2]]
    r_1=np.linalg.norm(pos_2-zhongji_gl_2)
    r_2=np.linalg.norm(zhongji_gl_2-mubiaodian_pos_2)
    zhongji_lc=global_to_local(pos,obs['agent_state']['rotation'],zhongji_gl)
    theta_1=math.atan2(zhongji_lc[0], -zhongji_lc[2])
    zhongji_rot=rotate_around_y_quaternion(obs['agent_state']['rotation'],-theta_1)
    mubiaodian_lc=global_to_local(zhongji_gl,zhongji_rot,mubiaodian_pos)
    theta_2=math.atan2(mubiaodian_lc[0],-mubiaodian_lc[2])
    return (r_1,theta_1),(r_2,theta_2)

        



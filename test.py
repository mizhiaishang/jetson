import sys
import quaternion
from PIL import Image
sys.path.append("/home/nvidia/mf/test_file/datanavi")
from testnavi import *
from car_control import *
from util import *
from dongtai import *
obs={}
agent_state={}
pipeline,config,hole_filling_filter,temporal_filter=set_config()
pipeline.start(config)
start=time.time()
end=time.time()
time.sleep(4)
count=0
car = CarController(port='/dev/ttyUSB0')
rgb_image, dep_image =image_type(pipeline,hole_filling_filter,temporal_filter)
model, device = init_model(TRAINED_WEIGHTS, DEVICE)
while True:
    if time.time()-start>1:
        start=time.time()
        img,mask_ini=process_image(model,car,device,rgb_image, dep_image)
        img4 = Image.fromarray(img, mode='RGB')
        img4.save(f'/home/wheeltec/test_file/test1_{count}.png')
        # 创建新矩阵，保留原数据
        converted_matrix = mask_ini.copy()
        converted_matrix = (converted_matrix == 255)
        mask = converted_matrix.copy()
        if car.connect():
            car.initialize_car()
            pos, rot = car.get_agent_state()
            success = car.move(0)
        resolution=(720,1280)
        obs=obs_solve(pos,rot,rgb_image,dep_image)
        hov=91
        vov=65
        a_final,pos_ex=navigability(obs,mask,resolution,hov,vov)
        img,a_final_projected=projection(a_final,obs,hov,vov,resolution,pos_ex)
        for obj in a_final_projected:
            print(obj)
        img4 = Image.fromarray(img, mode='RGB')
        img4.save(f'/home/wheeltec/test_file/test2_{count}.png')
        count=count+1
        if time.time()-end>30:
            break
print('finish')
# pos=obs['agent_state']['position']
# rot=obs['agent_state']['rotation']
# obt,zhangaiweizhi,zhangaidaxiao=get_obt_position(rgb_image,dep_image,pos,rot)
# mubiaodian_pos=[0,0,-3]
# state=0
# if obt:
#     rgb_image, dep_image = image_type(pipeline,hole_filling_filter,temporal_filter)
#     pos, rot = car.get_agent_state()
#     img,mask_ini=process_image(model,car,device,rgb_image, dep_image)
#     converted_matrix = mask_ini.copy()
#     converted_matrix = (converted_matrix == 255)
#     mask = converted_matrix.copy()
#     obs=obs_solve(pos,rot,rgb_image,dep_image)
#     action_2_1,action_2_2=action_make(zhangaiweizhi,zhangaidaxiao,pos,rot,dep_image,mubiaodian_pos,resolution,hov,vov,obs,mask)
#     state=1
# if state!=1:
#     r_1=np.linalg.norm(pos-mubiaodian_pos)
#     zhongji_lc=global_to_local(pos,rot,mubiaodian_pos)
#     theta_1=math.atan2(zhongji_lc[0], zhongji_lc[1])
#     if zhongji_lc[0]<0:
#         theta_1=2*np.pi-theta_1
#     success=car.rotate(theta_1/np.pi*180)
#     success = car.move(r_1)
# else:
#     success=car.rotate(-action_2_1[1]/np.pi*180)
#     success = car.move(action_2_1[0])
#     success=car.rotate(-action_2_2[1]/np.pi*180)
#     success = car.move(action_2_2[0])

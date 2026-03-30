from car_control import *
from guo_doc import *

def get_inf():
    rgb,dep=main()
    rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    obs={}
    obs['color_sensor']=rgb
    obs['depth_sensor']=dep
    r1=process_image(rgb)
    return rgb,dep,obs,r1

car = CarController(port='/dev/ttyUSB0')
rgb,dep=car.get_images()
if car.connect():
        car.initialize_car()
        time.sleep(1)
        pos1, rot1 = car.get_agent_state()
        rgb,dep,obs,r1=get_inf()
        image_information(r1,obs,car)
        success=car.rotate(-90)
        time.sleep(1)
        rgb,dep,obs,r1=get_inf()
        image_information(r1,obs,car)
        success=car.rotate(-90)
        success = car.move(3)
        time.sleep(1)       
        success=car.rotate(180)
        time.sleep(1)
        success=car.move(3)
        time.sleep(1)
        rgb,dep,obs,r1=get_inf()
        image_information(r1,obs,car)
        success=car.rotate(-90)
        time.sleep(1)
        rgb,dep,obs,r1=get_inf()
        image_information(r1,obs,car)
        print('finish')

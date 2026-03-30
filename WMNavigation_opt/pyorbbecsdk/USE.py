from camera import OrbbecCamera

def main():
    camera = OrbbecCamera()
    camera.initialize()
    
    try:
        while True:
            cmd = input("输入's'保存当前帧，'q'退出: ")
            if cmd == 's':
                if camera.save_current_frame():
                    print("帧数据保存成功")
                else:
                    print("无有效帧数据")
            elif cmd == 'q':
                break
    finally:
        camera.shutdown()

if __name__ == "__main__":
    main()

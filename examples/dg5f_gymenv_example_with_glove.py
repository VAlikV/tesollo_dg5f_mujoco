import numpy as np
from gymlike_env.env_dg import HandEnv
import matplotlib.pyplot as plt
import cv2
import time
import serial

glove = serial.Serial("/dev/ttyUSB0", baudrate=115200)

env = HandEnv(xml_path="robot/scene_dg.xml",
            sim_timestep = 0.001,
            control_hz = 20.0,
            mode = "fast",   # "realtime" | "fast"
            max_episode_steps = 1000,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

start_pos = obs["state"]["joint_pos"]

print(start_pos)

digits = start_pos

t = time.time()

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for _ in range(1001):

    a = glove.readline().decode("utf-8")
    try: 
        digits = np.array(list(map(float, a[1:-4].split(',')))[1:])
        digits[0] = 1*(digits[0]+30)
        digits[1] = -1*(digits[1] - 78)
        digits[2] = -1*(digits[2] - 17)
        digits[3] = -1*(digits[3] - 17)

        digits[5] *= -1
        digits[6] *= -1
        digits[7] *= -1

        digits[9] *= -1
        digits[10] *= -1
        digits[11] *= -1

        digits[12] *= 1
        digits[13] *= -1
        digits[14] *= -1
        digits[15] *= -1

        # digits[16] = -1*(digits[16]+15)
        digits[18] *= -1
        digits[19] *= -1

        # print(digits)
        # print(len(digits))
    except:
        print("SOS")

    obs, reward, terminated, truncated, info = env.step(digits*np.pi/180)

    imgs = obs["images"]

    # for ax, (name, img) in zip(axes, imgs.items()):
    #     ax.clear()
    #     ax.imshow(img)
    #     ax.set_title(name)
    #     ax.axis("off")

    # plt.pause(0.001)

    print("JOINTS:",obs["state"]["joint_pos"])
    print()

    if terminated or truncated:
        print("Episode ended:", terminated, truncated, info)
        obs, info = env.reset()

        print("Время:", time.time() - t)

    # if _ % 100 == 0:
    #     obs, info = env.reset()

env.close()

# plt.ioff()
# plt.show()
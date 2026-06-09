import numpy as np
from gymlike_env.env_dg import HandEnv
import matplotlib.pyplot as plt
import cv2
import time
import serial
import threading

glove = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=0.1)

env = HandEnv(xml_path="robot/scene_dg.xml",
            sim_timestep = 0.001,
            control_hz = 20.0,
            mode = "realtime",   # "realtime" | "fast"
            max_episode_steps = 1000,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

start_pos = obs["state"]["joint_pos"]

print(start_pos)

digits = start_pos

t = time.time()
stop_glove_reader = threading.Event()
digits_lock = threading.Lock()


def read_glove():
    global digits
    while not stop_glove_reader.is_set():
        a = glove.readline().decode("utf-8", errors="ignore")
        if not a:
            continue
        try:
            new_digits = np.array(list(map(float, a[1:-4].split(',')))[1:])
            new_digits[0] = 1*(new_digits[0]+30)
            new_digits[1] = -1*(new_digits[1] - 78)
            new_digits[2] = -1*(new_digits[2] - 17)
            new_digits[3] = -1*(new_digits[3] - 17)

            new_digits[5] *= -1
            new_digits[6] *= -1
            new_digits[7] *= -1

            new_digits[9] *= -1
            new_digits[10] *= -1
            new_digits[11] *= -1

            new_digits[12] *= 1
            new_digits[13] *= -1
            new_digits[14] *= -1
            new_digits[15] *= -1

            new_digits[16] *= 0
            new_digits[18] *= -1
            new_digits[19] *= -1

            with digits_lock:
                digits = new_digits
        except:
            print("SOS")


glove_thread = threading.Thread(target=read_glove, daemon=True)
glove_thread.start()

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for _ in range(1001):

    with digits_lock:
        current_digits = digits.copy()

    obs, reward, terminated, truncated, info = env.step(current_digits*np.pi/180)

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

stop_glove_reader.set()
glove_thread.join(timeout=1.0)
glove.close()
env.close()

# plt.ioff()
# plt.show()

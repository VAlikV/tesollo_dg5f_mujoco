import numpy as np
from gymlike_env.env_dg import HandEnv
import matplotlib.pyplot as plt
import cv2
import time
import serial
import threading
import csv
import queue


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
start_vel = obs["state"]["joint_vel"]

print(start_pos)

digits = start_pos
digits_time = 0.0

t = time.time()
stop_glove_reader = threading.Event()
digits_lock = threading.Lock()

que = queue.Queue(10)
prev_commanded_time = None
prev_commanded_pos = None

def read_glove():
    global digits, digits_time
    while not stop_glove_reader.is_set():
        a = glove.readline().decode("utf-8", errors="ignore")
        if not a:
            continue
        arrival_time = time.time() - t
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
                digits_time = arrival_time
        except:
            print("SOS")


glove_thread = threading.Thread(target=read_glove, daemon=True)
glove_thread.start()

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

log_file = open("joint_positions_log.csv", "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(
    ["step", "time", "glove_time", "control_time", "control_delay"]
    + [f"current_pos_{i}" for i in range(len(start_pos))]
    + [f"commanded_pos_{i}" for i in range(len(start_pos))]
    + [f"current_vel_{i}" for i in range(len(start_vel))]
    + [f"commanded_vel_{i}" for i in range(len(start_pos))]
)

try:
    for step in range(1001):

        with digits_lock:
            current_digits = digits.copy()
            current_digits_time = digits_time
            que.put((current_digits_time, current_digits))

        if que.qsize() > 0:
            commanded_time, delayed_digits = que.get()
            commanded_pos = delayed_digits*np.pi/180
            if prev_commanded_time is None or commanded_time <= prev_commanded_time:
                commanded_vel = np.zeros_like(commanded_pos)
            else:
                commanded_vel = (commanded_pos - prev_commanded_pos) / (commanded_time - prev_commanded_time)
            prev_commanded_time = commanded_time
            prev_commanded_pos = commanded_pos.copy()

            control_time = time.time() - t
            obs, reward, terminated, truncated, info = env.step(commanded_pos)

            imgs = obs["images"]
            current_pos = obs["state"]["joint_pos"]
            current_vel = obs["state"]["joint_vel"]

            log_writer.writerow(
                [
                    step,
                    commanded_time,
                    commanded_time,
                    control_time,
                    control_time - commanded_time,
                    *current_pos,
                    *commanded_pos,
                    *current_vel,
                    *commanded_vel,
                ]
            )

            # for ax, (name, img) in zip(axes, imgs.items()):
            #     ax.clear()
            #     ax.imshow(img)
            #     ax.set_title(name)
            #     ax.axis("off")

            # plt.pause(0.001)

            print("JOINTS:", current_pos)
            print()

            if terminated or truncated:
                print("Episode ended:", terminated, truncated, info)
                obs, info = env.reset()

                print("Время:", time.time() - t)

        # if step % 100 == 0:
        #     obs, info = env.reset()
finally:
    log_file.close()
    stop_glove_reader.set()
    glove_thread.join(timeout=1.0)
    glove.close()
    env.close()

# plt.ioff()
# plt.show()

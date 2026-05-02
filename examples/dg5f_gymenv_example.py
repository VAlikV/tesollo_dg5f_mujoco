import numpy as np
from gymlike_env.env_dg import HandEnv
import matplotlib.pyplot as plt
import cv2
import time

env = HandEnv(xml_path="robot/scene_dg.xml",
            sim_timestep = 0.001,
            control_hz = 20.0,
            mode = "realtime",   # "realtime" | "fast"
            max_episode_steps = 1000,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

start_pos = obs["state"]["joint_pos"]

t = time.time()

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for _ in range(1001):
    # start_pos[0] -= s

    obs, reward, terminated, truncated, info = env.step(start_pos)

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
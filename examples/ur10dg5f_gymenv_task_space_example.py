import numpy as np
from gymlike_env.env_dg_ur import AssemblingEnv
import matplotlib.pyplot as plt
import cv2
import time

env = AssemblingEnv(xml_path="robot/scene.xml",
            sim_timestep = 0.001,
            control_hz = 20.0,
            mode = "fast",   # "realtime" | "fast"
            max_episode_steps = 1000,
            use_task_space=True,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

start_pos = np.concatenate([obs["state"]["ee_pos"],obs["state"]["ee_quat"], [0]*20])
start_pos[0] = 0.1
start_pos[1] = -0.5
start_pos[2] = 0.45

t = time.time()

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for _ in range(1001):
    s = np.sin(_/(10*np.pi))/50
    # start_pos[0] -= s

    obs, reward, terminated, truncated, info = env.step(start_pos)

    imgs = obs["images"]

    # for ax, (name, img) in zip(axes, imgs.items()):
    #     ax.clear()
    #     ax.imshow(img)
    #     ax.set_title(name)
    #     ax.axis("off")

    # plt.pause(0.001)

    print("POS:", obs["state"]["joint_vel"])
    print("POS:", obs["state"]["ee_pos"])
    print("QUAT:",obs["state"]["ee_quat"]) 
    print("LIN_VEL:",obs["state"]["ee_lin_vel"])
    print("ANG_VEL:",obs["state"]["ee_ang_vel"])
    print("JOINTS:",obs["state"]["joint_pos"])
    print("OBJECTS:")
    for k in obs["objectsSSS"].keys():
        print(k, obs["objects"][k]["lin_vel"])
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
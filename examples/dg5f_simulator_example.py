import numpy as np
import os
from control_simulator import Simulator
from pathlib import Path
from typing import Dict
# import pinocchio as pin
import matplotlib.pyplot as plt
import serial

name = "dg5f_test"

glove = serial.Serial("/dev/ttyUSB0", baudrate=115200)

def joint_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:


    a = glove.readline().decode("utf-8")
    
    try: 
        digits = list(map(float, a[1:-4].split(',')))

        angles = digits[1:]
        
        angles[0] = 1*(angles[0]+30)
        angles[1] = -1*(angles[1] - 78)
        angles[2] = -1*(angles[2] - 17)
        angles[3] = -1*(angles[3] - 17)

        angles[5] *= -1
        angles[6] *= -1
        angles[7] *= -1

        angles[9] *= -1
        angles[10] *= -1
        angles[11] *= -1

        angles[12] *= -1
        angles[13] *= -1
        angles[14] *= -1
        angles[15] *= -1

        # digits[16] = -1*(digits[16]+15)
        angles[18] *= -1
        angles[19] *= -1

        tau = np.array(angles, dtype=float) * np.pi / 180
    except:
        tau = np.array([0.0,]*20, dtype=float)

    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning controller...")
    sim = Simulator(
        xml_path="robot/scene_dg.xml",
        enable_task_space=False,
        show_viewer=True,
        record_video=False,
        # video_path="logs/videos/" + name + ".mp4",
        fps=30,
        width=640,
        height=480
    )

    sim.set_controller(joint_controller)
    sim.run(time_limit=10.0)

if __name__ == "__main__":
    main() 
    
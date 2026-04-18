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
        digits = list(map(float, a[1:-3].split(',')))
        
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

        digits[12] *= -1
        digits[13] *= -1
        digits[14] *= -1
        digits[15] *= -1

        # digits[16] = -1*(digits[16]+15)
        digits[18] *= -1
        digits[19] *= -1

        tau = np.array(digits, dtype=float) * np.pi / 180
    except:
        tau = np.array([0.0,]*20, dtype=float)

    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning controller...")
    sim = Simulator(
        xml_path="robot/scene.xml",
        enable_task_space=False,
        show_viewer=True,
        record_video=False,
        # video_path="logs/videos/" + name + ".mp4",
        fps=30,
        width=1920,
        height=1080
    )

    sim.set_controller(joint_controller)
    sim.run(time_limit=10.0)

if __name__ == "__main__":
    main() 
    
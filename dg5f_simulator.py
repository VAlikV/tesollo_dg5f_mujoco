import numpy as np
import os
from simulator import Simulator
from pathlib import Path
from typing import Dict
# import pinocchio as pin
import matplotlib.pyplot as plt

name = "dg5f_test"

def joint_controller(q: np.ndarray, dq: np.ndarray, t: float) -> np.ndarray:

    tau = np.array([0.2,]*20, dtype=float)

    return tau

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning controller...")
    sim = Simulator(
        xml_path="robot/scene.xml",
        enable_task_space=False,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/" + name + ".mp4",
        fps=30,
        width=1920,
        height=1080
    )

    sim.set_controller(joint_controller)
    sim.run(time_limit=10.0)

if __name__ == "__main__":
    main() 
    
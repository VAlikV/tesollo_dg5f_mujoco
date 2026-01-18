import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Callable, Optional, Dict, Union, List, Any
from pathlib import Path
import mediapy as media
import signal
import sys

mujoco.viewer.launch()

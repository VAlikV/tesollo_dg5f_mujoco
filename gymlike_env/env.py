import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import time
import pinocchio as pin
from importlib.resources import as_file, files

class PinKinematics:
    def __init__(self, model_path, ee_name="ee_link", max_it=1000, eps=1e-4):
        self.model = pin.buildModelFromMJCF(model_path)
        self.data = self.model.createData()

        self.ee_frame_id = self.model.getFrameId(ee_name)

        self.max_it = max_it
        self.eps = eps
        self.dt = 0.02
        self.damp = 1e-4

    def solve_ik(self, target_pos, target_orient, current_joint):
        '''
        target_pos - np.array(x, y, z) в метрах
        target_orient - np.array(w, x, y, z) кватернион
        current_joint - np.array(q1 ... q6) углы в радианах
        '''
        target_position = pin.SE3(target_orient, target_pos)
        q = current_joint
        i = 0
        while True:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            
            oMf = self.data.oMf[self.ee_frame_id]
            
            iMd = oMf.actInv(target_position)
            err = pin.log(iMd).vector  
            
            if np.linalg.norm(err) < self.eps:
                success = True
                break
            if i >= self.max_it:
                success = False
                break
                
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.LOCAL)
            
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + self.damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * self.dt)
            
            # if not i % 10:
            #     print(f"{i}: error = {err.T}")
            i += 1

        return success, q
    
    def solve_fk(self, current_joint):
        pin.forwardKinematics(self.model, self.data, current_joint)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        oMf = self.data.oMf[self.ee_frame_id]

        return oMf.translation, oMf.rotation

# ===================================================================================================
# ===================================================================================================
# ===================================================================================================

class AssemblingEnv(gym.Env):

    ee_name = "rl_dg_mount"

    joints_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", 
        "rj_dg_1_1", "rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4", 
        "rj_dg_2_1", "rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4", 
        "rj_dg_3_1", "rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4", 
        "rj_dg_4_1", "rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4", 
        "rj_dg_5_1", "rj_dg_5_2", "rj_dg_5_3", "rj_dg_5_4"
    ]

    actuators_names = [
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", 
        "rj_dg_1_1_ctrl", "rj_dg_1_2_ctrl", "rj_dg_1_3_ctrl", "rj_dg_1_4_ctrl", 
        "rj_dg_2_1_ctrl", "rj_dg_2_2_ctrl", "rj_dg_2_3_ctrl", "rj_dg_2_4_ctrl", 
        "rj_dg_3_1_ctrl", "rj_dg_3_2_ctrl", "rj_dg_3_3_ctrl", "rj_dg_3_4_ctrl", 
        "rj_dg_4_1_ctrl", "rj_dg_4_2_ctrl", "rj_dg_4_3_ctrl", "rj_dg_4_4_ctrl", 
        "rj_dg_5_1_ctrl", "rj_dg_5_2_ctrl", "rj_dg_5_3_ctrl", "rj_dg_5_4_ctrl"
    ]

    camera_names = ["cam_front", "cam_side"]

    objects_names = ["bottom", "mid", "cap"]
    objects_joints = ["bottom_joint", "mid_joint", "cap_joint"]

    initial_pose = [1.42010733, -1.74898752,  2.36328641, -2.1743184, -1.57146472, -0.14989266, 
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0]

    def __init__(
                self,
                xml_path: str,
                sim_timestep: float = 0.001,
                control_hz: float = 10.0,
                mode: str = "realtime",   # "realtime" | "fast"
                max_episode_steps: int = 1000,
                use_task_space: bool = False,
                render_mode=None,           # "human" | "rgb_array" | "all"
    ):
        super().__init__()

        assert mode in ["realtime", "fast"]
        self.mode = mode
        self.realtime = (mode == "realtime")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # ===================== misc =====================
        self.use_task_space = use_task_space
        self.step_count = 0
        self.max_episode_steps = max_episode_steps

        # ===================== timing =====================
        self._setup_idx()
        self._setup_spaces()

        # ===================== timing =====================
        self.model.opt.timestep = sim_timestep
        self.sim_timestep = self.model.opt.timestep

        self.control_hz = control_hz
        self.control_dt = 1.0 / self.control_hz

        # согласование control_hz и physics
        self.frame_skip = max(1, int(round(self.control_dt / self.sim_timestep)))
        self.sim_dt = self.frame_skip * self.sim_timestep

        self.next_step_time = None

        # ===================== rendering =====================
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

        if self.render_mode == "all" or self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model)

        # ===================== kinematics ===============

        self.kinematics = PinKinematics(model_path="robot/ur10edg5f.xml",ee_name=self.ee_name)

    # ======================================================================

    def _setup_idx(self):

        # ---------------------- Для reset и наблюдений

        self.joints_qpos_idx = []
        self.joints_qvel_idx = []
        for name in self.joints_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]

            self.joints_qpos_idx.append(qpos_adr)
            self.joints_qvel_idx.append(qvel_adr)

        self.joints_qpos_idx = np.array(self.joints_qpos_idx)
        self.joints_qvel_idx = np.array(self.joints_qvel_idx)

        # ---------------------- Для задания действия в джоинтах

        self.actuator_idx = []
        for name in self.actuators_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.actuator_idx.append(actuator_id)

        self.actuator_idx = np.array(self.actuator_idx)

        # ---------------------- Для положения и скорости объектов в наблюдениях 

        self.objects_idx = []
        for name in self.objects_names:
            object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.objects_idx.append(object_id)

        self.objects_idx = np.array(self.objects_idx)

        # ---------------------- Для reset объектов

        self.objects_qpos_adr = []
        for name in self.objects_joints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            self.objects_qpos_adr.append(qpos_adr)

        self.objects_qpos_adr = np.array(self.objects_qpos_adr)

        # ---------------------- Для положения и скорости в наблюдениях
        
        self.ee_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_name)

    # ======================================================================

    def _setup_spaces(self):

        if self.use_task_space:
            self.action_dim = 27

            self.action_space = spaces.Box(
                low= np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0]+20*[-10.0], dtype=np.float32),
                high=np.array([ 10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0]+20*[ 10.0], dtype=np.float32),
                shape=(27,),
                dtype=np.float32
            )
        
        else:
            self.action_dim = self.actuator_idx.shape[0]

            self.action_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(self.action_dim,),
                dtype=np.float32
            )

        # ----------------------
        n_joints = len(self.joints_qpos_idx)

        state_space = spaces.Dict({
            "joint_pos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_joints,), dtype=np.float32
            ),
            "joint_vel": spaces.Box(
                low=-np.inf, high=np.inf, shape=(n_joints,), dtype=np.float32
            ),
            "ee_pos": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "ee_quat": spaces.Box(
                low=-1, high=1, shape=(4,), dtype=np.float32
            ),
            "ee_lin_vel": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            "ee_ang_vel": spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
        })

        # ----------------------
        objects_space = {}

        for obj in self.objects_names:
            objects_space[obj] = spaces.Dict({
                obj: spaces.Dict({
                    "pos": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                    ),
                    "quat": spaces.Box(
                        low=-1, high=1, shape=(4,), dtype=np.float32
                    ),
                    "lin_vel": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                    ),
                    "ang_vel": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                    ),
                })
            })

        objects_space = spaces.Dict(objects_space)

        # ----------------------
        image_space = {}

        if (self.render_mode == "rgb_array") or (self.render_mode == "all"):
            # лучше взять из renderer
            H = self.renderer.height
            W = self.renderer.width

            for cam in self.camera_names:
                image_space[cam] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(H, W, 3),
                    dtype=np.uint8
                )

        image_space = spaces.Dict(image_space)

        # ----------------------
        self.observation_space = spaces.Dict({
            "state": state_space,
            "objects": objects_space,
            "images": image_space,
        })

    # ======================================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        for i, id in enumerate(self.joints_qpos_idx):
            self.data.qpos[id] = self.initial_pose[i]
            self.data.qvel[id] = 0.0

        for i, id in enumerate(self.objects_qpos_adr):
            pos = self.data.qpos[id:id+3]
            quat = self.data.qpos[id+3:id+7]

            dx = np.random.uniform(-0.02, 0.02)
            dy = np.random.uniform(-0.02, 0.02)

            new_pos = pos.copy()
            new_pos[0] += dx
            new_pos[1] += dy

            yaw = np.random.uniform(-np.pi/4, np.pi/4)
            q_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])

            new_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(new_quat, q_yaw, quat)

            self.data.qpos[id:id+3] = new_pos
            self.data.qpos[id+3:id+7] = new_quat

        mujoco.mj_forward(self.model, self.data)

        if ((self.render_mode == "human") or (self.render_mode == "all")) and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            cam = self.viewer.cam
            cam.lookat[:] = [-0.430, -0.367, -0.075]
            cam.distance = 1
            cam.azimuth = 145
            cam.elevation = -24

        self.step_count = 0

        if self.realtime:
            self.next_step_time = time.perf_counter() + self.control_dt
        else:
            self.next_step_time = None

        return self._get_obs(), {}

    # ======================================================================

    def step(self, action):
        '''
            action - целевые углы в rad + положение гриппера, если use_task_space = True
            action - целевое положение в метрах (x, y, z) + кватернион (w, x, y, z) + положение гриппера, если use_task_space = False
        '''
        
        self._apply_action(action)

        # physics step
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # viewer update
        if ((self.render_mode == "human") or (self.render_mode == "all")) and self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()

        obs = self._get_obs()
        
        reward = 0.0
        terminated = False
        truncated = False

        if self.max_episode_steps > 0:
            self.step_count += 1
            truncated = self.step_count >= self.max_episode_steps

        # ===================== realtime pacing =====================
        if self.realtime:
            now = time.perf_counter()
            lag = now - self.next_step_time

            if lag < 0.0:
                time.sleep(-lag)
                self.next_step_time += self.control_dt
            else:
                if lag > self.control_dt:
                    print(f"Accumulated lag: {lag:.3f}s")

                if lag > 5.0 * self.control_dt:
                    self.next_step_time = now + self.control_dt
                else:
                    self.next_step_time += self.control_dt

        return obs, reward, terminated, truncated, {}

    # ======================================================================

    def _apply_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.use_task_space:
            pos = action[0:3]
            quat = action[3:7]
            gripper = action[7]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, quat)
            R = R.reshape(3, 3)

            # Исправление поворота осей
            R_fix = np.diag([-1, -1, 1])
            R_corrected = R @ R_fix
            pos_corrected = pos @ R_fix

            success, all_action = self.kinematics.solve_ik(pos_corrected, R_corrected, self.data.qpos[self.joints_qpos_idx])

            if success:
                action = all_action[self.actuator_idx]
            else:
                action = self.data.qpos[self.joints_qpos_idx].copy()

        self.gripper_action = action[-1]
        self.data.ctrl[:] = action

    # ======================================================================

    def _get_obs(self):

        obs = {"state":{}, "images": {}, "objects":{}}

        obs["state"]["joint_pos"] = self.data.qpos[self.joints_qpos_idx]
        obs["state"]["joint_vel"] = self.data.qvel[self.joints_qvel_idx]

        ee_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.ee_idx,
            ee_vel,
            0
        )

        obs["state"]["ee_pos"] = self.data.xpos[self.ee_idx].copy()
        obs["state"]["ee_quat"] = self.data.xquat[self.ee_idx].copy()
        obs["state"]["ee_lin_vel"] = ee_vel[3:].copy()
        obs["state"]["ee_ang_vel"] = ee_vel[:3].copy()

        for i, obj in enumerate(self.objects_names):
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                self.objects_idx[i],
                vel,
                0
            )
            obs["objects"][obj] = {}
            obs["objects"][obj] = {}
            obs["objects"][obj]["pos"] = self.data.xpos[self.objects_idx[i]].copy()
            obs["objects"][obj]["quat"] = self.data.xquat[self.objects_idx[i]].copy()
            obs["objects"][obj]["lin_vel"] = vel[3:].copy()
            obs["objects"][obj]["ang_vel"] = vel[:3].copy()

        images = self.render()
        if images is not None:
            for cam, img in images.items():
                obs["images"][cam] = img

        return obs

    # ======================================================================

    def render_cameras(self):
        if self.renderer is None:
            raise RuntimeError("render_cameras requires render_mode='rgb_array'.")

        images = {}
        for cam_name in self.camera_names:
            self.renderer.update_scene(self.data, camera=cam_name)
            images[cam_name] = self.renderer.render().copy()
        return images

    # ======================================================================

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_cameras()

        elif self.render_mode == "all":
            return self.render_cameras()

        return None

    # ======================================================================

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
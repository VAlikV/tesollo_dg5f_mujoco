import mujoco

model = mujoco.MjModel.from_xml_path("robot/urdf/dg5f_right.urdf")
mujoco.mj_saveLastXML("robot_converted.xml", model)
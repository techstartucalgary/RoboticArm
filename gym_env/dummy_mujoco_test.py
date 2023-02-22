# %%
import mujoco
# %%
xml_file = "./mujoco_models/kenova_gen3_robotiq3f85/Kenova_2f85_pick_place.xml"
xml_file_arm = "../mujoco_models/kenova_gen3_robotiq3f85/Gen3Robotiq_2f85.xml"
model = mujoco.MjModel.from_xml_path(xml_file)

names = ['Base_Link', 'Bracelet_Link', 'ForeArm_Link', 'HalfArm1_Link', 'HalfArm2_Link', 'Shoulder_Link', 'SphericalWrist1_Link', 'SphericalWrist2_Link', 'base', 'base_mount', 'ee_link', 'floor0', 'left_coupler', 'left_driver', 'left_follower', 'left_pad', 'left_silicone_pad', 'left_spring_link', 'object0', 'right_coupler', 'right_driver', 'right_follower', 'right_pad', 'right_silicone_pad', 'right_spring_link', 'world']
print(len(names))
# s = model.geom('floor0').size
# pos = model.geom('floor0').pos
# objectxpos = model.body('object0').pos

data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)

# %%
# print(model.body('Base_Link'))
# print(model.body('world'))
# print(model.body('ee_link'))
# print(data.body('left_spring_link').xpos)
# print(data.names)
print(len(data.qpos))
print(len(data.qvel))
# print(data.xpos)
print(data.xpos.shape)
# print(data.ypos)
print(data.qpos)
print(data.qvel)

# %%
# joints = [data.joint(i)  for i in range(model.nv)]

# %%
# print(data.body('left_follower'))
data.site('target0').xpos = [0.8, 0,  0.2]
print(data.site('target0'))
print(data.ctrl)
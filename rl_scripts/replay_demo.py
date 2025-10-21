"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import sys
import time
import glob
import numpy as np
from os import path
from mplib import Pose as mpPose
sys.path.append(path.join(path.dirname(__file__), '..'))
from rl_sim.kinematics.manipulator import ManipulatorRobot
from rl_sim.kinematics.mano_robot_hand import MANORobotHand


def replay(f_pkl):
    # load data
    print(f_pkl)
    all_data = np.load(f_pkl, allow_pickle=True)
    meta_data = all_data["meta_data"]
    data = all_data["data"]

    # sim env
    env_class = meta_data["env_class"].lower()
    if "relocate" in env_class:
        from rl_sim.env.relocate_env import RelocateEnv
        env = RelocateEnv(**meta_data["env_kwargs"])
    elif "door" in env_class:
        from rl_sim.env.tabledoor_env import TableDoorEnv
        env = TableDoorEnv(**meta_data["env_kwargs"])
    elif "flip" in env_class:
        from rl_sim.env.mugflip_env import MugFlipEnv
        env = MugFlipEnv(**meta_data["env_kwargs"])
    else:
        raise ValueError(env_class)

    # robot
    if meta_data["robot_name"] == 'customized':
        robot = MANORobotHand(env.scene, ik_link='eef_calib_link',
                              init_joint_pos=meta_data["zero_joint_pos"])
    else:
        robot = ManipulatorRobot(env.scene, meta_data['robot_name'], ik_link='eef_calib_link')

    for i in range(meta_data["data_len"]):
        # load entity poses
        for j in range(len(data[i]['simulation']['entity']['name'])):
            ent_name = data[i]['simulation']['entity']['name'][j]
            ent_pose = data[i]['simulation']['entity']['pose'][j]
            for ent in env.scene.entities:
                if ent.get_name() == ent_name:
                    ent.set_pose(ent_pose)
            if ent_name == 'eef_calib_link':
                eef_calib_pose = ent_pose
        # load articulation joints
        for j in range(len(data[i]['simulation']['articulation']['name'])):
            art_name = data[i]['simulation']['articulation']['name'][j]
            art_qpos = data[i]['simulation']['articulation']['qpos'][j]
            if meta_data["robot_name"] in art_name:
                _, result = robot.planner.IK(goal_pose=mpPose(eef_calib_pose), return_closest=True,
                                             start_qpos=robot.robot.get_qpos()[:6])
                art_qpos[:6] = result
                robot.robot.set_qpos(art_qpos)
            else:
                for art in env.scene.get_all_articulations():
                    if art.get_name() == art_name:
                        art.set_qpos(art_qpos)

        env.render()
        time.sleep(0.1)
    env.viewer.close()


if __name__ == '__main__':
    collection_path = path.join(path.dirname(__file__), 'teleop_collection', sys.argv[1], '*.pkl')
    for pkl_path in sorted(glob.glob(collection_path)):
        replay(pkl_path)

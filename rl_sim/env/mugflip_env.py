"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
Based on code from https://github.com/yzqin/dex-hand-teleop with MIT licence
"""
import sys
import numpy as np
from os import path
import transforms3d.euler
import sapien.core as sapien
sys.path.append(path.join(path.dirname(__file__), '..'))
from env.base_env import BaseEnv
from utils.ycb_object_utils import load_ycb_object, YCB_SIZE


class MugFlipEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        # init variable
        self.current_step = 0
        self.palm_init = np.array([-0.1, 0, 0.2, 0, 0, 0])

        # Construct scene
        self.table = self.create_table()
        self.manipulated_object = load_ycb_object(self.scene, 'mug')
        self.original_object_pos = np.zeros(3)

    def reset_env(self):
        self.current_step = 0

        # object pose
        pose = self.generate_random_init_pose()
        self.manipulated_object.set_pose(pose)
        rigid_component = self.manipulated_object.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        rigid_component.set_angular_velocity(np.zeros(3))
        rigid_component.set_linear_velocity(np.zeros(3))
        self.original_object_pos = pose.p

    def generate_random_init_pose(self, randomness_scale=1):
        pos = self.np_random.uniform(low=-0.1, high=0.1, size=2) * randomness_scale
        ycb_height = max(YCB_SIZE['mug'][1])
        random_z_rotate = self.np_random.uniform(np.pi * 0.4, np.pi * 0.6)
        orientation = transforms3d.euler.euler2quat(-np.pi / 2, 0, random_z_rotate)
        position = np.array([pos[0], pos[1], ycb_height])
        pose = sapien.Pose(position, orientation)
        return pose

    def is_success(self, th_ang=0.9, th_dist=0.08):
        obj_pose = self.manipulated_object.get_pose()
        z_axis = obj_pose.to_transformation_matrix()[:3, 2]
        ang_cos = np.sum(np.array([0, 0, 1]) * z_axis)
        obj_tgt_dist = np.linalg.norm(self.original_object_pos[:2] - obj_pose.p[:2])
        return (ang_cos > th_ang) and (obj_tgt_dist < th_dist)

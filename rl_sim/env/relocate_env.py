"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
Based on code from https://github.com/yzqin/dex-hand-teleop with MIT licence
"""
import sys
import numpy as np
from os import path
import transforms3d
import sapien.core as sapien
sys.path.append(path.join(path.dirname(__file__), '..'))
from env.base_env import BaseEnv
from utils.ycb_object_utils import load_ycb_object, YCB_SIZE


class RelocateEnv(BaseEnv):
    def __init__(self, object_name='tomato_soup_can'):
        super().__init__()
        # init variable
        self.current_step = 0
        self.palm_init = np.array([-0.1, 0, 0.2, 0, 0, 0])
        self.object_name = object_name
        self.object_height = YCB_SIZE[self.object_name][2][0]

        # Construct scene
        self.tables = self.create_table()
        self.manipulated_object = load_ycb_object(self.scene, object_name)
        self.target_object = load_ycb_object(self.scene, object_name, visual_only=True)

    def generate_random_object_pose(self, height, randomness_scale=1.0):
        pos = self.np_random.uniform(low=-0.1, high=0.1, size=2) * randomness_scale
        pose = sapien.Pose([pos[0], pos[1], height])
        return pose

    def reset_env(self):
        self.current_step = 0

        # object pose
        pose = self.generate_random_object_pose(self.object_height)
        self.manipulated_object.set_pose(pose)
        rigid_component = self.manipulated_object.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        rigid_component.set_angular_velocity(np.zeros(3))
        rigid_component.set_linear_velocity(np.zeros(3))

        # Target pose
        pose = self.generate_random_object_pose(0.15 + self.object_height, 1.5)
        self.target_object.set_pose(pose)

    def is_success(self, th_dist=0.1):
        object_pose = self.manipulated_object.get_pose()
        return np.linalg.norm(object_pose.p - self.target_object.get_pose().p) < th_dist

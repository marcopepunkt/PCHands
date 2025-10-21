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
sys.path.append(path.join(path.dirname(__file__), '.'))
from base_env import BaseEnv


class TableDoorEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        # init variable
        self.current_step = 0
        self.palm_init = np.array([-0.1, 0, 0.2, 0, 0, 0])

        # Construct scene
        self.table = self.create_table()
        self.table_door = self.create_door()
        self.table_door.set_pose(sapien.Pose([0, 0, 0.01]))
        self.table_door.set_qpos(np.zeros(self.table_door.dof))

    def reset_env(self):
        self.current_step = 0
        self.table_door.set_qpos(np.zeros(self.table_door.dof))
        rand_xy = self.np_random.uniform(low=-0.05, high=0.05, size=2)
        self.table_door.set_pose(sapien.Pose([rand_xy[0], rand_xy[1], 0.01]))
        self.table_door.set_qvel([0, 0])

    def is_success(self, th_ang=60.):
        return self.table_door.get_qpos()[0] > np.deg2rad(th_ang)

    def create_door(self):
        builder = self.scene.create_articulation_builder()
        # frame
        root = builder.create_link_builder()
        root.set_name('frame')
        table_physics_mat = self.scene.create_physical_material(1.0, 1.0, 0)
        root.add_capsule_collision(pose=sapien.Pose([0, 0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
                                   radius=0.025, half_length=0.175)
        root.add_capsule_collision(pose=sapien.Pose([0, -0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
                                   radius=0.025, half_length=0.175)
        # door
        door = builder.create_link_builder(root)
        door.set_name('door')
        door.add_box_collision(pose=sapien.Pose([0, 0, 0.2]), half_size=[0.025, 0.15, 0.2], density=100,
                               material=table_physics_mat)
        door.set_joint_properties(
            'revolute',
            limits=[[0, np.pi / 2]],
            pose_in_parent=sapien.Pose([0, -0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
            pose_in_child=sapien.Pose([0, -0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
            friction=0.,
            damping=0.)
        # handle
        hinge = builder.create_link_builder(door)
        hinge.add_capsule_collision(pose=sapien.Pose([-0.02, 0, 0]), radius=0.015,
                                    half_length=0.07, material=table_physics_mat)
        hinge.add_capsule_collision(pose=sapien.Pose([-0.09, -0.05, 0], [0.707, 0, 0, 0.707]), radius=0.015,
                                    half_length=0.05, material=table_physics_mat)
        hinge.add_capsule_collision(pose=sapien.Pose([0.05, 0.04, 0], [0.707, 0, 0, 0.707]), radius=0.015,
                                    half_length=0.04, material=table_physics_mat)
        hinge.set_joint_properties(
            'revolute',
            limits=[[0, np.pi / 2]],
            pose_in_parent=sapien.Pose([0, 0.10, 0.2]),
            pose_in_child=sapien.Pose([0, 0, 0]),
            friction=0.,
            damping=0.)
        hinge.set_name('handle')

        # Visual
        if self.render_enabled:
            frame_viz_mat = sapien.render.RenderMaterial()
            frame_viz_mat.set_base_color([0.25, 0.0, 0.0, 1])
            door_viz_mat = sapien.render.RenderMaterial()
            door_viz_mat.set_base_color([0.8, 0.3, 0.0, 1])
            door_viz_mat.set_roughness(0.2)
            root.add_capsule_visual(pose=sapien.Pose([0, 0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
                                    radius=0.025, half_length=0.175, material=frame_viz_mat)
            root.add_capsule_visual(pose=sapien.Pose([0, -0.185, 0.2], transforms3d.euler.euler2quat(0, -np.pi / 2, 0)),
                                    radius=0.025, half_length=0.175, material=frame_viz_mat)
            door.add_box_visual(pose=sapien.Pose([0, 0, 0.2]), half_size=[0.025, 0.15, 0.2], material=door_viz_mat)
            hinge.add_capsule_visual(pose=sapien.Pose([-0.02, 0, 0]), radius=0.015, half_length=0.07)
            hinge.add_capsule_visual(pose=sapien.Pose([-0.09, -0.05, 0], [0.707, 0, 0, 0.707]), radius=0.015,
                                     half_length=0.05)
            hinge.add_capsule_visual(pose=sapien.Pose([0.05, 0.04, 0], [0.707, 0, 0, 0.707]), radius=0.015,
                                     half_length=0.04)

        door = builder.build(fix_root_link=True)
        door.set_name('table_door')
        return door

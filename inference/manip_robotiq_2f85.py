"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import sys
import numpy as np
from os import path
from pyRobotiqGripper import RobotiqGripper
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import transform_from
sys.path.append(path.join(path.dirname(__file__), '..'))
from adf.manipulator import Manipulator


class ManipRobotiq2F85:
    # panda flange-frame to robotiq-eef_base_link-frame; with ft300 in between
    flange2root = transform_from(matrix_from_euler((0, 0, np.pi / 2), 0, 1, 2, True),
                              [0, 0, 0.052])

    def __init__(self, n_pc):
        self.n_pc = n_pc
        self.manip = RobotiqGripper()
        self.manip.activate()
        self.hand = Manipulator('robotiq_2f85')
        root2eef = self.hand.get_links_transform('palm_base_link', 'eef_link')
        self.flange2eef = self.flange2root @ root2eef

    def get_manip(self):
        return self.manip.getPosition()

    def move_manip(self, val):
        self.manip.goTo(val, speed=1, force=1)

    def get_pc(self):
        """
        linear mapping [0, 255]int observation value to [0, 0.82]rad revolute joint value
        """
        q = self.get_manip()
        q = q / 255. * 0.82
        self.hand.forward_kinematic([q])
        pc = self.hand.anchor_to_pc(self.hand.get_anchor())[:self.n_pc]
        return pc

    def set_pc(self, pc):
        """
        linear mapping [0, 0.82]rad revolute joint value to [0, 255]int observation value
        """
        self.hand.reset()
        self.hand.inverse_kinematic(self.hand.pc_to_anchor(pc))
        q = np.array(self.hand.get_joint(all=False))
        q = q / 0.82 * 255
        self.move_manip(q)

    def close(self):
        self.manip.close()

    def open(self):
        self.manip.open()

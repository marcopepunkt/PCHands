"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import sys
import numpy as np
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))
from inference.leap_hand.utils import LEAPhand_to_LEAPsim, LEAPsim_to_LEAPhand
from inference.leap_hand.driver import LeapNode
from adf.manipulator import Manipulator


class ManipLeapHandRight:
    qid_s2r = [5, 4, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15, 0, 1, 2, 3]
    qid_r2s = [12, 13, 14, 15, 1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11]

    def __init__(self, n_pc, pid=(600, 0, 100)):
        self.n_pc = n_pc
        self.manip = LeapNode(pid=pid, port='/dev/ttyUSB0')
        self.hand = Manipulator('leap_hand_right')
        self.flange2eef = self.hand.get_links_transform('flange_link', 'eef_link')

    def get_manip(self):
        q = self.manip.read_pos()
        q = LEAPhand_to_LEAPsim(q)
        q = q[self.qid_r2s]
        return q

    def move_manip(self, q):
        q = q[self.qid_s2r]
        q = LEAPsim_to_LEAPhand(q)
        self.manip.set_leap(q)

    def get_pc(self):
        q = self.get_manip()
        self.hand.forward_kinematic(q.tolist())
        pc = self.hand.anchor_to_pc(self.hand.get_anchor())[:self.n_pc]
        return pc

    def set_pc(self, pc):
        self.hand.reset()
        self.hand.inverse_kinematic(self.hand.pc_to_anchor(pc))
        q = np.array(self.hand.get_joint(all=False))
        self.move_manip(q)

    def close(self):
        self.move_manip(np.array([1.6, 0, -0.1, 1,
                                  0, 0, 0, 1,
                                  0, 0, 0, 1,
                                  0, 0, 0, 1]))

    def open(self):
        self.move_manip(np.zeros(16))


if __name__ == '__main__':
    import time
    a = ManipLeapHandRight(n_pc=1)

    for i in range(2):
        for q in np.linspace(2, -2, 40):
            a.set_pc(np.array([q]))
            time.sleep(0.1)
        for q in np.linspace(-2, 2, 40):
            a.set_pc(np.array([q]))
            time.sleep(0.1)

"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import sys
import numpy as np
from os import path
from mplib import Pose as mpPose
sys.path.append(path.join(path.dirname(__file__), '../..'))
from rl_sim.utils.common_robot_utils import (load_robot, load_planner,
                                             encode_pose, decode_pose)
from adf.manipulator import Manipulator
from adf.mano_hand import ManoHand


class ManipulatorRobot:
    def __init__(self, scene, robot_name, n_pc=10, ik_link='eef_link',
                 frame_skip=5, teleop=False, dps=(0.3, 1.0, 1.0)):
        # Create robot
        self.scene = scene
        self.frame_skip = frame_skip
        if teleop:
            self.manoh = ManoHand()
        self.manip = Manipulator(robot_name)
        self.robot = load_robot(scene, robot_name)
        self.robot.set_qpos(np.zeros(self.robot.dof))
        self.q_ctrl = False
        self.n_pc = n_pc
        if n_pc == 0:
            self.n_pc = self.robot.dof - 6
            self.q_ctrl = True
        self.planner = load_planner(robot_name, ik_link)
        self.joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.qidx_fward = [sorted(self.joint_names[6:]).index(joint) for joint in self.joint_names[6:]]
        self.qidx_bward = [self.joint_names[6:].index(joint) for joint in sorted(self.joint_names[6:])]

        # robot Dynamics
        joints = self.robot.get_active_joints()
        for i in range(3):
            joints[i].set_drive_property(stiffness=1000, damping=40, force_limit=50)
        for i in range(3, 6):
            joints[i].set_drive_property(stiffness=400, damping=15, force_limit=50)
        for i in range(6, self.robot.dof):
            joints[i].set_drive_property(stiffness=300, damping=10, force_limit=10)
        ctrl_freq = 1 / (frame_skip * self.scene.get_timestep())
        self.dq_clip = np.array([dps[0] / ctrl_freq] * 3 +
                                [dps[1] / ctrl_freq] * 3 +
                                [dps[2] / ctrl_freq] * (self.robot.dof - 6))

    def reorder_joint(self, qpos, forward=True):
        return qpos.copy()[self.qidx_fward if forward else self.qidx_bward]

    def get_qpospc(self):
        qpos = self.robot.get_qpos()
        if self.q_ctrl:
            return qpos
        f_qpos = self.reorder_joint(qpos[6:], False)
        self.manip.set_joint(f_qpos)
        pc = self.manip.anchor_to_pc(self.manip.get_anchor())[:self.n_pc]
        return np.concatenate((qpos[:6], pc))

    def get_posepc(self, eef_link='eef_link'):
        qpos = self.robot.get_qpos()
        # eef pose
        pose = self.robot.find_link_by_name(eef_link).get_entity_pose()
        pose = encode_pose(pose.to_transformation_matrix())
        if self.q_ctrl:
            return np.hstack((pose, qpos[6:]))
        # finger pc
        f_qpos = self.reorder_joint(qpos[6:], False)
        self.manip.set_joint(f_qpos)
        pc = self.manip.anchor_to_pc(self.manip.get_anchor())[:self.n_pc]
        return np.hstack((pose, pc))

    def mocap_to_posepc(self, data, w2cam):
        """
        convert mano mocap data to posepc format
        :param data: dict mocap data
        :param w2cam: np.array [4, 4] transformation w2cam
        :return: np.array [3+6+n_pc] : tsl + rot_2col + pc
        """
        anchor = self.manoh.vertex_to_anchor(data['vertices']).copy()
        pc = self.manoh.anchor_to_pc(self.manoh.anchor_transform(anchor), self.n_pc)[:self.n_pc]

        anchor = anchor + data['offset']
        anchor = np.matmul(w2cam, np.vstack((anchor.T, np.ones((1, anchor.shape[0]))))).T[:, :-1]
        w2eef = self.manoh.anchor_transform(anchor, return_tf=True)
        posepc = np.hstack((encode_pose(w2eef), pc))
        return posepc

    def control_robot(self, posepc, locked=False):
        # set finger
        if self.q_ctrl:
            f_qpos = posepc[9:]
        else:
            self.manip.reset()
            self.manip.inverse_kinematic(self.manip.pc_to_anchor(posepc[9:]))
            f_qpos = np.array(self.manip.get_joint(all=True))
            f_qpos = self.reorder_joint(f_qpos, True)

        # set eef
        eef_pose = mpPose(decode_pose(posepc[:9]))
        _, r_qpos = self.planner.IK(goal_pose=eef_pose, start_qpos=self.robot.get_qpos()[:6], return_closest=True)
        if r_qpos is None:
            r_qpos = self.robot.get_qpos()[:6]

        # control
        tgt_qpos = np.hstack((r_qpos, f_qpos))
        now_qpos = self.robot.get_qpos()
        tgt_qpos = now_qpos + np.clip(tgt_qpos - now_qpos, -self.dq_clip, self.dq_clip)
        if locked:
            tgt_qpos[6:] = [joint.get_drive_target().flatten()[0] for joint in self.robot.get_active_joints()[6:]]
        for idx, joint in enumerate(self.robot.get_active_joints()):
            joint.set_drive_target(tgt_qpos[idx])

    def set_qpospc(self, qpospc):
        if self.q_ctrl:
            self.robot.set_qpos(qpospc)
        else:
            self.manip.reset()
            self.manip.inverse_kinematic(self.manip.pc_to_anchor(qpospc[6:]))
            qpos = np.array(self.manip.get_joint(all=True))
            qpos = self.reorder_joint(qpos, True)
            self.robot.set_qpos(np.concatenate((qpospc[:6], qpos)))

    def set_posepc(self, posepc):
        # set finger
        if self.q_ctrl:
            f_qpos = posepc[9:]
        else:
            self.manip.reset()
            self.manip.inverse_kinematic(self.manip.pc_to_anchor(posepc[9:]))
            f_qpos = np.array(self.manip.get_joint(all=True))
            f_qpos = self.reorder_joint(f_qpos, True)

        # set eef
        eef_pose = mpPose(decode_pose(posepc[:9]))
        _, r_qpos = self.planner.IK(goal_pose=eef_pose, start_qpos=self.robot.get_qpos()[:6], return_closest=True)

        self.robot.set_qpos(np.hstack((r_qpos, f_qpos)))

    def rl_step(self, action):
        """
        :param action: nd.array [9 + n_pc]
        3 eef translation,
        6 eef rotation,
        n_pc absolute finger principal components
        :return:
        """
        self.control_robot(action)

        for i in range(self.frame_skip):
            self.robot.set_qf(self.robot.compute_passive_force())
            self.scene.step()

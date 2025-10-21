"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
Based on code from https://github.com/yzqin/dex-hand-teleop with MIT licence
"""
import sys
import numpy as np
from os import path
from functools import cached_property
sys.path.append(path.join(path.dirname(__file__), '..'))
from env.mugflip_env import MugFlipEnv
from kinematics.manipulator import ManipulatorRobot
from utils.common_robot_utils import encode_pose


class MugFlipRLEnv(MugFlipEnv):
    def __init__(self, robot_name='shadow_hand_right', n_pc=6, f_norm=None):
        super().__init__()
        self.norm_stat = dict(obs_mean=0, obs_std=1, act_min=-1, act_max=1)
        if f_norm is not None:
            self.norm_stat = np.load(f_norm)
        self.setup_robot(robot_name, n_pc)

    def setup_robot(self, robot_name, n_pc):
        if robot_name is None:
            return
        if isinstance(robot_name, str):
            self.robot = ManipulatorRobot(self.scene, robot_name, n_pc=n_pc,
                                          dps=self.np_random.uniform(low=[0.3, 1.0, 1.5], high=[0.7, 2.0, 2.5]))
            self.robot.set_qpospc(np.hstack((self.palm_init, np.zeros(self.robot.n_pc))))
            self.rl_step = self.robot.rl_step
        else:
            # setting reference manipulator
            self.robot = robot_name

        self.eef = self.robot.robot.find_link_by_name('eef_link')

    def get_observation(self):
        robot_posepc = self.robot.get_posepc()
        object_pose = self.manipulated_object.get_pose()
        object_pose = encode_pose(np.concatenate([object_pose.p, object_pose.q]))
        state = np.concatenate([robot_posepc, object_pose])
        state = (state - self.norm_stat['obs_mean']) / self.norm_stat['obs_std']
        return state

    def get_reward(self, action=None):
        object_pose = self.manipulated_object.get_pose()
        eef_pos = self.eef.get_entity_pose().p
        cnt_contact = self.count_contact(self.scene, self.robot.robot.get_links(),
                                        self.manipulated_object.get_components())
        hit_table = self.count_contact(self.scene, self.robot.robot.get_links(),
                                       self.table.get_components())
        z_axis = object_pose.to_transformation_matrix()[:3, 2]
        theta_cos = max(0, np.sum(np.array([0, 0, 1]) * z_axis))
        obj_target_distance = np.linalg.norm(self.original_object_pos[:2] - object_pose.p[:2])

        reward = -0.06 * bool(hit_table)
        if theta_cos < 0.7:
            reward -= 0.1 * min(np.linalg.norm(eef_pos - object_pose.p), 0.5)
            if cnt_contact:
                reward += 0.2 * bool(cnt_contact)
                reward += theta_cos
        else:
            reward += 0.3 + theta_cos
            reward += 10 * (0.1 - min(0.1, obj_target_distance))

        return reward / 2.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reset_env()
        palm_init = self.np_random.uniform(low=[ -0.35, -0.10, 0.20, -0.5, -0.5, -0.5],
                                           high=[-0.20,  0.10, 0.35,  0.5,  0.5,  0.5])
        if not self.robot.q_ctrl:
            pc_init = self.np_random.uniform(low=-1., high=2., size=self.robot.n_pc)
        else:
            pc_init = [self.np_random.uniform(low=j.limits[0][0],
                                              high=j.limits[0][0] + (j.limits[0][1] - j.limits[0][0]) / 2)
                       for j in self.robot.robot.get_active_joints()[6:]]
        self.robot.set_qpospc(np.hstack((palm_init, pc_init)))
        return self.get_observation(), {}

    def is_done(self):
        """ return: terminated, truncated """
        return False, self.current_step >= self.horizon

    @cached_property
    def action_dim(self):
        return 9 + self.robot.n_pc

    @cached_property
    def obs_dim(self):
        return 9 + self.robot.n_pc + 9

    @cached_property
    def horizon(self):
        return 300


def test_env():
    env = MugFlipRLEnv(robot_name='shadow_hand_right')

    for _ in range(10):
        env.reset()
        for i in range(env.horizon):
            obs, reward, _, done, _ = env.step(env.action_space.sample())
            env.render()


if __name__ == '__main__':
    test_env()

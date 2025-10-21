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
from env.relocate_env import RelocateEnv
from kinematics.manipulator import ManipulatorRobot
from rl_sim.utils.common_robot_utils import encode_pose


class RelocateRLEnv(RelocateEnv):
    def __init__(self, robot_name='allegro_hand_right', object_name='tomato_soup_can', n_pc=6, f_norm=None):
        super().__init__(object_name)
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
        object_pos = self.manipulated_object.get_pose().p
        target_pos = self.target_object.get_pose().p
        state = np.concatenate([robot_posepc, object_pos, target_pos[:2]])
        state = (state - self.norm_stat['obs_mean']) / self.norm_stat['obs_std']
        return state

    def get_reward(self, action=None):
        object_pos = self.manipulated_object.get_pose().p
        eef_pos = self.eef.get_entity_pose().p
        target_pos = self.target_object.get_pose().p
        cnt_contact = self.count_contact(self.scene, self.robot.robot.get_links(),
                                         self.manipulated_object.get_components())
        hit_table = self.count_contact(self.scene, self.robot.robot.get_links(),
                                       self.tables.get_components())

        reward = -0.1 * min(np.linalg.norm(eef_pos - object_pos), 0.5)
        reward -= 0.1 * bool(hit_table)
        if cnt_contact:
            reward += 0.1 * min(4, cnt_contact)
            lift = max(0, min(object_pos[2], target_pos[2]) - self.object_height)
            reward += 15 * lift
            if lift > 0.05:
                obj_target_distance = min(np.linalg.norm(object_pos - target_pos), 0.5)
                reward += 5 * (0.5 - obj_target_distance)

        return reward / 5.1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reset_env()
        if self.object_name == 'potted_meat_can':
            eef_init = self.np_random.uniform(low=[ -0.35, -0.10, 0.20, -0.5, 0.0, -0.5],
                                              high=[-0.20,  0.10, 0.35,  0.5, 1.0,  0.5])
        else:
            eef_init = self.np_random.uniform(low=[ -0.35, -0.10, 0.20, -0.5, -0.5, -0.5],
                                              high=[-0.20,  0.10, 0.35,  0.5,  0.5,  0.5])
        if not self.robot.q_ctrl:
            pc_init = self.np_random.uniform(low=1., high=2.5, size=self.robot.n_pc)
        else:
            pc_init = [self.np_random.uniform(low=j.limits[0][0],
                                              high=j.limits[0][0] + (j.limits[0][1] - j.limits[0][0]) / 2)
                       for j in self.robot.robot.get_active_joints()[6:]]
        self.robot.set_posepc(np.hstack((encode_pose(eef_init), pc_init)))
        return self.get_observation(), {}

    def is_done(self):
        """ return: terminated, truncated """
        return False, self.current_step >= self.horizon

    @cached_property
    def action_dim(self):
        return 9 + self.robot.n_pc

    @cached_property
    def obs_dim(self):
        return 9 + self.robot.n_pc + 3 + 2

    @cached_property
    def horizon(self):
        return 300


def test_env():
    env = RelocateRLEnv(robot_name='shadow_hand_right', object_name='mustard_bottle')

    for _ in range(10):
        env.reset()
        for i in range(env.horizon):
            obs, reward, _, done, _ = env.step(env.action_space.sample())
            env.render()


if __name__ == '__main__':
    test_env()

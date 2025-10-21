"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import sys
import time
import pickle
import numpy as np
from os import path
from panda_bot import PandaRobot
from pose_estimation import PoseEstimator
from pytransform3d.transformations import transform_sclerp
from rl_sim.utils.common_robot_utils import encode_pose, decode_pose


class PickUpEnv:
    def __init__(self, exp_name):
        n_pc = 2
        manip_name = 'robotiq_2f85'
        obj_name = 'potted_meat_can'
        # hardware system
        self.sys = PandaRobot(manip_name=manip_name, n_pc=n_pc)
        # pose estimator
        # self.poster = PoseEstimator(cam=self.sys.camera, object_name=obj_name)
        # self.poster.start()
        # norm stat
        f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/customized/relocate_{}_{}.npz'.format(obj_name, n_pc))
        self.norm_stat = np.load(f_norm)
        # policy
        self.policy = self.load_policy(exp_name)

    def step(self, action):
        action = ((action + 1) / 2) * (self.norm_stat['act_max'] - self.norm_stat['act_min']) + self.norm_stat['act_min']
        self.sys.set_posepc(action)
        return

    def reset(self):
        self.sys.home()
        return self.get_observation()

    def get_observation(self):
        # observations
        # obj_pose = self.poster.read()
        obj_pose = np.array([0, 0, 0.09, 1, 0, 0, 1, 0, 0])
        robot_posepc = self.sys.get_posepc()
        tgt_pos = np.array([-0.1, 0, 0.2])

        # relocate_task
        state = np.concatenate([robot_posepc, obj_pose[:3], tgt_pos[:2]])
        state = (state - self.norm_stat['obs_mean']) / self.norm_stat['obs_std']
        return state

    @staticmethod
    def load_policy(exp_name):
        f_policy = path.join(path.dirname(__file__), '../rl_scripts/experiments',
                             exp_name, 'policy_0400.pickle')
        return pickle.load(open(f_policy, 'rb'))

    def run(self):
        self.sys.robot.recover_from_errors()
        self.reset()
        for _ in range(1000):
            obs = self.get_observation()
            action = self.policy.get_action(obs)[1]['mean']
            self.step(action)
        self.poster.stop()

    def quit(self):
        self.poster.stop()
        self.sys.close()


if __name__ == "__main__":
    PickUpEnv(sys.argv[1]).run()

"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import os
import sys
import glob
import time
import pickle
import numpy as np
from os import path
sys.path.append(path.join(path.dirname(__file__), '../rl_sim'))
from env.tabledoor_rl import TableDoorRLEnv
from env.relocate_rl import RelocateRLEnv
from env.mugflip_rl import MugFlipRLEnv
from kinematics.manipulator import ManipulatorRobot
from kinematics.mano_robot_hand import MANORobotHand
from kinematics.retarget_optim import PositionRetargeting


class EnvPlayer:
    def __init__(self, data, env, robot, optim=False):
        self.env = env
        self.data = data
        self.robot = robot
        self.optim = None
        if optim:
            self.optim = PositionRetargeting(self.env.robot.robot)

    def bake_demonstration(self, vis=False):
        baked_data = dict(observations=[], actions=[])

        # trajectory steps
        for i in range(len(self.data)):
            self.env.scene.step()
            # load entity poses
            for j in range(len(self.data[i]['simulation']['entity']['name'])):
                ent_name = self.data[i]['simulation']['entity']['name'][j]
                ent_pose = self.data[i]['simulation']['entity']['pose'][j]
                for ent in self.env.scene.entities:
                    if ent.get_name() == ent_name:
                        ent.set_pose(ent_pose)
            # load articulation joints
            for j in range(len(self.data[i]['simulation']['articulation']['name'])):
                art_name = self.data[i]['simulation']['articulation']['name'][j]
                art_qpos = self.data[i]['simulation']['articulation']['qpos'][j]
                for art in self.env.scene.get_all_articulations():
                    if art.get_name() == art_name:
                        art.set_qpos(art_qpos)

            # retargeting poses
            if self.optim is None:
                posepc = self.robot.get_posepc()
                if vis:
                    self.robot.robot.set_qpos([-1] + [0] * (self.robot.robot.dof - 1))  # move source robot away
                    self.env.robot.set_posepc(posepc)
            else:
                if i == 0:
                    self.optim.last_qpos[:6] = self.robot.robot.get_qpos()[:6]
                source_anc = np.stack([self.robot.robot.find_link_by_name('A_{:02d}'.format(i)).get_entity_pose().p
                                       for i in range(22)])
                self.robot.robot.set_qpos([-1] + [0] * (self.robot.robot.dof - 1))  # move source robot away
                qpospc = self.optim.retarget(source_anc)
                self.env.robot.set_qpospc(qpospc)
                posepc = self.env.robot.get_posepc()

            # baking action
            if i >= 1:
                baked_data["actions"].append(posepc)

            # baking observation
            baked_data["observations"].append(self.env.get_observation())

            # visualize
            if vis:
                self.env.render()
                time.sleep(0.025)

        baked_data["actions"].append(baked_data["actions"][-1])
        if vis  :
            self.env.viewer.close()

        return baked_data


def norm_data(baked_data):
    obs = np.concatenate([data['observations'] for data in baked_data], axis=0)
    act = np.concatenate([data['actions'] for data in baked_data], axis=0)
    # states
    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs, axis=0)
    # actions
    act_min = np.min(act, axis=0)
    act_max = np.max(act, axis=0)
    return dict(obs_mean=obs_mean, obs_std=obs_std,
                 act_min=act_min, act_max=act_max)


def bake_optim(task, name_tgt, vis=False):
    # prepare folder
    demo_path = path.join(path.dirname(__file__), 'demo/customized')
    os.makedirs(demo_path, exist_ok=True)
    f_data = path.join(demo_path, '{}_{}_{}.pkl'.format(task, name_tgt, 0))
    stat_path = path.join(path.dirname(__file__), '../rl_sim/env/norm/customized')
    os.makedirs(stat_path, exist_ok=True)
    f_norm = path.join(stat_path, '{}_{}_{}.npz'.format(task, name_tgt, 0))

    # demo data
    f_demo = path.join(path.dirname(__file__), 'teleop_collection/customized', task, '*.pkl')
    pkls = sorted(glob.glob(f_demo))

    # baking
    baked_data = []
    for pkl in pkls:
        all_data = np.load(pkl, allow_pickle=True)
        meta_data = all_data['meta_data']
        data = all_data['data']

        # tgt env
        if task == 'table_door':
            env = TableDoorRLEnv(robot_name=name_tgt, n_pc=0)
        elif task == 'mug_flip':
            env = MugFlipRLEnv(robot_name=name_tgt, n_pc=0)
        elif 'relocate' in task:
            env = RelocateRLEnv(object_name=meta_data['env_kwargs']['object_name'],
                                robot_name=name_tgt, n_pc=0)

        # src manip
        bot_src = MANORobotHand(env.scene, init_joint_pos=meta_data["zero_joint_pos"])

        player = EnvPlayer(data, env, bot_src, optim=True)
        baked_ep = player.bake_demonstration(vis=vis)

        # save baked data
        baked_data.append(baked_ep)

    # saving
    norm_stat = norm_data(baked_data)
    pickle.dump(baked_data, open(f_data, 'wb'))
    np.savez_compressed(f_norm, **norm_stat)
    print('Baked {} trajectories'.format(len(pkls)))


def bake_pchands(task, name_src, n_pc, vis=False):
    # prepare folder
    demo_path = path.join(path.dirname(__file__), 'demo', name_src)
    os.makedirs(demo_path, exist_ok=True)
    f_data = path.join(demo_path, '{}_{}.pkl'.format(task, n_pc))
    stat_path = path.join(path.dirname(__file__), '../rl_sim/env/norm', name_src)
    os.makedirs(stat_path, exist_ok=True)
    f_norm = path.join(stat_path, '{}_{}.npz'.format(task, n_pc))

    # demo data
    f_demo = path.join(path.dirname(__file__), 'teleop_collection', name_src, task, '*.pkl')
    pkls = sorted(glob.glob(f_demo))

    # baking
    baked_data = []
    for pkl in pkls:
        all_data = np.load(pkl, allow_pickle=True)
        meta_data = all_data['meta_data']
        data = all_data['data']

        # tgt env
        if task == 'table_door':
            env = TableDoorRLEnv(robot_name=None, n_pc=n_pc)
        elif task == 'mug_flip':
            env = MugFlipRLEnv(robot_name=None, n_pc=n_pc)
        elif 'relocate' in task:
            env = RelocateRLEnv(object_name=meta_data['env_kwargs']['object_name'],
                                robot_name=None, n_pc=n_pc)

        # src manip
        if meta_data["robot_name"] == 'customized':
            robot = MANORobotHand(env.scene, init_joint_pos=meta_data["zero_joint_pos"], n_pc=n_pc)
        else:
            robot = ManipulatorRobot(env.scene, meta_data["robot_name"], n_pc=n_pc)
        env.setup_robot(robot, n_pc)

        player = EnvPlayer(data, env, robot)
        baked_ep = player.bake_demonstration(vis=vis)

        # save baked data
        baked_data.append(baked_ep)

    # saving
    norm_stat = norm_data(baked_data)
    pickle.dump(baked_data, open(f_data, 'wb'))
    np.savez_compressed(f_norm, **norm_stat)
    print('Baked {} trajectories'.format(len(pkls)))


def bake_vis(task, name_src, name_tgt, n_pc, vis=True):
    # demo data
    f_demo = path.join(path.dirname(__file__), 'teleop_collection', name_src, task, '*.pkl')
    pkls = sorted(glob.glob(f_demo))

    # baking
    for pkl in pkls:
        all_data = np.load(pkl, allow_pickle=True)
        meta_data = all_data['meta_data']
        data = all_data['data']

        # tgt env
        if task == 'table_door':
            env = TableDoorRLEnv(robot_name=name_tgt, n_pc=n_pc)
        elif task == 'mug_flip':
            env = MugFlipRLEnv(robot_name=name_tgt, n_pc=n_pc)
        elif 'relocate' in task:
            env = RelocateRLEnv(object_name=meta_data['env_kwargs']['object_name'],
                                robot_name=name_tgt, n_pc=n_pc)

        # src manip
        if meta_data["robot_name"] == 'customized':
            robot = MANORobotHand(env.scene, init_joint_pos=meta_data["zero_joint_pos"], n_pc=n_pc)
        else:
            robot = ManipulatorRobot(env.scene, meta_data["robot_name"], n_pc=n_pc)

        player = EnvPlayer(data, env, robot, optim=n_pc == 0)
        player.bake_demonstration(vis=vis)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    # bake exp1 baseline
    # bots = ['allegro_hand_right', 'schunk_hand_right', 'shadow_hand_right']
    # tasks = ['table_door', 'relocate_mustard_bottle', 'relocate_potted_meat_can', 'relocate_tomato_soup_can', 'mug_flip']
    # for bot in bots:
    #     for task in tasks:
    #         bake_optim(task=task, name_tgt=bot, vis=False)

    # bake exp1 pchands
    # n_pcs = [1, 2, 4]
    # tasks = ['table_door', 'relocate_mustard_bottle', 'relocate_potted_meat_can', 'relocate_tomato_soup_can', 'mug_flip']
    # for n_pc in n_pcs:
    #     for task in tasks:
    #         bake_pchands(task=task, name_src='customized', n_pc=n_pc)

    # bake exp2
    # n_pc = 2
    # name_srcs = ['robotiq_2f85', 'kinova_3f_right', 'leap_hand_right']
    # tasks = ['table_door', 'relocate_mustard_bottle', 'relocate_potted_meat_can', 'relocate_tomato_soup_can', 'mug_flip']
    # for name_src in name_srcs:
    #     for task in tasks:
    #         bake_pchands(task=task, name_src=name_src, n_pc=n_pc)

    # vis translate
    bake_vis(task='table_door', name_src='customized', name_tgt='allegro_hand_right', n_pc=1)

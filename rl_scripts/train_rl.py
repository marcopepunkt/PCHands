"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import sys
import csv
import json
import tyro
import time
import torch
import wandb
import pickle
import random
import numpy as np
from dataclasses import dataclass
from os import path, makedirs, getcwd, umask
sys.path.append(path.join(path.dirname(__file__), '..'))
from rl_scripts.algo_rl import TRPO, DAPG, MLP, MLPBaseline


@dataclass
class Args:
    # experiment
    algo: str = ''
    """algorithm name"""
    task: str = ''
    """task name"""
    robot: str = ''
    """robot name"""
    n_pc: int = 0
    """number of pc """
    seed: int = 0
    """seed of the experiment"""
    save_freq: int = 100
    """save frequency for model and log"""
    demo_source: str = 'customized'
    """dapg demo source"""

    # model
    lr: float = 5e-4
    """learning rate"""
    update_epochs: int = 4
    """number of epochs to update"""
    batch_size: int = 1024
    """batch size"""
    wd: float = 1e-5
    """weight decay"""
    hidden0: int = 64
    """actor hidden layer 0 size"""
    hidden1: int = 64
    """actor hidden layer 1 size"""

    # training
    num_iter: int = 400
    """number of iterations"""
    num_cpu: int = 64
    """number of parallel game environments"""
    num_ep: int = 256
    """number of episode"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    lam0: float = 0.3
    """dapg lambda 0"""
    lam1: float = 0.98
    """dapg lambda 1"""
    max_kl: float = 0.01
    """maximum allowed kl distance between updates"""
    damping: float = 1e-4
    """damping factor for conjugate gradient"""
    rpo_alpha: float = 0
    """robust policy optimization alpha"""


def main(args):
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.CRITICAL)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # env
    if args.task == 'table_door':
        from rl_sim.env.tabledoor_rl import TableDoorRLEnv
        env = TableDoorRLEnv(robot_name=args.robot, n_pc=args.n_pc)
        if args.n_pc == 0:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/table_door_{}_{}.npz'.format(
                'customized', args.robot, args.n_pc))
        else:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/table_door_{}.npz'.format(
                'customized', args.n_pc))
        env_spec = args.task, args.robot, args.n_pc, f_norm
    elif args.task == 'mug_flip':
        from rl_sim.env.mugflip_rl import MugFlipRLEnv
        env = MugFlipRLEnv(robot_name=args.robot, n_pc=args.n_pc)
        if args.n_pc == 0:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/mug_flip_{}_{}.npz'.format(
                'customized', args.robot, args.n_pc))
        else:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/mug_flip_{}.npz'.format(
                'customized', args.n_pc))
        env_spec = args.task, args.robot, args.n_pc, f_norm
    elif 'relocate' in args.task:
        from rl_sim.env.relocate_rl import RelocateRLEnv
        object_name = '_'.join(args.task.split('_')[1:])
        env = RelocateRLEnv(robot_name=args.robot, n_pc=args.n_pc, object_name=object_name)
        if args.n_pc == 0:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/relocate_{}_{}_{}.npz'.format(
                'customized', object_name, args.robot, args.n_pc))
        else:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/relocate_{}_{}.npz'.format(
                'customized', object_name, args.n_pc))
        env_spec = args.task, args.robot, args.n_pc, object_name, f_norm
    else:
        raise NotImplementedError
    dim_obs, dim_act = env.obs_dim, env.action_dim

    # policy
    policy = MLP(dim_obs, dim_act, hidden_sizes=(args.hidden0, args.hidden1), rpo_alpha=args.rpo_alpha)
    baseline = MLPBaseline(dim_obs, wd=args.wd, batch_size=args.batch_size, epochs=args.update_epochs, lr=args.lr)
    # agent
    if args.algo == 'trpo':
        agent = TRPO(env_spec, policy, baseline, max_kl=args.max_kl, damping=args.damping, seed=args.seed)
    elif args.algo == 'dapg':
        # demo
        if args.n_pc > 0:
            f_demo = path.join(path.dirname(__file__), 'demo', args.demo_source, '{}_{}.pkl'.format(args.task, args.n_pc))
        else:
            f_demo = path.join(path.dirname(__file__), 'demo', args.demo_source, '{}_{}_{}.pkl'.format(args.task, args.robot, args.n_pc))
        agent = DAPG(env_spec, policy, baseline, max_kl=args.max_kl, damping=args.damping,
                     f_demo=f_demo, f_norm=f_norm, seed=args.seed, lam0=args.lam0, lam1=args.lam1)
    else:
        raise NotImplementedError

    # prepare folder
    timestamp = time.strftime('%y%m%d_%H%M%S')
    job_name = '_'.join([args.algo, args.task, args.robot, str(args.n_pc), timestamp])
    job_dir = path.join(getcwd(), job_name)
    umask(0)
    makedirs(job_dir)
    try:
        wandb.init(project='pchands_rl', name=job_name, config=vars(args), force=False)
    except:
        wandb.init(project='pchands_rl', name=job_name, config=vars(args), mode='offline')
    json.dump(vars(args), open(path.join(job_dir, 'cfg.json'), 'w'), indent=2)

    # training loop
    for i in range(args.num_iter):
        agent.train_step(num_ep=args.num_ep, gamma=args.gamma, gae_lambda=args.gae_lambda, num_cpu=args.num_cpu)
        wandb.log(step=i, commit=True, data={k: v[i] for k, v in agent.log.items()})
        # dump logging
        if (i + 1) % args.save_freq == 0 and i > 0:
            with open(path.join(job_dir, 'log.csv'), 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(agent.log.keys())
                for v in zip(*agent.log.values()):
                    writer.writerow(v)
            f_policy = path.join(job_dir, 'policy_{:04d}.pickle'.format(i + 1))
            pickle.dump(agent.policy, open(f_policy, 'wb'))
    wandb.finish()


if __name__ == '__main__':
    main(tyro.cli(Args))

"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import os
import sys
import numpy as np
import multiprocessing as mp
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def get_environment(env_spec):
    if env_spec[0] == 'table_door':
        from rl_sim.env.tabledoor_rl import TableDoorRLEnv
        env = TableDoorRLEnv(robot_name=env_spec[1], n_pc=env_spec[2], f_norm=env_spec[3])
    elif env_spec[0] == 'mug_flip':
        from rl_sim.env.mugflip_rl import MugFlipRLEnv
        env = MugFlipRLEnv(robot_name=env_spec[1], n_pc=env_spec[2], f_norm=env_spec[3])
    elif 'relocate' in env_spec[0]:
        from rl_sim.env.relocate_rl import RelocateRLEnv
        env = RelocateRLEnv(robot_name=env_spec[1], n_pc=env_spec[2], object_name=env_spec[3], f_norm=env_spec[4])
    else:
        raise NotImplementedError
    return env


def sample_paths_parallel(num_ep, policy, max_ep_len=1e5, env_spec=None,
                          num_cpu=1, seed=None, max_process_time=300):
    # launch processes
    if num_cpu == 1:
        results = do_rollout((num_ep, policy, max_ep_len, env_spec, seed))
    else:
        ep_per_cpu = int(np.ceil(num_ep / num_cpu))
        pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
        parallel_runs = [pool.apply_async(do_rollout,
                                          args=([ep_per_cpu, policy, max_ep_len, env_spec, seed + i * ep_per_cpu],))
                         for i in range(num_cpu)]
        try:
            results = [p.get(timeout=max_process_time) for p in parallel_runs]
        except Exception as e:
            print(str(e))
            pool.close()
            pool.terminate()
            pool.join()
            raise TimeoutError

        pool.close()
        pool.terminate()
        pool.join()

    # result is a paths type and results is list of paths
    paths = []
    for result in results:
        for path in result:
            paths.append(path)
    return paths


def do_rollout(args):
    num_ep, policy, max_ep_len, env_spec, seed = args
    env = get_environment(env_spec)
    max_ep_len = min(max_ep_len, env.horizon)

    paths = []
    for ep in range(num_ep):
        env.seed(seed + ep)
        np.random.seed(seed + ep)

        observations = []
        actions = []
        rewards = []

        obs = env.reset()[0].clip(-5, 5)
        terminated = False
        truncated = False
        t = 0
        while t < max_ep_len and not (terminated or truncated):
            act, _ = policy.get_action(obs)
            next_obs, rwd, terminated, truncated, _ = env.step(act)
            observations.append(obs)
            actions.append(act)
            rewards.append(rwd)
            obs = next_obs.clip(-5, 5)
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            terminated=terminated or truncated
        )

        paths.append(path)

    del env
    return paths

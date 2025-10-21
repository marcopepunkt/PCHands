"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import os
import sys
import csv
import json
import time
import glob
import pickle
import matplotlib
import numpy as np
from os import path
import matplotlib.pyplot as plt
from multiprocessing import Pool
sys.path.append(path.join(path.dirname(__file__), '..'))


def vis_rollout(env, f_exp):
    print(f_exp)
    f_model = sorted(glob.glob(path.join(f_exp, 'policy_*.pickle')))[-1]
    policy = pickle.load(open(f_model, 'rb'))

    skip, quit = False, False

    def key_skip():
        nonlocal skip
        skip = True
        time.sleep(0.1)
    def key_quit():
        nonlocal quit
        quit = True
    env.key_map['q'] = key_quit
    env.key_map['a'] = key_skip

    # eval
    for i in range(10):
        skip = False
        state, done = env.reset()[0], False
        reward_sum = 0
        while True:
            action = policy.get_action(state)[1]['mean']
            state, reward, _, done, _ = env.step(action)
            env.render()
            time.sleep(0.025)
            reward_sum += reward
            if done or skip or quit:
                break
        print('ep: {}, total reward: {}'.format(i, reward_sum))
        if quit:
            break


def get_csv(algo, task, robot_name, n_pc, kwarg={}):
    f_exp = path.join(path.dirname(__file__), 'experiments',
                      '_'.join([algo, task, robot_name, str(n_pc), '*']))
    # filter: source
    f_exp = sorted(glob.glob(f_exp))
    for kw in kwarg.keys():
        f_exp = [f for f in f_exp
                 if json.load(open(path.join(f, 'cfg.json'), 'r'))[kw] == kwarg[kw]]
    f_exp = sorted(f_exp)[-1]

    data = csv.DictReader(open(path.join(f_exp, 'log.csv'), 'r'))
    data = np.array([[float(col['stoc_pol_mean']), float(col['stoc_pol_std'])] for col in data])
    means, stds = np.split(data, 2, axis=1)
    means = means.flatten()
    stds = stds.flatten()
    return means, stds


def get_env(algo, task, robot_name, n_pc, src=None):
    # experiment
    f_exp = path.join(path.dirname(__file__), 'experiments',
                      '_'.join([algo, task, robot_name, str(n_pc), '*']))
    if src is None:
        f_exp = sorted(glob.glob(f_exp))[-1]
    else:
        f_exp = [f for f in sorted(glob.glob(f_exp))
                 if json.load(open(path.join(f, 'cfg.json'), 'r'))['demo_source'] == src]
        f_exp = sorted(f_exp)[-1]

    if task == 'table_door':
        from rl_sim.env.tabledoor_rl import TableDoorRLEnv
        if n_pc == 0:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/table_door_{}_{}.npz'.format(
                'customized', robot_name, n_pc))
        else:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/table_door_{}.npz'.format(
                'customized', n_pc))
        env = TableDoorRLEnv(robot_name=robot_name, n_pc=n_pc, f_norm=f_norm)
    elif task == 'mug_flip':
        if n_pc == 0:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/mug_flip_{}_{}.npz'.format(
                'customized', robot_name, n_pc))
        else:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/mug_flip_{}.npz'.format(
                'customized', n_pc))
        from rl_sim.env.mugflip_rl import MugFlipRLEnv
        env = MugFlipRLEnv(robot_name=robot_name, n_pc=n_pc, f_norm=f_norm)
    elif 'relocate' in task:
        from rl_sim.env.relocate_rl import RelocateRLEnv
        object_name = '_'.join(task.split('_')[1:])
        if n_pc == 0:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/relocate_{}_{}_{}.npz'.format(
                'customized', object_name, robot_name, n_pc))
        else:
            f_norm = path.join(path.dirname(__file__), '../rl_sim/env/norm/{}/relocate_{}_{}.npz'.format(
                'customized', object_name, n_pc))
        env = RelocateRLEnv(robot_name=robot_name, n_pc=n_pc, object_name=object_name, f_norm=f_norm)
    else:
        raise NotImplementedError
    return env, f_exp


def plot_exp1(algos=['trpo', 'dapg'],
              tasks=['table_door', 'relocate_mustard_bottle', 'relocate_potted_meat_can', 'relocate_tomato_soup_can', 'mug_flip'],
              n_pcs=[0, 1, 2, 4],
              robot_names=['allegro_hand_right', 'schunk_hand_right', 'shadow_hand_right'],
              seeds=[0, 1, 2]):
    # matplotlib.use('TkAgg')
    cmap = matplotlib.colormaps['plasma']

    for idx_task, task in enumerate(tasks):
        fig, axs = plt.subplots(len(algos), len(robot_names), sharey=True, figsize=(7, 5))
        for idx_algo, algo in enumerate(algos):
            for idx_robot, robot_name in enumerate(robot_names):
                for idx_n_pc, n_pc in enumerate(n_pcs):
                    color = cmap(np.linspace(0, 1, 6)[idx_n_pc + (0 if n_pc == 0 else 1)])
                    meanss = []
                    for idx_seed , seed in enumerate(seeds):
                        try:
                            means, stds = get_csv(algo, task, robot_name, n_pc, kwarg={'seed': seed})
                            meanss.append(means)
                        except:
                            print('Missing!', algo, task, robot_name, 'npc:', n_pc, 'seed', seed)
                            continue
                    means = np.mean(meanss, axis=0)
                    stds = np.std(meanss, axis=0)
                    axs[idx_algo, idx_robot].plot(means, label='Baseline' if n_pc == 0 else '{}pc'.format(n_pc), color=color)
                    axs[idx_algo, idx_robot].fill_between(np.arange(means.shape[0]), means - stds, means + stds,
                                                          alpha=0.2, color=color)
                if idx_task ==0 and idx_algo == 0:
                    axs[idx_algo, idx_robot].set_title('{}'.format(robot_name.replace('_hand_right', '').capitalize()))
                axs[idx_algo, idx_robot].ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)

        task_name = {'relocate_mustard_bottle': 'RelocateMustard', 'relocate_potted_meat_can': 'RelocateMeatCan',
                     'relocate_tomato_soup_can': 'RelocateSoupCan', 'table_door': 'OpenDoor', 'mug_flip': 'FlipMug'}
        for idx_ax, ax in enumerate(axs.flat):
            ax.set(xlabel='Iterations', ylabel='{} \n {} Return'.format(task_name[task], ['TRPO', 'DAPG'][idx_ax // 3]))
            ax.label_outer()
        if idx_task == 0:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=len(n_pcs))
        fig.subplots_adjust(bottom=0.1, left=0.1, top=0.88, right=0.98, wspace=0.07, hspace=0.15)
        fig.savefig('baseline_{}.png'.format(task), dpi=150)


def plot_exp2(n_pc=2,
              srcs=['trpo', 'robotiq_2f85', 'kinova_3f_right', 'leap_hand_right'],
              tasks=['table_door', 'relocate_mustard_bottle', 'relocate_potted_meat_can', 'relocate_tomato_soup_can', 'mug_flip'],
              robot_names=['robotiq_2f85', 'kinova_3f_right', 'leap_hand_right']):
    # matplotlib.use('TkAgg')
    cmap = matplotlib.colormaps['plasma']
    task_name = {'relocate_mustard_bottle': 'RelocateMustard', 'relocate_potted_meat_can': 'RelocateMeatCan',
                 'relocate_tomato_soup_can': 'RelocateSoupCan', 'table_door': 'OpenDoor', 'mug_flip': 'FlipMug'}

    fig, axs = plt.subplots(len(robot_names), len(tasks), sharey=True, figsize=(5, 7))
    for idx_task, task in enumerate(tasks):
        for idx_robot, robot_name in enumerate(robot_names):
            for idx_src, src in enumerate(srcs):
                color = cmap(np.linspace(0, 1, 4)[idx_src])
                try:
                    if src == 'trpo':
                        means, stds = get_csv('trpo', task, robot_name, n_pc)
                    else:
                        means, stds = get_csv('dapg', task, robot_name, n_pc, kwarg={'demo_source': src})
                    # means = means[:500]
                    # stds = stds[:500]
                except:
                    print('Missing!', 'dapg', task, robot_name, n_pc, src)
                    continue
                axs[idx_robot, idx_task].plot(means, label='demo:{}'.format(src.replace('_right', '').capitalize()))
                axs[idx_robot, idx_task].fill_between(np.arange(means.shape[0]), means - stds, means + stds, alpha=0.2, color=color)
                if idx_robot == 0:
                    axs[idx_robot, idx_task].set_title(task_name[task])
                axs[idx_robot, idx_task].ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)

    for idx_ax, ax in enumerate(axs.flat):
        ax.set(xlabel='iteration', ylabel='{} \n Reward'.format(robot_names[idx_ax // len(tasks)].replace('_right', '').capitalize()))
        ax.label_outer()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(srcs))
    fig.subplots_adjust(bottom=0.05, left=0.05, top=0.91, right=0.98, wspace=0.07, hspace=0.15)
    plt.show()


def plot_exp22():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.use('TkAgg')
    src = ['--', '2F', '3F', '4F']
    tgt = ['2F', '3F', '4F']
    sr = {
        'table_door': [
            [92,95,97,92],[92,83,87,76],[29,30,82,90]
        ],
        'relocate_mustard_bottle': [
            [94,96,85,84],[88,84,95,96],[64,92,90,92]
        ],
        'relocate_potted_meat_can': [
            [95,90,88,87],[97,79,95,93],[81,78,87,78]
        ],
        'relocate_tomato_soup_can': [
            [84,91,89,96],[61,89,89,77],[0,72,62,89]
        ],
        'mug_flip': [
            [75,94,78,77],[65,71,79,89],[56,71,65,62]
        ]
    }

    def heatmap(data, row_labels, col_labels, title, ax=None):
        if ax is None:
            ax = plt.gca()

        # Plot the heatmap
        # im = ax.imshow(data, vmin=-100, vmax=100, cmap=matplotlib.colormaps['bone'])
        im = ax.imshow(data, vmin=0, vmax=110, cmap=matplotlib.colormaps['Greys'])

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(range(data.shape[1]), labels=col_labels, fontsize=15, color='red')
        ax.set_yticks(range(data.shape[0]), labels=row_labels, fontsize=15, color='blue')
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
        ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(title, fontsize=15)
        return im

    def annotate_heatmap(im, data=None,
                         textcolors=("black", "white"),
                         threshold=None):
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center",
                  fontsize=16)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                # kw.update(color='black')
                im.axes.text(j, i, data[i, j], **kw)

    task_name = {'relocate_mustard_bottle': 'Relocate-Mustard', 'relocate_potted_meat_can': 'Relocate-MeatCan',
                 'relocate_tomato_soup_can': 'Relocate-SoupCan', 'table_door': 'Open-Door', 'mug_flip': 'Flip-Mug'}
    # fig, axs = plt.subplots(1, 6, sharey=True, figsize=(14, 2.1))
    fig, axs = plt.subplots(1, 6, sharey=True, figsize=(14, 2.9))

    for idx, taskname in enumerate(sr.keys()):
        # im = heatmap(np.array(sr[taskname]), tgt, src, task_name[taskname], axs[idx])
        im = heatmap(np.array(sr[taskname]).T, src, tgt, task_name[taskname], axs[idx])
        annotate_heatmap(im)
        if idx == 0:
            # axs[idx].set_ylabel('tgt:', c='red', fontsize=15)
            # axs[idx].set_xlabel('src:', c='blue', fontsize=15, loc='left')
            # axs[idx].xaxis.set_label_coords(-0.3, -0.03)
            axs[idx].set_ylabel('src:', c='blue', fontsize=15)
            axs[idx].set_xlabel('tgt:', c='red', fontsize=15, loc='left')
            axs[idx].xaxis.set_label_coords(-0.27, -0.03)
    # average
    # im = heatmap(np.mean(np.array([sr[k] for k in sr.keys()]), axis=0).astype(int), tgt, src, 'Tasks-Average', axs[5])
    im = heatmap(np.mean(np.transpose(np.array([sr[k] for k in sr.keys()]), (0, 2, 1)), axis=0).astype(int), src, tgt, 'Tasks-Average', axs[5])
    annotate_heatmap(im)

    # plt.tight_layout()
    # fig.subplots_adjust(bottom=0, left=0.045, top=1.0, right=0.998, wspace=0.1)
    fig.subplots_adjust(bottom=0, left=0.042, top=1.0, right=0.998, wspace=0.4)
    plt.show()


def eval_sr(args):
    os.environ["SAPIEN_RENDER"] = "0"
    algo, task, robot_name, n_pc, src, n_ite, n_pi = args
    ep_horizon = 400
    try:
        env, f_exp = get_env(algo, task, robot_name, n_pc, src)
    except:
        print('Missing', algo, task, robot_name, n_pc, src)
        return None
    if n_pi is None:
        f_model = sorted(glob.glob(path.join(f_exp, 'policy_*.pickle')))[-1]
    else:
        f_model = path.join(f_exp, 'policy_{:04d}.pickle'.format(n_pi))
    policy = pickle.load(open(f_model, 'rb'))

    env.horizon = ep_horizon
    sr = 0
    for i in range(n_ite):
        state, done = env.reset()[0], False
        while True:
            action = policy.get_action(state)[1]['mean']
            state, reward, terminated, truncated, _ = env.step(action)
            if env.is_success():
                sr += 1
                break
            if terminated or truncated:
                break
    return sr


def sr_exp1(niter=50):
    args = []
    for algo in ['trpo', 'dapg']:
        for task in ['table_door', 'relocate_mustard_bottle', 'relocate_potted_meat_can', 'relocate_tomato_soup_can', 'mug_flip']:
            for robot_name in ['allegro_hand_right', 'schunk_hand_right', 'shadow_hand_right']:
                for n_pc in [0, 1, 2, 4]:
                    args.append((algo, task, robot_name, n_pc, None, niter, None))
    sr = Pool().map(eval_sr, args)
    for i in range(len(args)):
        print(args[i][:-1], 'sr={}/{}, {:.03f}'.format(sr[i], niter, sr[i] * 1.0 / niter))


def sr_exp2(n_pc=2, niter=50, n_pis=(100, 200, 300, 400)):
    args = []
    for task in ['table_door', 'relocate_mustard_bottle', 'relocate_potted_meat_can', 'relocate_tomato_soup_can', 'mug_flip']:
        for robot_name in ['robotiq_2f85', 'kinova_3f_right', 'leap_hand_right']:
            for src in [None, 'robotiq_2f85', 'kinova_3f_right', 'leap_hand_right']:
                for n_pi in n_pis:
                    args.append(('trpo' if src is None else 'dapg', task, robot_name, n_pc, src, niter, n_pi))
    sr = Pool().map(eval_sr, args)

    # print results
    sr = np.array(sr).reshape((-1, len(n_pis)))
    for i in range(sr.shape[0]):
        print(args[i * len(n_pis)][:-2], sr[i], 'sr: {:.03f}'.format(np.mean(sr[i]) / niter))


def main():
    task = sys.argv[1]
    if task == 'vis':
        algo = sys.argv[2]
        task = sys.argv[3]
        robot_name = sys.argv[4]
        n_pc = int(sys.argv[5])
        src = sys.argv[6]
        env = get_env(algo, task, robot_name, n_pc, src)
        if env:
            vis_rollout(*env)

    elif task == 'plot_exp1':
        plot_exp1()

    elif task == 'plot_exp2':
        plot_exp2()

    elif task == 'plot_exp22':
        plot_exp22()

    # elif task == 'plot_exp3':
    #     plot_exp3()

    elif task == 'sr_exp1':
        sr_exp1()

    elif task == 'sr_exp2':
        sr_exp2()

    # elif task == 'sr_exp3':
    #     sr_exp3()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    main()

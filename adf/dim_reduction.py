"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import torch
import os, sys
import numpy as np
import os.path as path
from pytorch3d.ops import sample_farthest_points
sys.path.append(os.path.dirname(__file__))
from manipulator import Manipulator
from mano_hand import ManoHand
from cvae import AnchorAE


def sample_manip(manip, n_sample, reuse_stats=False):
    """
    sample manipulator joint configuration uniformly between limits
    :param manip: manipulator instance
    :param n_sample: int - number of sample to collect
    :param reuse_stats: bool - reuse stats from manip instance; else stats from samples
    :return: tuple (dict, np.ndarray, np.ndarray)
                - anchor normalizing statistic
                - anchor position [n_sample, 22, 3]
    """
    print('Collecting anchors from {} with {} dof...'.format(manip.name, manip.dof))

    # sampling
    q_rnd = torch.rand((n_sample * np.clip(manip.dof, 2, 8), manip.dof))
    idx = sample_farthest_points(q_rnd[None], K=n_sample)[1][0]
    q_rnd = q_rnd[idx].numpy()

    # collecting
    anchors = []
    for q in q_rnd:
        manip.forward_kinematic(manip.denormalize_joint(q))
        anchors.append(manip.get_anchor())
    anchors = np.array(anchors)  # [n 22 3]

    # normalizing stats
    if reuse_stats:
        means = manip.stats['means']
        stds = manip.stats['stds']
    else:
        means = anchors.mean()
        stds = anchors.std()
    stats = {'means': means.copy(), 'stds': stds.copy()}
    anchors = (anchors - means) / stds
    return stats, anchors


def sample_mano(hand, n_sample, reuse_stats=False):
    """
    sample mano-hand PCA configuration uniformly between [-1, 1]
    :param hand: mano-hand instance
    :param n_sample: int - number of sample to collect
    :param reuse_stats: bool - reuse stats from manip instance; else stats from samples
    :return: tuple (dict, np.ndarray, np.ndarray)
                - anchor normalizing statistic
                - anchor position [n_sample, 22, 3]
    """
    print('Collecting anchors from mano hand with {} dof...'.format(hand.n_comp))

    # sampling
    q_rnd = torch.rand((n_sample * 10, hand.n_comp))
    idx = sample_farthest_points(q_rnd[None], K=n_sample)[1][0]
    q_rnd = q_rnd[idx].numpy()

    # collecting
    anchors = []
    for q in q_rnd:
        pose = np.hstack((np.zeros(3), q * 2. - 1.))
        shape = np.random.uniform(-0.5, 0.5, 10)
        anc = hand.pose_to_anchor(pose * 2, shape)
        anchors.append(anc)
    anchors = np.array(anchors)

    # normalizing stats
    if reuse_stats:
        means = hand.stats['means']
        stds = hand.stats['stds']
    else:
        means = anchors.mean()
        stds = anchors.std()
    stats = {'means': means.copy(), 'stds': stds.copy()}
    anchors = (anchors - means) / stds
    return stats, anchors


def fit_pca(reuse_model=False, n_sample=10000, num_epoch=2000):
    """
    collect manipulator and hand anchor position, and fit PCA
    :param reuse_model: bool - reuse trained model
    :param n_sample: int - number of sample to collect
    :return: None
    """
    # manipulators
    manips = [Manipulator(name_manip) for name_manip in Manipulator.names]
    hand = ManoHand()

    # collection
    anchors = []
    names = []
    stats = {}
    labels = []

    # collect manipulator
    for i, manip in enumerate(manips):
        stat_, anchors_ = sample_manip(manip, n_sample)
        anchors.append(anchors_)
        stats[manip.name] = stat_
        names.append(manip.name)
        labels.extend([i] * n_sample)

    # collect hand
    stat_, anchors_ = sample_mano(hand, n_sample)
    anchors.append(anchors_)
    stats['mano_hand'] = stat_
    names.append('mano_hand')
    labels.extend([len(manips)] * n_sample)

    # train CVAE+PCA
    print('Computing PCA...')
    anchors = np.stack(anchors, axis=0).reshape((-1, 22 * 3))
    print('PCA-input size:', anchors.shape)
    pca = AnchorAE(labels=names, reuse_model=reuse_model, num_epoch=num_epoch)
    pca.train(anchors, labels)
    f_model = path.join(path.dirname(__file__), 'pca.pth')
    pca.save_dict(f_model)
    f_stats = path.join(path.dirname(__file__), 'stats.npy')
    np.save(f_stats, stats)
    print('Saved PCA.')


def vis_pca():
    """
    visualize pca pose on manipulators
    :return: None
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TKAgg')

    # model
    pca = AnchorAE(load_model='pca.pth')

    # manipulator
    manips = [Manipulator(name_manip) for name_manip in Manipulator.names]
    hand = ManoHand()

    # collect configs
    n_sample = 50
    anchors = []
    for i, manip in enumerate(manips):
        anchors.append(sample_manip(manip, n_sample, reuse_stats=True)[1])
    anchors.append(sample_mano(hand, n_sample, reuse_stats=True)[1])
    anchors = np.concatenate(anchors, axis=0)

    # plot anchors
    anchors_ = anchors.reshape((-1, 3))
    lbl = np.arange(len(manips) + 1).repeat(n_sample * 22)
    rnd = np.random.rand(anchors_.shape[0]) > 0.5
    fig = plt.figure()
    p1 = fig.add_subplot(1, 2, 1, projection='3d')
    cmap = p1.scatter(anchors_[rnd, 0], anchors_[rnd, 1], anchors_[rnd, 2], c=lbl[rnd])
    p1.set_xlabel('X')
    p1.set_ylabel('Y')
    p1.set_zlabel('Z')
    p1.set_title('Anchors Position')

    # plot PCs
    anchors_ = anchors.reshape((-1, 22 * 3))
    lbl = np.arange(len(manips) + 1).repeat(n_sample)
    pc = pca.transform(anchors_, lbl.tolist())
    print('PCA-pc size:', pc.shape)
    p2 = fig.add_subplot(1, 2, 2, projection='3d')
    p2.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker='o', c=lbl)
    p2.set_xlabel('1st PC')
    p2.set_ylabel('2nd PC')
    p2.set_zlabel('3rd PC')
    p2.set_title('First 3 PCs')
    p2.set_xlim(-4, 4)
    p2.set_ylim(-4, 4)
    p2.set_zlim(-4, 4)
    plt.colorbar(cmap, ax=[p1, p2], label='Type of Manipulator',
                 orientation="horizontal", shrink=0.3, anchor=(0.5, 0))
    plt.tight_layout()
    plt.show()

    # vis pc
    for sign in [1, -1]:
        pc = np.zeros(10)
        pc[0] = sign * 3
        print('PC-sample:', pc)
        for i, manip in enumerate(manips):
            anchor = manip.pc_to_anchor(pc)
            manip.inverse_kinematic(anchor)
            manip.vis_model()
        pose, shape, _ = hand.anchor_to_pose(hand.pc_to_anchor(pc))
        hand.vis_model(pose=pose, shape=shape)


if __name__ == "__main__":
    # fit_pca()
    vis_pca()

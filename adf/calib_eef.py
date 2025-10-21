"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import yaml
import torch
import numpy as np
from os import path
from mano_hand import ManoHand
from dim_reduction import fit_pca
from manipulator import Manipulator
import matplotlib.pyplot as plt
from collections import defaultdict
from pytransform3d.transformations import transform_from, vectors_to_points, transform
from pytransform3d.rotations import euler_from_matrix, matrix_from_euler, axis_angle_from_matrix, matrix_from_axis_angle


n_pc = 10
n_cycle = 8
name_ref = ['robotiq_2f85', 'google_gripper', 'kinova_3f_right', 'armar_hand_right']


def clear_calib():
    calib = defaultdict(dict)
    for name in Manipulator.names:
        calib[name] = dict(tx=0, ty=0, tz=0, rx=0, ry=0, rz=0)
    calib['mano_hand'] = dict(tx=0, ty=0, tz=0, rx=0, ry=0, rz=0)
    yaml.safe_dump(dict(calib), open(path.join(path.dirname(__file__), 'calib_eef.yaml'), 'w'))


def ls_fit(p_a, p_b, clip=(0.5, 0.06)):
    """least-square-fit between p_a and p_b, assumed direct row-to-row correspondences
    source: https://github.com/ClayFlannigan/icp/blob/master/icp.py

    Args:
        p_a np.array [a, 3]: reference points
        p_b np.array [a, 3]: moving points
        threshold list [2] : [th_rotation(rad), th_translation(m)]:

    Return:
        [tx, ty, tz, rx, ry, rz]: for p_a
    """
    assert p_a.shape == p_b.shape

    # anchor weight
    s = np.ones(22)
    for idx in [0, 1, 2, 3]:  # thumb
        s[idx] += 3.0
    for idx in [3, 7, 11, 15, 19]:  # tips
        s[idx] += 4.0
    for idx in [20, 21]:  # palm
        s[idx] -= 0.5
    s = (s / s.sum()) * 22
    s = np.diag(np.tile(s, (p_a.shape[0] // 22,)))

    # fitting
    u_a = np.mean(p_a, axis=0, keepdims=True)
    u_b = np.mean(p_b, axis=0, keepdims=True)
    w = np.matmul(np.matmul((p_a - u_a).T, s), p_b - u_b)
    u, _, v = np.linalg.svd(w)
    rot = np.dot(v.T, u.T)
    if np.linalg.det(rot) < 0:
       v[2, :] *= -1
       rot = np.dot(v.T, u.T)
    tsl = (u_b.T - np.dot(rot, u_a.T)).squeeze()
    
    # clip rot
    axisang = axis_angle_from_matrix(rot)
    axisang[3] = np.clip(axisang[3], 0, clip[0])
    rot = matrix_from_axis_angle(axisang)
    # clip tsl
    tsl_norm = np.linalg.norm(tsl)
    tsl = tsl / tsl_norm * min(tsl_norm, clip[1])
    return transform_from(rot, tsl), (axisang[3], min(tsl_norm, clip[1]))


def calib_to_tf(calib):
    tf = transform_from(matrix_from_euler((calib['rx'], calib['ry'], calib['rz']), 0, 1, 2, True),
                        (calib['tx'], calib['ty'], calib['tz']))
    return tf


def tf_to_calib(tf):
    euler = euler_from_matrix(tf[:3, :3], 0, 1, 2, True)
    return dict(tx=tf[0, 3].item(), ty=tf[1, 3].item(), tz=tf[2, 3].item(),
                rx=euler[0].item(), ry=euler[1].item(), rz=euler[2].item())


def vis_calib(p_a, p_b, tf):
    """visualize calibration off p_a with tf into p_b

    Args:
        p_a (np.array [a, 3]): anchor position to be calibrated
        p_b (np.array [a, 3]): anchor position of reference
        tf (np.array [4, 4]): transformation for p_a
    """
    fig = plt.figure()
    p1 = fig.add_subplot(1, 2, 1, projection='3d')
    p1.scatter(p_a[:, 0], p_a[:, 1], p_a[:, 2], marker='^', c=Manipulator.colors, s=200)
    p1.scatter(p_b[:, 0], p_b[:, 1], p_b[:, 2], marker='o', c=Manipulator.colors, s=200)
    p1.set_xlabel('X')
    p1.set_ylabel('Y')
    p1.set_zlabel('Z')
    p1.set_title('Before')

    p2 = fig.add_subplot(1, 2, 2, projection='3d')
    p_a_ = transform(tf, vectors_to_points(p_a))[:, :3]
    p2.scatter(p_a_[:, 0], p_a_[:, 1], p_a_[:, 2], marker='^', c=Manipulator.colors, s=200)
    p2.scatter(p_b[:, 0], p_b[:, 1], p_b[:, 2], marker='o', c=Manipulator.colors, s=200)
    p2.set_xlabel('X')
    p2.set_ylabel('Y')
    p2.set_zlabel('Z')
    p2.set_title('After')

    plt.tight_layout()
    plt.show()
    return


def compute_calib(names_mfy, vis=False, clip=(0.5, 0.06)):
    # manipulators
    manip_ref = [Manipulator(name) for name in name_ref]
    manip_mfy = [Manipulator(name) for name in names_mfy]
    calibs = yaml.safe_load(open(path.join(path.dirname(__file__), 'calib_eef.yaml'), 'r'))

    # get reference anchor at open and close cfg
    pc_ref, anc_ref = [], []
    for sign in np.linspace(-1, 1, 8, endpoint=True):
        pc = np.zeros(n_pc)
        pc[0] = sign * 2.5
        anchors = []
        for manip in manip_ref:
            anc = manip.pc_to_anchor(pc)
            manip.reset()
            manip.inverse_kinematic(anc)
            anchors.append(manip.get_anchor())
        pc_ref.append(pc)
        anc_ref.append(np.stack(anchors).mean(axis=0))  # [n, a, 3] -> [a, 3]
    anc_ref = np.concatenate(anc_ref)  # [n, a, 3] -> [a*n, 3]
    
    # calib manips
    calib_vals = []
    for _, manip in enumerate(manip_mfy):
        ancs = []
        for pc in pc_ref:
            anc = manip.pc_to_anchor(pc)
            manip.reset()
            manip.inverse_kinematic(anc)
            anc = manip.get_anchor()
            ancs.append(anc)
        ancs = np.concatenate(ancs)
        # least square fit
        tf, calib_val = ls_fit(ancs, anc_ref, clip=clip)
        calib_vals.append(calib_val)
        if vis:
            print(manip.name, tf_to_calib(tf))
            vis_calib(np.array(anc), anc_ref, tf)
        tf = np.matmul(tf, calib_to_tf(calibs[manip.name]))
        calibs[manip.name] = tf_to_calib(tf)

    mano_mfy = ManoHand()
    # calib for mano
    ancs = []
    for pc in pc_ref:
        anc = mano_mfy.pc_to_anchor(pc)
        pose, _, _ = mano_mfy.anchor_to_pose(anc)
        anc = mano_mfy.pose_to_anchor(pose)
        ancs.append(anc)
    ancs = np.concatenate(ancs)
    tf, calib_val = ls_fit(ancs, anc_ref, clip=clip)
    calib_vals.append(calib_val)
    if vis:
        print('mano_hand', tf_to_calib(tf))
        vis_calib(anc, anc_ref, tf)
    tf = np.matmul(tf, calib_to_tf(calibs['mano_hand']))
    calibs['mano_hand'] = tf_to_calib(tf)

    # save calib
    yaml.safe_dump(calibs, open(path.join(path.dirname(__file__), 'calib_eef.yaml'), 'w'))

    return np.mean(list(zip(*calib_vals)), axis=1)


def main():
    torch.manual_seed(0)
    clear_calib()
    calib_vals = []
    num_epoch = 1500
    for i in range(n_cycle):
        # train and align
        fit_pca(reuse_model=(i > 0), num_epoch=int(num_epoch * (0.9 ** i)))
        calib_vals.append(compute_calib(names_mfy=Manipulator.names))

        # print result
        print('########################################################')
        print('################## Iteration: {:02d} / {:02d} ##################'.format(i + 1, n_cycle))
        print('############## Average calib rot: {:.5f} ##############'.format(calib_vals[-1][0]))
        print('############## Average calib tsl: {:.5f} ##############'.format(calib_vals[-1][1]))
        print('########################################################')

    # last fitting
    fit_pca(reuse_model=True, num_epoch=1500)
    # print result
    for i, val in enumerate(calib_vals):
        print('ite: {:02d}, calib_rot: {:.04f}, calib_tsl: {:.04f}'.format(i + 1, val[0], val[1]))


if __name__ == "__main__":
    main()

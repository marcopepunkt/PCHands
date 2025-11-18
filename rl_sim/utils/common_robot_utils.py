"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import xacro
import tempfile
import numpy as np
from os import path
from mplib import Planner
from pytransform3d.rotations import matrix_from_quaternion, matrix_from_euler


def load_robot(scene, robot_name):
    loader = scene.create_urdf_loader()
    try:
        _ = scene.get_render_system().device.name
    except:
        render_enabled = False
    else:
        render_enabled = True
    
    # get urdf
    dir_urdf = path.join(path.dirname(__file__), '../../assets', robot_name)
    urdf = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', dir=dir_urdf)
    urdf.write(xacro.process_file(path.join(dir_urdf, 'model_sapien.urdf')).toprettyxml(indent='    '))
    urdf.flush()
    robot_builder = loader.load_file_as_articulation_builder(urdf.name)#, no_visual=not render_enabled)
    urdf.close()

    # disable self-collision
    for link_builder in robot_builder.link_builders:
        link_builder.collision_groups = [1, 1, 17, 0]

    robot = robot_builder.build(fix_root_link=True)
    robot.set_name('robot_{}'.format(robot_name))

    mat = scene.create_physical_material(1, 1, 0)
    for link in robot.get_links():
        for geom in link.get_collision_shapes():
            geom.min_patch_radius = 0.1
            geom.patch_radius = 0.1
            geom.set_physical_material(mat)

    return robot


def load_planner(robot_name, eef_link):
    dir_urdf = path.join(path.dirname(__file__), '../../assets', robot_name)
    urdf = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', dir=dir_urdf)
    urdf.write(xacro.process_file(path.join(dir_urdf, 'model_mp.urdf')).toprettyxml(indent='    '))
    urdf.flush()
    planner = Planner(urdf=urdf.name, srdf=path.join(dir_urdf, 'model_mp.srdf'), move_group=eef_link)
    urdf.close()
    return planner


def sixd2mat(sixd):
    """
    https://arxiv.org/pdf/1812.07035
    :param sixd: np.array [6]
    :return: np.array [3, 3]
    """
    rot = np.eye(3)
    sixd = sixd.reshape(3, 2)
    rot[:, 0] = sixd[:, 0] / np.linalg.norm(sixd[:, 0])
    rot[:, 1] = sixd[:, 1] / np.linalg.norm(sixd[:, 1])
    rot[:, 2] = np.cross(rot[:, 0], rot[:, 1])
    return rot


def encode_pose(pose):
    """
    convert pose to sixd format
    :param pose: list or np.array: (p+q)/(p+euler)/tf
    :return: np.array [9] tsl(3)+sixd(6)
    """
    if isinstance(pose, list):
        pose = np.array(pose)
    if pose.shape == (7,):
        # qua: x,y,z,rw,rx,ry,rz
        tsl = pose[:3]
        sixd = matrix_from_quaternion(pose[3:])[:3, :2].flatten()
    elif pose.shape == (6,):
        # euler: x,y,z,ex,ey,ez
        tsl = pose[:3]
        sixd = matrix_from_euler(pose[3:], 0, 1, 2, True)[:3, :2].flatten()
    elif pose.shape == (4, 4):
        tsl = pose[:3, 3]
        sixd = pose[:3, :2].flatten()
    else:
        raise NotImplementedError('Unknown pose type')
    return np.hstack((tsl, sixd))


def decode_pose(pose):
    """
    decode sixd pose to transformation matrix
    :param pose: np.array [9]
    :return: np.array [4, 4]
    """
    tf = np.eye(4)
    tf[:3, 3] = pose[:3]
    tf[:3, :3] = sixd2mat(pose[3:])
    return tf

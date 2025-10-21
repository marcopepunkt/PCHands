"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
Based on code from https://github.com/yzqin/dex-hand-teleop with MIT licence
"""
import sys
import yaml
import numpy as np
from os import path
from sapien.core import Pose
from mplib import Pose as mpPose
from transforms3d.quaternions import mat2quat
from transforms3d.euler import euler2quat, axangle2euler
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import transform_from
sys.path.append(path.join(path.dirname(__file__), '../..'))
from rl_sim.utils.model_utils import (build_free_root, build_ball_joint, fix_link_inertia,
                                      rot_from_connected_link, create_visual_material)
from rl_sim.utils.common_robot_utils import encode_pose, load_planner
from adf.mano_hand import ManoHand


PALM_THICKNESS = 0.011
FINGER_RADIUS = 0.01
INVERSE_ALONG_AXIS = 0
INVERSE_FACE_AXIS = 1
LITTLE_TO_THUMB_AXIS = 2


class MANORobotHand:
    def __init__(self, scene, init_joint_pos, n_pc=10, ik_link='eef_calib_link', frame_skip=5):
        # Create robot
        self.scene = scene
        self.frame_skip = frame_skip
        self.manoh = ManoHand()
        self.robot = create_mano_bot(scene, init_joint_pos)
        self.robot.set_qpos(np.zeros(self.robot.dof))
        self.n_pc = n_pc
        self.planner = load_planner('mano_hand', ik_link)

        # robot Dynamics
        joints = self.robot.get_active_joints()
        for i in range(3):
            joints[i].set_drive_property(stiffness=100, damping=30, force_limit=50)
        for i in range(3, 6):
            joints[i].set_drive_property(stiffness=100, damping=30, force_limit=50)
        for i in range(6, self.robot.dof):
            joints[i].set_drive_property(stiffness=50, damping=3, force_limit=4)

    def get_qpospc(self):
        qpos = self.robot.get_qpos()
        # finger pc
        anchors = np.array([self.robot.find_link_by_name('A_{:02d}'.format(i)).get_entity_pose().get_p()
                            for i in range(23)])
        pc = self.manoh.anchor_to_pc(self.manoh.anchor_transform(anchors))[:self.n_pc]
        return np.concatenate((qpos[:6], pc))

    def get_posepc(self):
        anchors = np.array([self.robot.find_link_by_name('A_{:02d}'.format(i)).get_entity_pose().get_p()
                            for i in range(23)])
        # eef pose
        pose = self.manoh.anchor_transform(anchors, return_tf=True)
        pose = encode_pose(pose)
        # finger pc
        pc = self.manoh.anchor_to_pc(self.manoh.anchor_transform(anchors))[:self.n_pc]
        return np.hstack((pose, pc))

    def mocap_to_posepc(self, data, w2cam):
        # stay in mano qpos format, without pose synergy
        return self.mocap_to_qpos(data, w2cam)

    @staticmethod
    def mano_to_qpos(pose_param):
        if pose_param.size != 45:
            raise ValueError(f"pose_param should be in shape of 45")
        smplx_to_panoptic = np.array([12, 13, 14, 0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8])
        pose_param = pose_param.reshape([15, 3])[smplx_to_panoptic, :]
        qpos = []
        for i in range(15):
            vec = pose_param[i]
            angle = np.linalg.norm(vec)
            if np.isclose(angle, 0):
                qpos.append(np.zeros(3))
            else:
                axis = vec / angle
                euler = axangle2euler(axis, angle, "rxyz")
                qpos.append(euler)

        qpos = np.concatenate(qpos)
        idx_reorder = [0,  9, 18, 27, 36,
                       1, 10, 19, 28, 37,
                       2, 11, 20, 29, 38,
                       3, 12, 21, 30, 39,
                       4, 13, 22, 31, 40,
                       5, 14, 23, 32, 41,
                       6, 15, 24, 33, 42,
                       7, 16, 25, 34, 43,
                       8, 17, 26, 35, 44]
        qpos = qpos[idx_reorder]
        return qpos

    def mocap_to_qpos(self, data, w2cam):
        # root qpos
        anchor = self.manoh.vertex_to_anchor(data['vertices']).copy()
        anchor = anchor + data['offset']
        anchor = np.matmul(w2cam, np.vstack((anchor.T, np.ones((1, anchor.shape[0]))))).T[:, :-1]
        w2eef = self.manoh.anchor_transform(anchor, return_tf=True)
        w2root = np.matmul(w2eef, self.manoh.calib_eef)
        _, root_qpos = self.planner.IK(goal_pose=mpPose(w2root), return_closest=True,
                                       start_qpos=self.robot.get_qpos()[:6])
        if root_qpos is None:
            root_qpos = self.robot.get_qpos()[:6]

        # finger qpos
        finger_qpos = self.mano_to_qpos(data["pose_params"][3:])
        return np.concatenate((root_qpos, finger_qpos))

    def control_robot(self, target_qpos, locked=False):
        if locked:
            target_qpos[6:] = [joint.get_drive_target().flatten()[0] for joint in self.robot.get_active_joints()[6:]]
        for idx, joint in enumerate(self.robot.get_active_joints()):
            joint.set_drive_target(target_qpos[idx])


def create_mano_bot(scene, joint_pos):
    finger_viz = dict(specular=0.5, metallic=0.0, roughness=0.1, base_color=(0.1, 0.1, 0.1, 1))
    tip_viz = dict(specular=0.07, metallic=0.2, roughness=0.5, base_color=(0.9, 0.9, 0.9, 1))

    # Compute shape related params
    joint_pos = joint_pos - joint_pos[:1, :]
    finger_palm_width = np.abs(joint_pos[17, LITTLE_TO_THUMB_AXIS] - joint_pos[5, LITTLE_TO_THUMB_AXIS]) / 3

    # Build palm and four palm finger geom
    mat = scene.create_physical_material(1.0, 1.0, 0.01)
    friction_dict = {"material": mat, "patch_radius": 0.04, "min_patch_radius": 0.02}
    robot_builder = scene.create_articulation_builder()
    palm = _create_palm(robot_builder,
                        eef_pose=Pose([-PALM_THICKNESS, np.abs(joint_pos[5 + 4, INVERSE_ALONG_AXIS]) / 1.5, 0],
                                      euler2quat(0, 0, np.pi * 0.5)))

    palm_half_width = finger_palm_width / 2
    palm_viz_mat = create_visual_material(**finger_viz)
    use_visual = True

    # build palm
    along_axis_sign = 1
    for i in range(4):
        finger_palm_length = np.abs(joint_pos[5 + 4 * i, INVERSE_ALONG_AXIS])
        pos = np.array([-finger_palm_length * along_axis_sign / 2,
                        0, joint_pos[5, LITTLE_TO_THUMB_AXIS] - i * finger_palm_width])
        if use_visual:
            palm.add_box_visual(pose=Pose(pos), material=palm_viz_mat,
                                half_size=np.array([finger_palm_length / 2, PALM_THICKNESS, palm_half_width]))
        palm.add_box_collision(pose=Pose(pos),
                               half_size=np.array([finger_palm_length / 2, PALM_THICKNESS, finger_palm_width / 2]),
                               **friction_dict)
        if i == 0 or i == 3:
            pos = np.array([-finger_palm_length * along_axis_sign / 2 - finger_palm_length * along_axis_sign / 2 * 0.5,
                            -PALM_THICKNESS, joint_pos[5, LITTLE_TO_THUMB_AXIS] - i * finger_palm_width])
            _create_anchor(robot_builder, palm, "A_{:02d}".format(20 if i == 0 else 21), Pose(pos), Pose())
        if i == 1:
            pos = np.array([-finger_palm_length * along_axis_sign / 2 + finger_palm_length * along_axis_sign / 2 * 0.5,
                            -PALM_THICKNESS, joint_pos[5, LITTLE_TO_THUMB_AXIS] - i * finger_palm_width])
            _create_anchor(robot_builder, palm, "A_22", Pose(pos), Pose())

    # Build fingers
    finger_names = ["thumb", "index", "middle", "ring", "little"]
    radius = FINGER_RADIUS
    for i in range(5):
        finger_name = finger_names[i]
        finger_viz_mat = create_visual_material(**finger_viz)
        tip_viz_mat = create_visual_material(**tip_viz)
        # Link 0
        if i == 0:
            bjoint_quat = euler2quat(0, 0, np.pi / 5)
        else:
            bjoint_quat = np.array([1., 0, 0, 0])
        _, _, link0 = build_ball_joint(robot_builder, palm, name=f"{finger_name}_0",
                                       pose=Pose(joint_pos[1 + 4 * i], bjoint_quat))
        pos_link1_in_link0 = joint_pos[2 + 4 * i] - joint_pos[1 + 4 * i]
        length_link1_in_link0 = np.linalg.norm(pos_link1_in_link0)
        quat_link1_in_link0 = mat2quat(rot_from_connected_link(joint_pos[1 + 4 * i], joint_pos[2 + 4 * i]))
        if use_visual:
            link0.add_capsule_visual(pose=Pose(pos_link1_in_link0 / 2, quat_link1_in_link0), radius=radius,
                                     half_length=length_link1_in_link0 / 2, material=finger_viz_mat)
        link0.add_capsule_collision(pose=Pose(pos_link1_in_link0 / 2, quat_link1_in_link0), radius=radius,
                                    half_length=length_link1_in_link0 / 2, **friction_dict)
        # anchor 0
        pos = [0, radius, 0]
        if i == 0:
            pos = [0, 0, radius]
        _create_anchor(robot_builder, link0, "A_{:02d}".format((i * 4) + 0),
                       Pose(pos_link1_in_link0 / 2, quat_link1_in_link0), Pose(pos))

        # Link 1
        if i == 0:
            bjoint_quat = euler2quat(np.pi / 5, 0, 0)
        else:
            bjoint_quat = np.array([1., 0, 0, 0])
        _, _, link1 = build_ball_joint(robot_builder, link0, name=f"{finger_name}_1",
                                       pose=Pose(pos_link1_in_link0, bjoint_quat))
        pos_link2_in_link1 = joint_pos[3 + 4 * i] - joint_pos[2 + 4 * i]
        length_link2_in_link1 = np.linalg.norm(pos_link2_in_link1)
        quat_link2_in_link1 = mat2quat(rot_from_connected_link(joint_pos[2 + 4 * i], joint_pos[3 + 4 * i]))
        if use_visual:
            link1.add_capsule_visual(pose=Pose(pos_link2_in_link1 / 2, quat_link2_in_link1), radius=radius,
                                     half_length=length_link2_in_link1 / 2, material=finger_viz_mat)
        link1.add_capsule_collision(pose=Pose(pos_link2_in_link1 / 2, quat_link2_in_link1), radius=radius,
                                    half_length=length_link2_in_link1 / 2, **friction_dict)
        # anchor 1
        pos = [0, radius, 0]
        if i == 0:
            pos = [0, 0, radius]
        _create_anchor(robot_builder, link1, "A_{:02d}".format((i * 4) + 1),
                       Pose(pos_link2_in_link1 / 2, quat_link2_in_link1), Pose(pos))

        # Link 2
        if i == 0:
            bjoint_quat = euler2quat(np.pi / 6, 0, 0)
        else:
            bjoint_quat = np.array([1., 0, 0, 0])
        _, _, link2 = build_ball_joint(robot_builder, link1, name=f"{finger_name}_2",
                                       pose=Pose(pos_link2_in_link1, bjoint_quat))
        finger_tip_middle = joint_pos[4 + 4 * i] + np.array([0, -1, 0]) * FINGER_RADIUS / 2
        pos_link3_in_link2 = finger_tip_middle - joint_pos[3 + 4 * i]
        quat_link3_in_link2 = mat2quat(rot_from_connected_link(joint_pos[3 + 4 * i], finger_tip_middle))
        length_link3_in_link2 = np.linalg.norm(pos_link3_in_link2)
        if use_visual:
            link2.add_capsule_visual(pose=Pose(pos_link3_in_link2 / 2, quat_link3_in_link2), radius=radius,
                                     half_length=length_link3_in_link2 / 2, material=tip_viz_mat)
        link2.add_capsule_collision(pose=Pose(pos_link3_in_link2 / 2, quat_link3_in_link2), radius=radius,
                                    half_length=length_link3_in_link2 / 2, **friction_dict)
        # anchor 2
        pos = [0, radius, 0]
        if i in [2, 3, 4]:
            pos = [0, 0, -radius]
        if i == 0:
            pos = [0, 0, radius]
        _create_anchor(robot_builder, link2, "A_{:02d}".format((i * 4) + 2),
                       Pose(pos_link3_in_link2 / 2, quat_link3_in_link2), Pose(pos))

        # Link 3
        link3 = robot_builder.create_link_builder(link2)
        link3.set_name(f"{finger_name}_tip_link")
        link3.set_joint_name(f"{finger_name}_tip_joint")
        link3.set_joint_properties("fixed", limits=np.array([]), pose_in_child=Pose(),
                                   pose_in_parent=Pose(joint_pos[4 + 4 * i] - joint_pos[3 + 4 * i]))
        # anchor 3
        pos = [length_link3_in_link2 / 2 + radius, 0, 0]
        _create_anchor(robot_builder, link2, "A_{:02d}".format((i * 4) + 3),
                       Pose(pos_link3_in_link2 / 2, quat_link3_in_link2), Pose(pos))

    for link_builder in robot_builder.link_builders:
        # link_builder.collision_groups = [0, 1, 2, 2]
        link_builder.collision_groups = [1, 1, 17, 0]
    fix_link_inertia(robot_builder)
    robot = robot_builder.build(fix_root_link=True)
    robot.set_name("mano_hand")
    return robot


def _create_anchor(robot_builder, parent, name, pose_in_parent, pose_in_child):
    anchor = robot_builder.create_link_builder(parent)
    anchor.set_name(name)
    anchor.set_joint_name(name + '_joint')
    # import sapien
    # mat = sapien.render.RenderMaterial()
    # mat.set_base_color(np.array([0, 1, 0, 1]))
    # anchor.add_sphere_visual(radius=0.003, material=mat)
    # anchor.add_sphere_collision(radius=0.003, density=0.001)
    anchor.set_joint_properties("fixed", limits=[], pose_in_parent=pose_in_parent, pose_in_child=pose_in_child)


def _create_palm(robot_builder, eef_pose):
    # eef calib
    eef_ori = build_free_root(robot_builder, rotate_final=False,
                          rotation_range=[-9., 9.], translation_range=[-3., 3.])
    eef_ori.set_name('eef_calib_link')

    # eef
    calib_eef = yaml.safe_load(open(path.join(path.dirname(__file__),
                                              '../../adf/calib_eef.yaml'), 'r'))['mano_hand']
    calib_eef = transform_from(
        matrix_from_euler((calib_eef['rx'], calib_eef['ry'], calib_eef['rz']), 0, 1, 2, True),
        (calib_eef['tx'], calib_eef['ty'], calib_eef['tz']))
    eef = robot_builder.create_link_builder(eef_ori)
    eef.set_joint_properties('fixed', pose_in_child=Pose(calib_eef), limits=[], pose_in_parent=Pose())
    eef.set_name("eef_link")

    # palm
    palm = robot_builder.create_link_builder(eef_ori)
    palm.set_joint_properties('fixed', pose_in_child=Pose(), limits=[], pose_in_parent=eef_pose)
    palm.set_name("palm_base_link")
    return palm

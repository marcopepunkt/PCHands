"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause

              [world]
    [camera]         [base]->[flange]->[root]->[eef]

Frames:
   1. world: table-top center; x-forward; z-up
   2. camera: base-rgb-camera; x-right; z-forward
   3. base: arm link0; x-forward, z-up
   4. flange: arm flange;
   5. root: hand_eef_base;
   6. eef: hand_eef;
"""
import sys
import cv2
import time
import numpy as np
from os import path
from realsense_cam import RealSenseCamera
from manip_robotiq_2f85 import ManipRobotiq2F85
from manip_leap_hand_right import ManipLeapHandRight
from franky import Robot, Affine, CartesianMotion, JointWaypointMotion, JointWaypoint
sys.path.append(path.join(path.dirname(__file__), '..'))
from rl_sim.utils.common_robot_utils import encode_pose, decode_pose


class PandaRobot:
    q_home = [0.8, -0.1, 0.0, -2.5, -1.2, 1.3, 1.0]

    def __init__(self, manip_name='robotiq_2f85', ip_robot='172.16.0.2', rel_dyna=0.05, n_pc=1,
                 hand_pid=(200, 0, 1000), rgb_hw=(480, 640)):
        # arm
        self.robot = Robot(ip_robot)
        self.robot.relative_dynamics_factor = rel_dyna

        # hand
        if manip_name == 'leap_hand_right':
            self.manip = ManipLeapHandRight(n_pc, pid=hand_pid)
        elif manip_name == 'robotiq_2f85':
            self.manip = ManipRobotiq2F85(n_pc)
        elif manip_name == 'franka_gripper':
            self.manip = ManipFrankaGripper(n_pc, ip_robot)
        else:
            print(manip_name, 'not implemented!')
            raise NotImplementedError
        # camera-base
        self.camera = RealSenseCamera(rgb_hw=rgb_hw)
        self.camera.start()
        # tfs
        dir_file = path.join(path.dirname(__file__), 'calib_base2cam.txt')
        try:
            self.base2cam = np.loadtxt(dir_file)
        except:
            self.base2cam = np.eye(4)
            print('Failed to load calib-base2cam file!')
        self.base2world = self.base2cam @ np.linalg.inv(self.camera.world2cam)
        self.world2base = np.linalg.inv(self.base2world)
        self.flange2eef = self.manip.flange2eef.copy()
        self.eef2flange = np.linalg.inv(self.flange2eef)
        dir_file = path.join(path.dirname(__file__), 'calib_eef.txt')
        try:
            self.calib_world_eef = np.loadtxt(dir_file)
        except:
            self.calib_world_eef = np.eye(4)
            print('Failed to load calib-base2cam file!')

    def home(self):
        self.robot.move(JointWaypointMotion([JointWaypoint(self.q_home)]), asynchronous=False)
        self.manip.open()
        self.move_eef(decode_pose(np.array([-0.3, 0, 0.15, 1, 0, 0, 1, 0, 0])), asynchronous=False)

    def close(self):
        self.robot.join_motion()
        self.robot.stop()
        self.camera.stop()
        self.manip.open()

    def post_calib(self, marker_id=8, marker_size=0.1):
        # move to world frame
        self.calib_world_eef = np.eye(4)
        self.home()
        self.move_eef(decode_pose(np.array([0, 0, 0.1, 1, 0, 0, 1, 0, 0])), asynchronous=False)
        # TODO: move marker under eef
        # get marker pose
        while True:
            cam2mkr = self.camera.get_marker_pose(marker_id, marker_size)
            if cv2.waitKey(1) == ord('q'):
                if cam2mkr is None:
                    print('retry...')
                else:
                    break
        # get diff
        self.calib_world_eef = self.camera.world2cam @ cam2mkr
        dir_file = path.join(path.dirname(__file__), 'calib_eef.txt')
        np.savetxt(dir_file, self.calib_world_eef)
        print('calib_eef')
        print(self.calib_world_eef)

    def get_eef(self):
        """
        get eef pose in world frame
        :return: np.ndarray [4, 4]
        """
        base2flange = self.robot.current_cartesian_state.pose.end_effector_pose.matrix
        world2eef = self.world2base @ base2flange @ self.flange2eef
        world2eef = world2eef @ self.calib_world_eef
        return world2eef

    def move_eef(self, world2eef, asynchronous=True):
        """
        set eef pose in world frame
        :param world2eef: np.ndarray [4, 4]
        :return:
        """
        world2eef = world2eef @ np.linalg.inv(self.calib_world_eef)
        base2flange = self.base2world @ world2eef @ self.eef2flange
        pose = Affine(base2flange)
        motion = CartesianMotion(pose)
        self.robot.move(motion, asynchronous=asynchronous)

    def get_posepc(self):
        pose = encode_pose(self.get_eef())
        pc = self.manip.get_pc()
        return np.hstack((pose, pc))

    def set_posepc(self, posepc):
        # move_eef
        pose = decode_pose(posepc[:9])
        self.move_eef(pose)
        # move_pc
        pc = posepc[9:]
        self.manip.set_pc(pc)


def hand_eye_calib(n_ite=16):
    # init robot
    robot = PandaRobot(rgb_hw=(720, 1280))
    time.sleep(2)

    # hold marker
    robot.manip.open()
    time.sleep(4)
    robot.manip.close()
    time.sleep(6)

    # posing
    flanges, mkrs = [], []
    while len(flanges) < n_ite:
        # move flange
        print('***********************')
        print('     Move {:02d}/{:02d}       '.format(len(flanges), n_ite))
        print('***********************')
        cv2.waitKey(5000)
        # record
        cam2mkr = robot.camera.get_marker_pose(marker_id=8, marker_size=0.1)
        if cam2mkr is not None:
            mkrs.append(np.linalg.inv(cam2mkr))
            flanges.append(robot.robot.current_cartesian_state.pose.end_effector_pose.matrix)

    # compute
    base2flange = np.array(flanges)
    mkrs2cam = np.array(mkrs)
    rt = cv2.calibrateHandEye(mkrs2cam[:, :3, :3], mkrs2cam[:, :3, 3],
                              base2flange[:, :3, :3], base2flange[:, :3, 3],
                              method=cv2.CALIB_HAND_EYE_TSAI)
    base2cam = np.eye(4)
    base2cam[:3, :3] = rt[0]
    base2cam[:3, 3] = rt[1].flatten()
    base2cam = np.linalg.inv(base2cam)
    dir_file = path.join(path.dirname(__file__), 'calib_base2cam.txt')
    np.savetxt(dir_file, base2cam)
    print('base2cam:')
    print(base2cam)
    robot.close()


if __name__ == "__main__":
    hand_eye_calib()

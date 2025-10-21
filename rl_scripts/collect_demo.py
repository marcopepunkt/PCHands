"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import os
import sys
import glob
import numpy as np
from os import path
from pytransform3d.rotations import quaternion_from_euler
from pytransform3d.transformations import transform_from_pq
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hand_monitor import ManoMocap
from rl_sim.player.recorder import DataRecorder
from rl_sim.kinematics.manipulator import ManipulatorRobot
from rl_sim.kinematics.mano_robot_hand import MANORobotHand
from rl_sim.env.mugflip_env import MugFlipEnv
from rl_sim.env.relocate_env import RelocateEnv
from rl_sim.env.tabledoor_env import TableDoorEnv
from rl_sim.gui.teleop_gui import GUIBase, DEFAULT_TABLE_TOP_CAMERAS


def main(task_name, robot_name):
    """
    :param task_name: ["table_door", "mug_flip", "relocate_{object_name}"],
         object_name: ['tomato_soup_can', "mustard_bottle", "potted_meat_can"]
    :return:
    """

    # file path
    demo_path = path.join(path.dirname(__file__), 'teleop_collection', task_name)
    os.makedirs(demo_path, exist_ok=True)
    demo_idx = len(glob.glob(path.join(demo_path, '*.pkl')))

    # sim env
    if "relocate" in task_name:
        env_dict = dict(object_name='_'.join(task_name.split('_')[1:]))
        env = RelocateEnv(**env_dict)
    elif task_name == "table_door":
        env_dict = dict()
        env = TableDoorEnv(**env_dict)
    elif task_name == "mug_flip":
        env_dict = dict()
        env = MugFlipEnv(**env_dict)
    else:
        raise NotImplementedError

    gui = GUIBase(env.scene)
    for name, params in DEFAULT_TABLE_TOP_CAMERAS.items():
        gui.create_camera(**params)
    gui.viewer.set_camera_rpy(0, -0.7, 0.01)
    gui.viewer.set_camera_xyz(-0.6, 0, 0.45)

    # real camera
    # w2cam = transform_from_pq([0.5, 0.5, 0.2] +
    #                           quaternion_from_euler((-np.pi / 2, 0, np.pi * 0.75), 0, 1, 2, True).tolist())
    w2cam = transform_from_pq([0.7, 0, 0.2] +
                              quaternion_from_euler((-np.pi / 2, 0, np.pi / 2), 0, 1, 2, True).tolist())
    mano_hand = ManoMocap()

    # init hand
    print('Initializing hand...')
    while not mano_hand.initialized:
        for _ in range(5):
            env.scene.step()
        _, motion_data = mano_hand.step()
        rgb = motion_data["rgb"]
        gui.render(additional_views=[rgb[..., ::-1]])
    print('Hand initialized')

    # robot
    if robot_name == 'customized':
        zero_joint_pos = mano_hand.compute_hand_zero_pos()
        robot = MANORobotHand(env.scene, init_joint_pos=zero_joint_pos, ik_link='eef_calib_link')
    else:
        zero_joint_pos = 0
        robot = ManipulatorRobot(env.scene, robot_name, teleop=True, ik_link='eef_link', dps=(6, 12, 10))

    # keyboard interface
    recorder = None
    recording = False
    locking = False

    def exit_teleop():
        mano_hand.camera.stop()
        gui.viewer.close()
        print('<exit>')

    def reset_teleop():
        nonlocal recorder, recording
        print('<reset>')
        env.reset_env()
        recorder = None
        recording = False
        gui.viewer.set_camera_rpy(0, -0.7, 0.01)
        gui.viewer.set_camera_xyz(-0.6, 0, 0.45)
        qs = np.hstack((env.palm_init, np.zeros(robot.robot.get_qpos().shape[0] - 6)))
        robot.robot.set_qpos(qs)

    def toggle_recording():
        nonlocal recording, recorder
        recording = not recording
        if recording:
            print('<recording>')
            recorder = DataRecorder(filename=path.join(demo_path, '{:03d}.pkl'.format(demo_idx)), scene=env.scene)
        else:
            print('<idling>')

    def save_collected():
        nonlocal demo_idx, recording
        if recording:
            print('Stop recording before saving')
            return
        if recorder is None:
            print('Record before save')
            return

        print('<saved>', len(recorder), 'steps')
        if len(recorder):
            meta_data = dict(env_class=env.__class__.__name__, env_kwargs=env_dict,
                             shape_param=mano_hand.calibrated_shape_params,
                             robot_name=robot_name, zero_joint_pos=zero_joint_pos)
            recorder.dump(meta_data)
            print('saving {}'.format(recorder.filename))
            demo_idx += 1
            recorder.clear()
        else:
            print('nothing to save')

    def toggle_locked():
        nonlocal locking
        locking = not locking
        if locking:
            print('<locked>')
        else:
            print('<unlocked>')

    gui.register_keydown_action('q', exit_teleop)
    gui.register_keydown_action('a', reset_teleop)
    gui.register_keydown_action('z', toggle_locked)
    gui.register_keydown_action('x', toggle_recording)
    gui.register_keydown_action('c', save_collected)

    # collection loop
    while not gui.closed:
        # skip frame
        for _ in range(5):
            env.scene.step()
        # capture motion
        success, motion_data = mano_hand.step()
        rgb = motion_data["rgb"]
        gui.render(additional_views=[rgb[..., ::-1]])

        # move
        if success:
            robot_posepc = robot.mocap_to_posepc(motion_data, w2cam)
            robot.control_robot(robot_posepc, locked=locking)

        # record
        if recording:
            record_data = motion_data.copy()
            record_data.update({"success": success})
            record_data.pop("rgb")
            recorder.step(record_data)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    main(sys.argv[1], sys.argv[2])

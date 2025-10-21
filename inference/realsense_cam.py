"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import cv2
import time
import json
import numpy as np
from os import path
import pyrealsense2 as pyrs
from threading import Thread


class RealSenseCamera:
    def __init__(self, rgb_hw=(480, 640), depth_hw=(480, 640), fps=30):
        self.rgb_hw, self.fps = rgb_hw, fps

        # option
        f_cfg = path.join(path.dirname(__file__), 'cfg_rs.json')
        cfg_rs = json.load(open(f_cfg, 'r'))
        depth_sensor = pyrs.context().query_devices()[0].first_depth_sensor()
        for key in cfg_rs:
            opt = getattr(pyrs.option, '_'.join(key.lower().split(' ')))
            try:
                depth_sensor.set_option(opt, cfg_rs[key])
            except:
                pass

        # config
        cfg = pyrs.config()
        cfg.enable_stream(pyrs.stream.color, rgb_hw[1], rgb_hw[0], pyrs.format.bgr8, self.fps)
        cfg.enable_stream(pyrs.stream.depth, depth_hw[1], depth_hw[0], pyrs.format.z16, self.fps)
        self.align = pyrs.align(pyrs.stream.color)
        self.stream = pyrs.pipeline()
        dev = self.stream.start(cfg)
        cam_int = dev.get_stream(pyrs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_scale = dev.get_device().first_depth_sensor().get_depth_scale()
        self.depth_clip = (0, 3.0)
        self.cam_k = np.array([[cam_int.fx, 0, cam_int.ppx],
                               [0, cam_int.fy, cam_int.ppy],
                               [0, 0, 1]])
        self.frame = None
        self.running = False
        try:
            dir_file = path.join(path.dirname(__file__), 'calib_world2cam.txt')
            self.world2cam = np.loadtxt(dir_file)
        except:
            print('No camera extrinsic calibration data found!')
            self.world2cam = np.eye(4)

        # marker detection
        self.mker_dtor = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
            cv2.aruco.DetectorParameters())

    def grab_frame(self):
        # fetch frames
        frames = self.align.process(self.stream.wait_for_frames())
        color = np.asarray(frames.get_color_frame().get_data())
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = np.asarray(frames.get_depth_frame().get_data())
        depth = np.clip(depth * self.depth_scale, *self.depth_clip)
        return color, depth

    def start(self):
        if self.running:
            print('Camera already started!')
            return
        self.running = True
        Thread(target=self.update, args=()).start()

    def update(self):
        while True:
            if not self.running:
                return
            self.frame = self.grab_frame()

    def read(self):
        if not self.running:
            print('Start camera before reading!')
            return
        while self.frame is None:
            time.sleep(0.03)
        return self.frame

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()

    def get_marker_pose(self, marker_id, marker_size, retry=16):
        """
        get camera to marker pose in [4, 4]
        :param marker_id: int
        :param marker_size: float
        :param retry: int
        :return: np.ndarray [4, 4] camera to marker transformation;
                  or None if failed
        """
        mker_pnts = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        cam2mkr = None
        for _ in range(retry):
            color, _ = self.read()
            corners, mids, _ = self.mker_dtor.detectMarkers(color)
            if mids is not None:
                for (corner, mid) in zip(corners, mids):
                    if mid == marker_id:
                        _, R, t = cv2.solvePnP(mker_pnts, corner, self.cam_k,
                                               None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        cv2.drawFrameAxes(color, self.cam_k, 0, R, t, 0.1)
                        cv2.aruco.drawDetectedMarkers(color, corners, mids)
                        cam2mkr = np.eye(4)
                        cam2mkr[:3, :3] = cv2.Rodrigues(R)[0]
                        cam2mkr[:3, 3] = t.flatten()
                        break
            cv2.imshow('Marker Detection', color)
            if cam2mkr is None:
                break
        return cam2mkr

    def calib_ext(self, marker_id=8, marker_size=0.1):
        """
        press key 'q' to save pose
        :param marker_id: int
        :param marker_size: float
        :return:
        """
        # start camera
        if not self.running:
            self.start()
        # loop
        while True:
            cam2w = self.get_marker_pose(marker_id, marker_size)
            if cv2.waitKey(1) == ord('q'):
                break
        # stop camera
        self.stop()

        # result cam2w
        if cam2w is None:
            print('No calibration done!')
            return
        self.world2cam = np.linalg.inv(cam2w)

        # saving
        dir_file = path.join(path.dirname(__file__), 'calib_world2cam.txt')
        np.savetxt(dir_file, self.world2cam)
        print('world2cam:')
        print(self.world2cam)

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()


def test_frame():
    # start camera
    cam = RealSenseCamera()
    cam.start()
    # loop
    while True:
        # get frame
        color, depth = cam.read()
        cv2.imshow('color', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == ord('q'):
            break
    cam.stop()


def calib_cam_ext():
    cam = RealSenseCamera(rgb_hw=(720, 1280))
    cam.calib_ext()


if __name__ == '__main__':
    calib_cam_ext()
    # test_frame()

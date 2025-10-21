"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import sys
import cv2
import time
import trimesh
import numpy as np
from os import path
from threading import Thread
import nvdiffrast.torch as dr
sys.path.append(path.join(path.dirname(__file__), 'FoundationPose'))
from estimater import FoundationPose
from Utils import draw_posed_3d_box, draw_xyz_axis
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
sys.path.append(path.join(path.dirname(__file__), '..'))
from inference.realsense_cam import RealSenseCamera
from rl_sim.utils.common_robot_utils import encode_pose


class PoseEstimator:
    def __init__(self, object_name, cam=None):
        object_mesh = path.join(path.dirname(__file__), '../assets/ycb', object_name, 'textured_simple.obj')
        print('loading {}...'.format(object_mesh))
        mesh = trimesh.load(object_mesh)
        self.det_to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.det_bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        self.det_est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                      mesh=mesh, debug=False, scorer=ScorePredictor(),
                                      refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext())
        if cam is None:
            self.cam = RealSenseCamera()
        else:
            self.cam = cam
        self.running = False
        self.result = None

    def start(self):
        if not self.cam.running:
            self.cam.start()
        self.locate()
        self.running = True
        Thread(target=self.update, args=()).start()

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()
        self.cam.stop()

    def update(self):
        while True:
            if not self.running:
                return
            self.result = self.track()

    def read(self):
        if not self.running:
            print('Start before reading!')
            return
        while self.result is None:
            time.sleep(0.3)
        return encode_pose(self.result)

    def locate(self):
        bb = []
        def onMouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                bb.append((x, y))
            if event == cv2.EVENT_LBUTTONUP:
                bb.append((x, y))

        color, depth = self.cam.read()
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imshow('get_bb', bgr)
        cv2.setMouseCallback('get_bb', onMouse)
        while len(bb) < 2:
            cv2.waitKey(100)

        mask = np.zeros((color.shape[0], color.shape[1]), dtype=bool)
        mask[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]] = True
        masked = np.clip((np.tile(mask[:, :, None], [1, 1, 3]) * 100) + bgr, 0, 255).astype(np.uint8)
        cv2.imshow('get_bb', masked)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        _ = self.det_est.register(K=self.cam.cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=5)

    def track(self, vis=False):
        color, depth = self.cam.read()
        cam2obj = self.det_est.track_one(K=self.cam.cam_k, rgb=color.copy(), depth=depth.copy(), iteration=2)
        w2obj = np.matmul(self.cam.world2cam, cam2obj)

        # output visualize
        if vis:
            cam2obj = cam2obj @ np.linalg.inv(self.det_to_origin)
            im_anno = draw_posed_3d_box(self.cam.cam_k, img=color.copy(), bbox=self.det_bbox, ob_in_cam=cam2obj)
            im_anno = draw_xyz_axis(im_anno, scale=0.1, K=self.cam.cam_k, ob_in_cam=cam2obj,
                                    thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('result', im_anno)
            cv2.waitKey(1)

        return w2obj

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()


if __name__ == "__main__":
    a = PoseEstimator('tomato_soup_can')
    a.cam.start()
    a.locate()
    tick = time.time()
    while time.time() - tick < 30:
        print(a.track(vis=True))
    a.stop()

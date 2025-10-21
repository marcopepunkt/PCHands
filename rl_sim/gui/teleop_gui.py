"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
Based on code from https://github.com/yzqin/dex-hand-teleop with MIT licence
"""
import cv2
import numpy as np
from sapien.core import Pose
from sapien.utils import Viewer


DEFAULT_TABLE_TOP_CAMERAS = {
    "left": dict(position=np.array([0, 0.7, 0.4]), look_at_dir=np.array([0, -1, -0.6]), right_dir=np.array([-1, 0, 0]),
                 name="left_view", ),
    "bird": dict(position=np.array([0, 0, 0.7]), look_at_dir=np.array([0, 0, -1]), right_dir=np.array([0, -1, 0]),
                 name="bird_view", ),
}


class GUIBase:
    def __init__(self, scene, resolution=(640, 480), window_scale=0.5):
        self.scene = scene
        self.cams = []
        self.cam_mounts = []

        # Context
        self.nodes = []
        self.sphere_nodes = {}
        self.sphere_model = {}

        # Viewer
        self.viewer = Viewer()
        self.viewer.set_scene(scene)
        self.viewer.set_camera_xyz(-0.3, 0, 0.5)
        self.viewer.set_camera_rpy(0, -1.4, 0)
        self.resolution = resolution
        self.window_scale = window_scale
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light(np.array([0, -1, -1]), np.array([0.8, 0.8, 0.8]), shadow=True)

        # Key down action map
        self.keydown_map = {}

    def create_camera(self, position, look_at_dir, right_dir, name):
        builder = self.scene.create_actor_builder()
        mount = builder.build_static(name=f"{name}_mount")
        cam = self.scene.add_mounted_camera(name, mount, Pose(), width=self.resolution[0], height=self.resolution[1],
                                            fovy=0.9, near=0.1, far=10)

        # Construct camera pose
        look_at_dir = look_at_dir / np.linalg.norm(look_at_dir)
        right_dir = right_dir - np.sum(right_dir * look_at_dir).astype(np.float64) * look_at_dir
        right_dir = right_dir / np.linalg.norm(right_dir)
        up_dir = np.cross(look_at_dir, -right_dir)
        rot_mat_homo = np.stack([look_at_dir, -right_dir, up_dir, position], axis=1)
        pose_mat = np.concatenate([rot_mat_homo, np.array([[0, 0, 0, 1]])])

        # Add camera to the scene
        mount.set_pose(Pose(pose_mat))
        self.cams.append(cam)
        self.cam_mounts.append(mount)

    def _fetch_all_views(self, use_bgr=False):
        views = []
        for cam in self.cams:
            cam.take_picture()
            rgb = (np.clip(0, 1, cam.get_picture('Color')[..., :3]) * 255).astype(np.uint8)[..., ::-1]
            views.append(rgb)
        return views

    def render(self, render_all_views=True, additional_views=None, horizontal=True):
        self.scene.update_render()
        self.viewer.render()
        if not self.viewer.closed:
            for key, action in self.keydown_map.items():
                if self.viewer.window.key_down(key):
                    action()
        if (additional_views is not None or len(self.cams) > 0) and render_all_views:
            views = self._fetch_all_views(use_bgr=True)
            if additional_views is not None:
                views.extend(additional_views)

            if horizontal:
                pad = np.ones([views[0].shape[0], 200, 3], dtype=np.uint8) * 255
            else:
                pad = np.ones([200, views[0].shape[1], 3], dtype=np.uint8) * 255

            final_views = [views[0]]
            for i in range(1, len(views)):
                final_views.append(pad)
                final_views.append(views[i])
            axis = 1 if horizontal else 0
            final_views = np.concatenate(final_views, axis=axis)
            target_shape = final_views.shape
            target_shape = (int(target_shape[1] * self.window_scale), int(target_shape[0] * self.window_scale))
            final_views = cv2.resize(final_views, target_shape)
            cv2.imshow("Monitor", final_views)
            cv2.waitKey(1)

    def register_keydown_action(self, key, action):
        if key in self.keydown_map:
            raise RuntimeError(f"Key {key} has already been registered")
        self.keydown_map[key] = action

    @property
    def closed(self):
        return self.viewer.closed

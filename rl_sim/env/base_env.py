"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
Based on code from https://github.com/yzqin/dex-hand-teleop with MIT licence
"""
import os
import numpy as np
import gymnasium as gym
import sapien.core as sapien
from sapien.utils import Viewer
from gymnasium.utils import seeding
from functools import cached_property


class BaseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.viewer = None
        self.key_map = {}
        self.render_enabled = os.environ.get('SAPIEN_RENDER', '1') == '1'
        if self.render_enabled:
            self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()])
        else:
            self.scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
        self.scene.set_timestep(0.005)

    def __del__(self):
        self.scene = None
        if self.viewer is not None:
            self.viewer.close()

    def close(self):
        self.scene = None
        if self.viewer is not None:
            self.viewer.close()

    def seed(self, seed):
        self._np_random, self._np_random_seed = seeding.np_random(seed)

    def render(self, mode='human'):
        if not self.render_enabled:
            print('Sapien Rendering Disabled!')
            return

        if self.viewer is None:
            self.viewer = Viewer()
            self.viewer.set_scene(self.scene)
            self.add_default_scene_light(self.scene)
            self.viewer.set_camera_rpy(0, -0.7, 0.01)
            self.viewer.set_camera_xyz(-0.6, 0, 0.45)
        self.viewer.render()
        for key, action in self.key_map.items():
            if self.viewer.window.key_down(key):
                action()

    def step(self, action):
        # action = np.clip(action, -np.sqrt(2), np.sqrt(2))
        act_min, act_max = self.norm_stat['act_min'], self.norm_stat['act_max']
        act = (action.squeeze() + 1) / 2 * (act_max - act_min) + act_min
        self.rl_step(act)
        self.current_step += 1
        obs = self.get_observation()
        reward = self.get_reward()
        terminated, truncated = self.is_done()
        if terminated or truncated:
            infos = {'final_observation': obs}
        else:
            infos = {}
        return obs, reward, terminated, truncated, infos

    def create_table(self, table_half_size=(0.65, 0.65, 0.025)):
        builder = self.scene.create_actor_builder()
        top_pose = sapien.Pose([0, 0, -table_half_size[2]])
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        builder.add_box_collision(pose=top_pose, half_size=table_half_size, material=top_material)
        if self.render_enabled:
            builder.add_box_visual(pose=top_pose, half_size=table_half_size)
        return builder.build_static("table")

    @staticmethod
    def add_default_scene_light(scene, shadow=True):
        scene.add_ground(-1)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light(np.array([-1, -1, -1]), np.array([0.5, 0.5, 0.5]), shadow=shadow)
        scene.add_directional_light([0, 0, -1], [0.9, 0.8, 0.8])

    @staticmethod
    def count_contact(scene, actors1, actors2, impulse_threshold=5e-3):
        actor_set1 = set(actors1)
        actor_set2 = set(actors2)
        bodies = 0
        for contact in scene.get_contacts():
            contact_actors = {contact.bodies[0], contact.bodies[1]}
            if len(actor_set1 & contact_actors) > 0 and len(actor_set2 & contact_actors) > 0:
                impulse = [point.impulse for point in contact.points]
                if np.sum(np.abs(impulse)) >= impulse_threshold:
                    bodies += 1
        return bodies

    @cached_property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,))

    @cached_property
    def observation_space(self):
        high = np.inf * np.ones(self.obs_dim).astype(np.float32)
        return gym.spaces.Box(low=-high, high=high)

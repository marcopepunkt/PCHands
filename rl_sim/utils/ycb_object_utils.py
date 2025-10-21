"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
Based on code from https://github.com/yzqin/dex-hand-teleop with MIT licence
"""
import glob
import numpy as np
from os import path
import sapien.core as sapien


YCB_SIZE = {
    'mustard_bottle': ((0.0498, 0.0473), (0.0358, 0.0309), (0.0839, 0.1075)),
    'potted_meat_can': ((0.0534, 0.0487), (0.0309, 0.0294), (0.0498, 0.0339)),
    'tomato_soup_can': ((0.0342, 0.036), (0.0323, 0.0354), (0.0593, 0.0427)),
    'mug': ((0.0502, 0.0668), (0.0459, 0.0472), (0.0390, 0.0424)),
}
YCB_ROOT = path.join(path.dirname(__file__), '../../assets/ycb')


def load_ycb_object(scene, object_name, scale=1, visual_only=False):
    visual_file = path.join(YCB_ROOT, object_name, 'textured_simple.obj')
    collision_files = glob.glob(path.join(YCB_ROOT, object_name, 'collision*.obj'))
    builder = scene.create_actor_builder()
    scales = np.array([scale] * 3)
    density = 100
    try:
        _ = scene.get_render_system().device.name
    except:
        render_enabled = False
    else:
        render_enabled = True

    if visual_only:
        if render_enabled:
            builder.add_visual_from_file(visual_file, scale=scales,
                                         material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.3]))
        actor = builder.build_static(name=f"{object_name}_visual")
    else:
        material = scene.create_physical_material(1, 1, 0)
        for collision_file in collision_files:
            builder.add_convex_collision_from_file(collision_file, scale=scales, density=density, material=material)
        if render_enabled:
            builder.add_visual_from_file(visual_file, scale=scales)
        actor = builder.build(name=object_name)

    return actor
"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import sys
import cv2
import torch
import numpy as np
from os import path
import pyrealsense2 as pyrs
from threading import Thread
from torch.cuda.amp import autocast
sys.path.append(path.join(path.dirname(__file__), '..'))
from arhand.acr.config import args
from arhand.acr.model import ACR as ACR_v1
from arhand.acr.mano_wrapper import MANOWrapper
from arhand.acr.utils import (justify_detection_state, reorganize_results, img_preprocess,
                              create_OneEuroFilter, load_model, smooth_results, get_remove_keys, process_image_ori)


class RealSenseCamera:
    def __init__(self):
        self.img_h, self.img_w, self.fps = 480, 640, 30
        cfg = pyrs.config()
        cfg.enable_stream(pyrs.stream.color, self.img_w, self.img_h, pyrs.format.bgr8, self.fps)
        cfg.enable_stream(pyrs.stream.depth, self.img_w, self.img_h, pyrs.format.z16, self.fps)
        self.align = pyrs.align(pyrs.stream.color)
        self.stream = pyrs.pipeline()
        dev = self.stream.start(cfg)
        cam_int = dev.get_stream(pyrs.stream.color).as_video_stream_profile().get_intrinsics()
        self.depth_scale = dev.get_device().first_depth_sensor().get_depth_scale()
        self.depth_clip = (0, 3.0)
        self.cam_k = np.array([[cam_int.fx, 0, cam_int.ppx],
                               [0, cam_int.fy, cam_int.ppy],
                               [0, 0, 1]])
        self.frame = self.grab_realsense()
        self.stopped = False

    def grab_realsense(self):
        frames = self.align.process(self.stream.wait_for_frames())
        color = frames.get_color_frame().get_data()
        depth = frames.get_depth_frame().get_data()
        if not color or not depth:
            return
        color = np.asarray(color)
        depth = np.asarray(depth) * self.depth_scale
        depth = np.clip(depth, *self.depth_clip)
        return color, depth

    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self):
        while True:
            if self.stopped:
                return
            self.frame = self.grab_realsense()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()


class ManoMocap:
    def __init__(self):
        # Camera app
        self.camera = RealSenseCamera()
        self.camera.start()

        # Init cache
        self.init_shape_param_list = []
        self.step = self.init_step
        self.init_process = 0

        # Init result
        self.calibrated_shape_params = np.zeros([10])
        self.previous_offset = np.zeros(3, dtype=np.float32)

        # constants
        self.demo_cfg = {'mode': 'parsing', 'calc_loss': False}
        for i, j in vars(args()).items():
            setattr(self, i, j)
        self.focal_length = (self.camera.cam_k[0, 0] + self.camera.cam_k[1, 1]) / 2. * args().input_size / self.camera.img_w
        self.temporal_optimization = True
        self.filter_dict = {}
        self.filter_dict[0] = create_OneEuroFilter(args().smooth_coeff)
        self.filter_dict[1] = create_OneEuroFilter(args().smooth_coeff)

        # model
        model = ACR_v1().eval()
        model = load_model('checkpoints/wild.pkl', model, prefix='module.', drop_prefix='', fix_loaded=False)
        self.model = model.to(args().device)
        self.model.eval()
        self.mano_regression = MANOWrapper('../../assets/mano_hand').to(args().device)

    def normal_step(self, path='0'):
        img, depth = self.camera.read()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        outputs = self.single_image_forward(img, path)

        success, output= False, {'rgb': img}
        if outputs is not None and outputs['detection_flag']:
            depth = cv2.resize(process_image_ori(depth)[0], (args().input_size, args().input_size),
                               interpolation=cv2.INTER_CUBIC)
            _, results = self.process_results(outputs, depth=depth)

            for result in results[path]:
                if result['hand_type'] == 1:
                    self.previous_offset = result['cam_trans']
                    output = dict(rgb=img, vertices=result["verts"],
                                  offset=result['cam_trans'], pose_params=result['poses'])
                    success = True
                    break
        return success, output

    def init_step(self, path='0'):
        img, depth = self.camera.read()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        outputs = self.single_image_forward(img, path)

        output = {'rgb': img}
        if outputs is not None and outputs['detection_flag']:
            depth = cv2.resize(process_image_ori(depth)[0], (args().input_size, args().input_size), interpolation=cv2.INTER_CUBIC)
            _, results = self.process_results(outputs, depth=depth)

            for result in results[path]:
                if result['hand_type'] == 1:
                    offset = result['cam_trans']
                    has_init_offset = np.sum(np.abs(self.previous_offset)) > 1e-2

                    # Stop initialization process and clear data
                    if np.linalg.norm(offset - self.previous_offset) > 0.05 and has_init_offset:
                        self.init_process = 0
                        self.init_shape_param_list.clear()
                        self.previous_offset = offset
                    else:
                        # Continue init if offset not vary too much
                        self.previous_offset = offset
                        hand_pose = result['poses'][3:]
                        hand_pose = np.reshape(hand_pose, [15, 3])
                        if np.linalg.norm(hand_pose, axis=1).mean() < 0.50:
                            self.init_process += 0.03
                            self.init_shape_param_list.append(result["betas"])
                        else:
                            self.init_process = 0
                            self.init_shape_param_list.clear()
                    break

        # Compute initialization cache if process reach 100%
        if self.init_process >= 1:
            self.calibrated_shape_params = np.mean(self.init_shape_param_list, axis=0)
            # Switch step function
            self.step = self.normal_step

        return True, output

    @torch.inference_mode()
    def process_results(self, outputs, depth):
        # temporal optimization
        if self.temporal_optimization:
            out_hand = []  # [0],[1],[0,1]
            for idx, i in enumerate(outputs['detection_flag_cache']):
                if i:
                    out_hand.append(idx)  # idx is also hand type, 0 for left, 1 for right
                else:
                    out_hand.append(-1)

            assert len(outputs['params_dict']['poses']) == 2
            for sid, tid in enumerate(out_hand):
                if tid == -1:
                    continue
                outputs['params_dict']['poses'][sid], outputs['params_dict']['betas'][sid] = (
                    smooth_results(self.filter_dict[tid], outputs['params_dict']['poses'][sid],
                                   outputs['params_dict']['betas'][sid]))

        outputs = self.mano_regression(outputs, outputs['meta_data'], depth, self.focal_length)
        reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
        new_results = reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)
        return outputs, new_results

    @torch.inference_mode()
    def single_image_forward(self, img, path):
        meta_data = img_preprocess(img, path, input_size=args().input_size, single_img_input=True)

        ds_org, imgpath_org = get_remove_keys(meta_data, keys=['data_set', 'imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision == 'fp16':
            with autocast():
                outputs = self.model(meta_data, **self.demo_cfg)
        else:
            outputs = self.model(meta_data, **self.demo_cfg)

        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(outputs['detection_flag'],
                                                                                       outputs['reorganize_idx'])
        meta_data.update({'imgpath': imgpath_org, 'data_set': ds_org})
        outputs['meta_data']['imgpath'] = [path]

        return outputs

    @property
    def initialized(self):
        return self.init_process >= 1

    def compute_hand_zero_pos(self):
        if not self.initialized:
            raise RuntimeError(f"Can not perform hand shape based computation before initialization")
        shape = torch.from_numpy(self.calibrated_shape_params.astype(np.float32))[None].cuda()
        mano = self.mano_regression.mano_layer['r']
        with torch.no_grad():
            j3d = mano(torch.zeros((1, 48)).cuda(), shape).joints[0].detach().cpu().numpy()
        return j3d

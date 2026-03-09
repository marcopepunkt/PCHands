"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause

MANO Hand Model wrapper using ManoLayer

Anchor arrangement:
-----------------------------------------
finger (root-to-tip)  |  idx
-----------------------------------------
thumb                 |  0, 1, 2, 3
index                 |  4, 5, 6, 7
middle                |  8, 9, 10, 11
ring                  |  12, 13, 14, 15
pinky                 |  16, 17, 18, 19
palm                  |  20, 21
-----------------------------------------
"""
import os
import sys
import yaml
import tqdm
import time
import torch
import numpy as np
from os import path
from manotorch.manolayer import ManoLayer

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


class ManoHand:
    def __init__(self, dir_assets=None, use_pca=True, n_comp=12, flat_hand=False, 
                 hand_side='right', verbose=True, viz=None, viewer = None, headless=False):
        """
        Initialize MANO hand model
        :param dir_assets: Path to MANO assets directory
        :param use_pca: Use PCA space for pose (True) or joint space (False)
        :param n_comp: Number of PCA components
        :param flat_hand: Use flat hand mean pose
        :param hand_side: 'right' or 'left'
        :param verbose: Print model information
        """
        torch.set_num_threads(1)
        self.verbose = verbose
        
        # Configuration
        if dir_assets is None:
            dir_assets = path.join(path.dirname(__file__), '../assets', 'mano_hand')
        self.base_path = dir_assets
        self.use_pca = use_pca
        self.n_comp = n_comp
        self.hand_side = hand_side
        self.name = 'mano_hand'
        
        # Load anchor vertex mapping
        self.ach_vert = np.loadtxt(path.join(dir_assets, "anchor/anchor_vertex.txt"), dtype=int)
        self.ach_weight = np.loadtxt(path.join(dir_assets, "anchor/anchor_weight.txt"))
        
        self.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'    

        # Initialize MANO layer
        self.hand = ManoLayer(
            mano_assets_root=dir_assets, 
            side=hand_side, 
            center_idx=None,
            flat_hand_mean=flat_hand, 
            rot_mode='axisang',
            use_pca=use_pca, 
            ncomps=n_comp
        ).to(self.device) # Push it all to GPU if available
        
        # Model parameters
        self.dof_actuated = self.n_comp if use_pca else 45

        # State variables
        self.pose = np.zeros(self.dof_actuated)  # [pose_params]
        self.shape = np.zeros(10)
        self.rrot = np.array([0.0, 0.0, 0.0])
        self.rtsl = np.array([0.0, 0.0, 0.0])


        # Load calibration if available
        self._load_calibration()
        
        # Visualization
        if viewer is not None:
            self.viz = viewer
        else:
            if not headless:
                self.viz = meshcat.Visualizer()
                self.viz.open()
                # Set camera
                self.viz["/Cameras/default/rotated/<object>"].set_property("zoom", 4.0)
            else:
                self.viz = None
        if verbose:
            print('-------------------------------------------')
            print(f"Model: {self.name}, Side: {hand_side}")
            print(f"Use PCA: {use_pca}")
            print(f"n_comp: {n_comp if use_pca else 45}")
            print('-------------------------------------------')
    
    def _load_calibration(self):
        """Load calibration from file if available"""
        calib_file = path.join(path.dirname(__file__), 'calib_eef.yaml')
        try:
            with open(calib_file, 'r') as f:
                calib = yaml.safe_load(f)
                if calib and self.name in calib:
                    c = calib[self.name]
                    self.base_transform_calibrated = [
                        c.get('rx', 0.0), c.get('ry', 0.0), c.get('rz', 0.0),
                        c.get('tx', 0.0), c.get('ty', 0.0), c.get('tz', 0.0)
                    ]

                    self.rrot = np.array(self.base_transform_calibrated[0:3])
                    self.rtsl = np.array(self.base_transform_calibrated[3:6])

                    if self.verbose:
                        print(f"Loaded calibration: {self.base_transform_calibrated}")
        except FileNotFoundError:
            if self.verbose:
                print("No calibration file found, using default")
    
    def save_calib_eef(self, base_pos = None , base_rot = None):
        """
        Save end-effector calibration to file
        :param base_pos: [tx, ty, tz]
        :param base_rot: [rx, ry, rz]
        """
        file_path = path.join(path.dirname(__file__), "calib_eef.yaml")
        
        try:
            with open(file_path, "r") as f:
                calib = yaml.safe_load(f) or {}
        except FileNotFoundError:
            calib = {}

        print(f"Saving calibration to {file_path}")
                
        if base_pos is None:
            base_pos = self.rtsl.tolist()
        if base_rot is None:
            base_rot = self.rrot.tolist()

        base_pos = [float(v) for v in base_pos]
        base_rot = [float(v) for v in base_rot]
        
        calib[self.name] = dict(
            tx=base_pos[0], ty=base_pos[1], tz=base_pos[2],
            rx=base_rot[0], ry=base_rot[1], rz=base_rot[2],
        )
        
        with open(file_path, "w") as f:
            yaml.safe_dump(calib, f)
        
        self.base_transform_calibrated = base_rot + base_pos
    
    def reset(self):
        """Reset all joint configurations to initial state"""
        self.pose = np.zeros(self.dof_actuated)
        self.shape = np.zeros(10)
        self.rrot = np.array([0.0, 0.0, 0.0])
        self.rtsl = np.array([0.0, 0.0, 0.0])
    
    def forward_kinematic(self, q,rrot = None, rtsl = None, use_scheme=True, normalized=False):
        """
        Compute forward kinematics given joint values (This loads the pose parameters)
        :param q: Joint values (pose parameters)
        :param use_scheme: Not used for MANO (kept for API consistency)
        :param normalized: If True, input q is normalized (0-1)
        """
        q = np.asarray(q)
        
        # Handle different input sizes
        if len(q) == self.dof_actuated + 3:
            
            # Only pose parameters provided, keep current root rotation
            pose_params = q
            self.rrot = pose_params[:3]
            pose_params = pose_params[3:]
            self.rtsl = rtsl if rtsl is not None else self.rtsl
        elif len(q) == self.dof_actuated:
            # Full configuration provided
            self.rrot = rrot if rrot is not None else self.rrot
            self.rtsl = rtsl if rtsl is not None else self.rtsl
            pose_params = q
        else:
            raise ValueError(f"Whats given is fucked")
        
        # Denormalize if needed
        if normalized:
            self.pose = self.denormalize_joint(pose_params)
        else:
            self.pose = pose_params
    
    def inverse_kinematic_anchors(self, pos_anchor, niter=100, lr=8e-2, 
                         th_loss=0.0008,min_grad_norm=5e-2, min_param_change=1e-3, floating_base=True, hotstart=True,
                          focus_tip=False, temporal_smoothing=True, verbose=False):
        """
        Numerical inverse kinematics to find pose from anchor positions
        :param pos_anchor: Target anchor positions [22, 3]
        :param niter: Number of optimization iterations
        :param lr: Learning rate
        :param th_loss: Loss threshold for early stopping
        :param min_grad_norm: Minimum gradient norm for convergence
        :param min_param_change: Minimum parameter change for convergence
        :param floating_base: Optimize base transform
        :param hotstart: Use previous solution as initialization
        :param focus_tip: Focus on fingertip accuracy
        :param temporal_smoothing: Enable temporal smoothing of parameters
        :param verbose: Print optimization progress
        :return: (pose, shape, rrot, rtsl) - MANO parameters
        """

        # Here are all my hyperparmeters 
        pose_reg_weight = .01
        shape_reg_weight = 1

        pose_temp_smoothing = .1
        shape_temp_smoothing = .1
        base_temp_smoothing = 0.5

        calib_reg_weight = 0.01

        palm_parallel_weight = 0.01
        anchor_weight = 1.0
        
        fingertop_weight = 10.0
        palm_weight = 10.0


        
        # Prepare targets
        assert pos_anchor.shape == (22, 3), "pos_anchor must be of shape (22, 3)"
        anchor = torch.from_numpy(np.asarray(pos_anchor)).float().to(self.device)
        ach_vert = torch.from_numpy(self.ach_vert[:-1]).long().to(self.device)
        ach_weight = torch.from_numpy(self.ach_weight[:-1]).float().to(self.device)

        
        base_calib = torch.tensor(self.base_transform_calibrated, dtype=torch.float32, device=self.device)
        rrot_calib, rtsl_calib = base_calib[:3].unsqueeze(0), base_calib[3:6].unsqueeze(0)
        
        # Initialize pose and shape
        if hasattr(self, "_prev_pose") and hotstart:
            pose = self._prev_pose.clone().unsqueeze(0).to(self.device).requires_grad_(True)
            shape = self._prev_shape.clone().unsqueeze(0).to(self.device).requires_grad_(True)
        else:
            pose = torch.zeros((1, self.dof_actuated), device=self.device).requires_grad_(True)
            shape = torch.zeros((1, 10), device=self.device).requires_grad_(True)
        
        # Setup optimizer
        params = []
        if floating_base:
            # Initialize from previous if hotstart, else from calibration
            if hasattr(self, '_prev_rrot') and hotstart:
                rrot = self._prev_rrot.clone().unsqueeze(0).to(self.device)
                rtsl = self._prev_rtsl.clone().unsqueeze(0).to(self.device)
            else:
                rrot = rrot_calib.clone()
                rtsl = rtsl_calib.clone()

            rrot.requires_grad_(True)
            rtsl.requires_grad_(True)
            params.append({"params": [rrot, rtsl], "weight_decay": 0, "lr": 0.001*lr})
        else:
            rrot = rrot_calib.detach()
            rtsl = rtsl_calib.detach()


        params.extend([
            {"params": [pose], "weight_decay": 0, "lr": lr},
            {"params": [shape], "weight_decay": 0, "lr": lr},
        ])
        
        optimizer = torch.optim.AdamW(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=25, verbose=False
        )
        
        palm_indices = [20, 21]
        fingertip_indices = [3, 7, 11, 15, 19]

        # Loss weights
        weights = torch.ones_like(anchor) * anchor_weight
        if focus_tip:
            weights[fingertip_indices] = fingertop_weight
            weights[palm_indices] = palm_weight
        
        # Optimization loop
        check_every = 5
        prev_params = None
        
        if verbose:
            proc_bar = tqdm.tqdm(range(niter)) 
        else:
            proc_bar = range(niter)
            
        for iteration in proc_bar:
            optimizer.zero_grad()
        
            # Forward pass
            vertex = self.hand(torch.cat((rrot, pose), dim=1), shape).verts + rtsl
            a = vertex[0, ach_vert[:, 1]] - vertex[0, ach_vert[:, 0]]
            b = vertex[0, ach_vert[:, 2]] - vertex[0, ach_vert[:, 0]]
            anchor_pred = a * ach_weight[:, 0:1] + b * ach_weight[:, 1:2] + vertex[0, ach_vert[:, 0]]
            
            # Weighted Loss between anchors
            loss_huber = torch.nn.functional.huber_loss(anchor_pred, anchor, reduction='none')
            loss = (loss_huber * weights).mean() + shape_reg_weight * torch.sum(shape ** 2)
            
            # Palm parallelism constraint
            palm_axis_pred = anchor_pred[palm_indices[1]] - anchor_pred[palm_indices[0]]
            palm_axis_pred = palm_axis_pred / (torch.norm(palm_axis_pred) + 1e-8)
            palm_axis_target = anchor[palm_indices[1]] - anchor[palm_indices[0]]
            palm_axis_target = palm_axis_target / (torch.norm(palm_axis_target) + 1e-8)
            cosine_sim = torch.abs(torch.dot(palm_axis_pred, palm_axis_target))
            loss_palm_parallel = 1.0 - cosine_sim
            loss = loss + palm_parallel_weight * loss_palm_parallel
            
            # and add pose regularization that penalizes large values outside of -2 to 2
            pose_reg = pose_reg_weight * torch.mean(torch.clamp(torch.abs(pose) - 2.0, min=0.0) ** 2)
            loss = loss + pose_reg  



            # Temporal smoothing of pose and shape
            if hasattr(self, '_prev_pose') and temporal_smoothing:
                pose_reg = pose_temp_smoothing * torch.nn.functional.mse_loss(pose, self._prev_pose.unsqueeze(0).to(self.device))
                loss = loss + pose_reg
                shape_reg = shape_temp_smoothing * torch.nn.functional.mse_loss(shape, self._prev_shape.unsqueeze(0).to(self.device))
                loss = loss + shape_reg

                # Temporal smoothing of base
                if floating_base:
                    rtsl_reg = base_temp_smoothing * torch.norm(rtsl - self._prev_rtsl.unsqueeze(0).to(self.device))
                    rrot_reg = base_temp_smoothing * torch.norm(rrot - self._prev_rrot.unsqueeze(0).to(self.device))
                    loss = loss + rtsl_reg + rrot_reg
            
            # Calibration regularization
            if floating_base:
                rrot_reg = calib_reg_weight * torch.norm(rrot_calib - rrot)
                rtsl_reg = calib_reg_weight * torch.norm(rtsl_calib - rtsl)
                loss = loss + rrot_reg + rtsl_reg

           
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            
            # Progress tracking
            # proc_bar.set_description(f"loss: {loss.item():.5f}")
            
            # Visualization during optimization
            if self.viz is not None and iteration % 2 == 0:
                self.vis_model(rrot.squeeze(0), rtsl.squeeze(0), pose.squeeze(0), shape.squeeze(0), pos_anchor)
                time.sleep(0.05)  # Small delay to see optimization steps
            
            #Check convergence every N iterations
            if iteration % check_every == 0 and iteration > 0:
                # Compute gradient norm
                grad_norm = sum(p.grad.norm().item() ** 2 for p in [pose, shape, rrot, rtsl] if p.grad is not None) ** 0.5
                
                # Compute parameter change
                current_params = torch.cat([p.flatten() for p in [pose, shape, rrot, rtsl]])
                if prev_params is not None:
                    param_change = torch.norm(current_params - prev_params).item()
                else:
                    param_change = float('inf')
                prev_params = current_params.detach().clone()
                
                # Update progress bar with convergence info
                if verbose:
                    proc_bar.set_description(f"loss: {loss.item():.5f} | grad: {grad_norm:.2e} | Δp: {param_change:.2e}")
                
                # Check stopping criteria
                if grad_norm < min_grad_norm:
                    if verbose:
                        print(f"\nConverged: Vanishing gradients ({grad_norm:.2e})")
                    break
                
                if param_change < min_param_change:
                    if verbose:
                        print(f"\nConverged: Parameters stable ({param_change:.2e})")
                    break
                
                if loss.item() < th_loss:
                    if verbose:
                        print(f"\nConverged: Loss threshold ({loss.item():.6f})")
                    break
            # else:
            #     proc_bar.set_description(f"loss: {loss.item():.5f}")

        # Store results
        self._prev_pose = pose.detach().cpu().squeeze(0)
        self._prev_shape = shape.detach().cpu().squeeze(0)
        self._prev_rrot = rrot.detach().cpu().squeeze(0)
        self._prev_rtsl = rtsl.detach().cpu().squeeze(0)
        
        
        # Update internal state
        self.pose = pose.squeeze(0).detach().cpu().numpy()
        self.shape = shape.squeeze(0).detach().cpu().numpy()
        self.rrot = rrot.squeeze(0).detach().cpu().numpy()
        self.rtsl = rtsl.squeeze(0).detach().cpu().numpy()

        if verbose:
            print(f"The resulting pose is : {self.pose}")

        return self.pose, self.shape, self.rrot, self.rtsl

    def inverse_kinematic_manopoints(self, pos_manopoints, niter=1000, lr=8e-2,
                         th_loss=0.0008, min_grad_norm=5e-3, min_param_change=1e-3, floating_base=False, hotstart=True,
                          focus_tip=False, temporal_smoothing=False, wrist_centered=True, verbose=True):
        """
        Numerical inverse kinematics to find pose from MANO joint positions
        :param pos_manopoints: Target MANO joint positions [22, 3] in MANO-22 retargeter format:
                              [0: forearm, 1: wrist, 
                               2-5: thumb (base to tip), 6-9: index (base to tip), 
                               10-13: middle (base to tip), 14-17: ring (base to tip), 
                               18-21: pinky (base to tip)]
        :param niter: Number of optimization iterations
        :param lr: Learning rate
        :param th_loss: Loss threshold for early stopping
        :param min_grad_norm: Minimum gradient norm for convergence
        :param min_param_change: Minimum parameter change for convergence
        :param floating_base: Optimize base transform
        :param hotstart: Use previous solution as initialization
        :param focus_tip: Focus on fingertip accuracy
        :param temporal_smoothing: Enable temporal smoothing of parameters
        :param wrist_centered: If True, assumes target wrist is the origin and performs loss in wrist-relative frame
        :param verbose: Print optimization progress
        :return: (pose, shape, rrot, rtsl) - MANO parameters
        """
        # Hyperparameters
        pose_reg_weight = .001
        shape_reg_weight = 1

        pose_temp_smoothing = .1
        shape_temp_smoothing = .1
        base_temp_smoothing = 0.5

        calib_reg_weight = 0.

        joint_weight = 1.0
        fingertop_weight = 10.0
        wrist_weight = 10.0

        # Prepare targets - skip forearm (index 0), use wrist and fingers (indices 1-21)
        pos_manopoints = np.asarray(pos_manopoints)
        if pos_manopoints.shape == (66,):  # Flattened input, reshape it
            pos_manopoints = pos_manopoints.reshape(22, 3)
        assert pos_manopoints.shape == (22, 3), f"pos_manopoints must be of shape (22, 3), got {pos_manopoints.shape}"

        target_joints_np = np.asarray(pos_manopoints[1:], dtype=np.float32)  # [21, 3]
        target_wrist_np = target_joints_np[0].copy()
        if wrist_centered:
            target_joints_np = target_joints_np - target_wrist_np[np.newaxis, :]

        target_joints = torch.from_numpy(target_joints_np).float().to(self.device)
        
        # # Initialize base transform from actual hand orientation
        # # Extract key points: wrist(1), index base(6), middle base(10), pinky base(18)
        wrist_pos = torch.from_numpy(target_wrist_np).float().to(self.device)
        # index_base = torch.from_numpy(np.asarray(pos_manopoints[6])).float().to(self.device)
        # middle_base = torch.from_numpy(np.asarray(pos_manopoints[10])).float().to(self.device)
        # pinky_base = torch.from_numpy(np.asarray(pos_manopoints[18])).float().to(self.device)
        
        # # Compute palm coordinate frame
        # # Forward: from wrist toward middle finger base
        # forward = middle_base - wrist_pos
        # forward = forward / (torch.norm(forward) + 1e-8)
        
        # # Right: from pinky to index (for right hand)
        # right = index_base - pinky_base
        # right = right / (torch.norm(right) + 1e-8)
        
        # # Up: cross product to get palm normal
        # up = torch.cross(forward, right)
        # up = up / (torch.norm(up) + 1e-8)
        
        # # Re-orthogonalize right
        # right = torch.cross(up, forward)
        # right = right / (torch.norm(right) + 1e-8)
        
        # # Build rotation matrix [right, up, forward] as rows
        # rot_matrix = torch.stack([right, up, forward], dim=0)
        
        # # Convert rotation matrix to axis-angle representation
        # # Using Rodrigues formula in reverse
        # trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]
        # if trace > 0:
        #     s = torch.sqrt(trace + 1.0) * 2
        #     w = 0.25 * s
        #     x = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
        #     y = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
        #     z = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
        # else:
        #     if rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
        #         s = torch.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
        #         w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
        #         x = 0.25 * s
        #         y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
        #         z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        #     elif rot_matrix[1, 1] > rot_matrix[2, 2]:
        #         s = torch.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
        #         w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
        #         x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
        #         y = 0.25 * s
        #         z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        #     else:
        #         s = torch.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
        #         w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
        #         x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
        #         y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
        #         z = 0.25 * s
        
        # # Quaternion to axis-angle
        # angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))
        # if torch.abs(angle) < 1e-8:
        #     rrot_calib = torch.zeros(1, 3, device=self.device)
        # else:
        #     axis = torch.stack([x, y, z])
        #     axis = axis / (torch.norm(axis) + 1e-8)
        #     rrot_calib = (axis * angle).unsqueeze(0)
        
        if wrist_centered:
            rtsl_calib = torch.zeros((1, 3), device=self.device)
        else:
            rtsl_calib = wrist_pos.unsqueeze(0)  # Use actual wrist position for translation
        
        # Initialize pose and shape
        if hasattr(self, "_prev_pose") and hotstart:
            pose = self._prev_pose.clone().unsqueeze(0).to(self.device).requires_grad_(True)
        
            shape = self._prev_shape.clone().unsqueeze(0).to(self.device).requires_grad_(True)
            
        else:
            pose = torch.zeros((1, self.dof_actuated), device=self.device).requires_grad_(True)
            shape = torch.zeros((1, 10), device=self.device).requires_grad_(True)
        
        rrot_calib = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0)  # Start with no rotation
        # rtsl_calib = torch.tensor([0.0, 0.0, 0.0], device=self.device).unsqueeze(0)  # Start with no translation

        # Setup optimizer
        params = []
        if floating_base:
            # Initialize from previous if hotstart, else from calibration
            if hasattr(self, '_prev_rrot') and hotstart:
                rrot = self._prev_rrot.clone().unsqueeze(0).to(self.device)
                rtsl = self._prev_rtsl.clone().unsqueeze(0).to(self.device)
            else:
                rrot = rrot_calib.clone()  # Start with no rotation
                rtsl = rtsl_calib.clone()  # Start with no translation

            rrot.requires_grad_(True)
            rtsl.requires_grad_(True)
            params.append({"params": [rrot, rtsl], "weight_decay": 0, "lr": 0.001*lr})
        else:
            rrot = rrot_calib.clone()  # Start with no rotation
            rtsl = rtsl_calib.clone()  # Start with no translation

        params.extend([
            {"params": [pose], "weight_decay": 0, "lr": lr},
            {"params": [shape], "weight_decay": 0, "lr": lr},
        ])
        
        optimizer = torch.optim.AdamW(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=25, verbose=False
        )
        
        # Joint indices in 21-joint format (after removing forearm from MANO-22):
        # wrist=0, thumb=1-4, index=5-8, middle=9-12, ring=13-16, pinky=17-20
        wrist_idx = 0
        fingertip_indices = [4, 8, 12, 16, 20]  # Tips: thumb, index, middle, ring, pinky

        # Loss weights
        weights = torch.ones_like(target_joints) * joint_weight
        if focus_tip:
            weights[fingertip_indices] = fingertop_weight
            weights[wrist_idx] = wrist_weight
        
        # Optimization loop
        check_every = 5
        prev_params = None
        
        if verbose:
            proc_bar = tqdm.tqdm(range(niter)) 
        else:
            proc_bar = range(niter)
            
        for iteration in proc_bar:
            optimizer.zero_grad()
        
            # Forward pass - get MANO joints
            output = self.hand(torch.cat((rrot, pose), dim=1), shape)
            joints_pred = output.joints[0] + rtsl  # [21, 3]

            if wrist_centered:
                joints_pred_loss = joints_pred - joints_pred[wrist_idx:wrist_idx + 1]
            else:
                joints_pred_loss = joints_pred
            
            # Weighted Loss between joints
            loss_huber = torch.nn.functional.huber_loss(joints_pred_loss, target_joints, reduction='none')
            loss = (loss_huber * weights).mean() + shape_reg_weight * torch.sum(shape ** 2)
            
            # Pose regularization that penalizes large values outside of -2 to 2
            pose_reg = pose_reg_weight * torch.mean(torch.clamp(torch.abs(pose) - 2.0, min=0.0) ** 2)
            loss = loss + pose_reg

            # Temporal smoothing of pose and shape
            if hasattr(self, '_prev_pose') and temporal_smoothing:
                pose_reg = pose_temp_smoothing * torch.nn.functional.mse_loss(pose, self._prev_pose.unsqueeze(0).to(self.device))
                loss = loss + pose_reg
                shape_reg = shape_temp_smoothing * torch.nn.functional.mse_loss(shape, self._prev_shape.unsqueeze(0).to(self.device))
                loss = loss + shape_reg

                # Temporal smoothing of base
                if floating_base:
                    rtsl_reg = base_temp_smoothing * torch.norm(rtsl - self._prev_rtsl.unsqueeze(0).to(self.device))
                    rrot_reg = base_temp_smoothing * torch.norm(rrot - self._prev_rrot.unsqueeze(0).to(self.device))
                    loss = loss + rtsl_reg + rrot_reg
            
            # Calibration regularization
            if floating_base:
                rrot_reg = calib_reg_weight * torch.norm(rrot_calib - rrot)
                rtsl_reg = calib_reg_weight * torch.norm(rtsl_calib - rtsl)
                loss = loss + rrot_reg + rtsl_reg

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            
            # Visualization during optimization
            if self.viz is not None and iteration % 2 == 0:
                self.vis_model(rrot.squeeze(0), rtsl.squeeze(0), pose.squeeze(0), shape.squeeze(0), 
                             target_manopoints=pos_manopoints)
                time.sleep(0.05)  # Small delay to see optimization steps
            
            # Check convergence every N iterations
            if iteration % check_every == 0 and iteration > 0:
                # Compute gradient norm
                grad_norm = sum(p.grad.norm().item() ** 2 for p in [pose, shape, rrot, rtsl] if p.grad is not None) ** 0.5
                
                # Compute parameter change
                current_params = torch.cat([p.flatten() for p in [pose, shape, rrot, rtsl]])
                if prev_params is not None:
                    param_change = torch.norm(current_params - prev_params).item()
                else:
                    param_change = float('inf')
                prev_params = current_params.detach().clone()
                
                # Update progress bar with convergence info
                if verbose:
                    proc_bar.set_description(f"loss: {loss.item():.5f} | grad: {grad_norm:.2e} | Δp: {param_change:.2e}")
                
                # Check stopping criteria
                if grad_norm < min_grad_norm:
                    if verbose:
                        print(f"\nConverged: Vanishing gradients ({grad_norm:.2e})")
                    break
                
                if param_change < min_param_change:
                    if verbose:
                        print(f"\nConverged: Parameters stable ({param_change:.2e})")
                    break
                
                if loss.item() < th_loss:
                    if verbose:
                        print(f"\nConverged: Loss threshold ({loss.item():.6f})")
                    break

        # Store results
        self._prev_pose = pose.detach().cpu().squeeze(0)
        self._prev_shape = shape.detach().cpu().squeeze(0)
        self._prev_rrot = rrot.detach().cpu().squeeze(0)
        self._prev_rtsl = rtsl.detach().cpu().squeeze(0)
        
        # Update internal state
        self.pose = pose.squeeze(0).detach().cpu().numpy()
        self.shape = shape.squeeze(0).detach().cpu().numpy()
        self.rrot = rrot.squeeze(0).detach().cpu().numpy()
        self.rtsl = rtsl.squeeze(0).detach().cpu().numpy()

        if verbose:
            print(f"The resulting pose is : {self.pose}")

        return self.pose, self.shape, self.rrot, self.rtsl

    def vis_model(self, rrot = None, rtsl = None, pose = None, shape = None, target_anchors = None, target_manopoints = None, return_image=False):
        """Helper to visualize IK optimization step using MeshCat"""

        if rrot is None:
            rrot = self.rrot
            # print("Using current rrot")
        if rtsl is None:
            # print("Using current rtsl")
            rtsl = self.rtsl
        if pose is None:
            # print("Using current pose")
            pose = self.pose
        if shape is None:
            # print("Using current shape")
            shape = self.shape

        anchors, vertices = self.get_anchors_vertices(pose, shape, rrot, rtsl)
        
        # Update hand mesh
        faces = self.hand.th_faces.detach().cpu().numpy().astype(np.uint32)
        mesh = g.TriangularMeshGeometry(vertices, faces)
        self.viz["hand"].set_object(mesh, g.MeshLambertMaterial(color=0x505050))
        
        # Update current anchors
        for i in range(22):
            self.viz[f"anchors/A_{i:02d}"].set_object(
                g.Sphere(0.004),
                g.MeshLambertMaterial(color=self._rgb_to_int(self.colors[i]))
            )
            self.viz[f"anchors/A_{i:02d}"].set_transform(
                tf.translation_matrix(anchors[i])
            )
        
        if target_anchors is not None:
        # Update target anchors
            for i, anchor in enumerate(target_anchors):
                self.viz[f"targets/T_{i:02d}"].set_object(
                    g.Box([0.002, 0.002, 0.002]),
                    g.MeshLambertMaterial(color=self._rgb_to_int(self.colors[i]), opacity=0.5, transparent=True)
                )
                self.viz[f"targets/T_{i:02d}"].set_transform(
                    tf.translation_matrix(anchor)
                )

        if target_manopoints is not None:
            target_manopoints = np.asarray(target_manopoints)
            if target_manopoints.shape == (66,):
                target_manopoints = target_manopoints.reshape(22, 3)
            assert target_manopoints.shape == (22, 3), f"target_manopoints must be (22, 3), got {target_manopoints.shape}"

            for i, point in enumerate(target_manopoints):
                self.viz[f"targets_mano/M_{i:02d}"].set_object(
                    g.Sphere(0.003),
                    g.MeshLambertMaterial(color=0x00FF00, opacity=0.8, transparent=True)
                )
                self.viz[f"targets_mano/M_{i:02d}"].set_transform(
                    tf.translation_matrix(point)
                )
        

        if return_image:
            # import cv2
            return self.viz.get_image()

    
    def get_anchors_vertices(self, pose, shape, rrot, rtsl = None):
        """
        Compute anchor points and vertices from pose and shape
        :param pose: MANO pose parameters [3 + n_comp]
        :param shape: MANO shape parameters [10]
        :param base_transform: Optional base transform [rx, ry, rz, tx, ty, tz]
        :return: (anchors [22, 3], vertices [778, 3])
        """        
        # Compute vertices
        pose_tensor = torch.as_tensor(pose, dtype=torch.float32, device=self.device).unsqueeze(0)
        rrot_tensor = torch.as_tensor(rrot, dtype=torch.float32, device=self.device).unsqueeze(0)
        rrot_pose_tensor = torch.cat((rrot_tensor, pose_tensor), dim=1)
        shape_tensor = torch.as_tensor(shape, dtype=torch.float32, device=self.device).unsqueeze(0)

        vertices = self.hand(rrot_pose_tensor, shape_tensor).verts[0].detach().cpu().numpy()
        
        # Compute anchors from vertices
        a = vertices[self.ach_vert[:, 1]] - vertices[self.ach_vert[:, 0]]
        b = vertices[self.ach_vert[:, 2]] - vertices[self.ach_vert[:, 0]]
        anchors = (a * self.ach_weight[:, 0:1] + 
                  b * self.ach_weight[:, 1:2] + 
                  vertices[self.ach_vert[:, 0]])
        
        # Apply base transform
        if rtsl is None:
            return anchors, vertices
        else:
            rtsl = rtsl.detach().cpu().numpy() if torch.is_tensor(rtsl) else rtsl
            anchors = anchors + rtsl
            vertices = vertices + rtsl
            return anchors, vertices
    
    def get_anchor(self):
        """
        Get anchor positions in world frame
        :param pose: Optional pose override
        :param shape: Optional shape override
        :return: np.ndarray [22, 3]
        """
        anchors, _ = self.get_anchors_vertices(self.pose, self.shape, self.rrot, self.rtsl)
        return anchors
    
    
    
    def denormalize_joint(self, qn):
        """
        Convert normalized values (0-1) to actual joint values
        :param qn: Normalized joint values
        :return: Denormalized joint values
        """
        # For MANO, typical range is approximately [-2, 2] for pose parameters
        qn = np.asarray(qn)
        lower = -2.0
        upper = 2.0
        return lower + (upper - lower) * qn
    
    def get_mano_keypoints(self):
        """
        Get MANO keypoints (anchor points)
        :param pose: Optional pose override
        :param shape: Optional shape override
        :return: np.ndarray [22, 3]
        """

         # Convert to tensors
        rrot = torch.from_numpy(self.rrot).float().unsqueeze(0)    # (1, 3) - root rotation
        pose = torch.from_numpy(self.pose).float().unsqueeze(0)    # (1, ncomps) - PCA coefficients
        shape = torch.from_numpy(self.shape).float().unsqueeze(0).to(self.device)  # (1, 10) - shape parameters
        
        # Concatenate root rotation and PCA coefficients
        state = torch.cat((rrot, pose), dim=1).to(self.device)  # (1, 3 + ncomps)
        
        # Forward pass through MANO
        output = self.hand(state, shape)
        joints = output.joints.detach().squeeze(0).cpu().numpy()  # (21, 3)

        # Shift the Joints along rtsl 
        if self.rtsl is not None:
            joints += self.rtsl[np.newaxis, :]

        # Extract MANO joints (0-indexed)
        wrist = joints[0]         # Wrist
        thumb = joints[1:5]       # Thumb (4 joints)
        index = joints[5:9]       # Index (4 joints)
        middle = joints[9:13]     # Middle (4 joints)
        ring = joints[13:17]      # Ring (4 joints)
        pinky = joints[17:21]     # Pinky (4 joints)

        # Create synthetic forearm point
        # Option 1: Offset along current wrist direction
        forearm = wrist.copy()
        
        # Get approximate hand direction from palm normal
        # Use middle finger base and wrist to determine palm direction
        palm_direction = wrist - middle[0]  # Vector from middle base to wrist
        palm_direction = palm_direction / (np.linalg.norm(palm_direction) + 1e-8)
        
        # Forearm is ~10cm along palm direction from wrist 
        forearm = wrist + 0.10 * palm_direction # Ugly and hardcoded but works decently
        forearm[0] = 0.0  # Set X to zero for better alignment, ugly and hardcoded but works decently # TODO: improve this
        
        # Assemble in retargeter order: [forearm, wrist, thumb, index, middle, ring, pinky]
        joints_retargeter_format = np.vstack([
            forearm[np.newaxis, :],  # 0: forearm
            wrist[np.newaxis, :],    # 1: wrist
            thumb,                    # 2-5: thumb
            index,                    # 6-9: index
            middle,                   # 10-13: middle
            ring,                     # 14-17: ring
            pinky                     # 18-21: pinky
        ])
        
        return joints_retargeter_format  # Shape: (22, 3)

        
        
    def _rgb_to_int(self, rgb_norm):
        """Helper to convert 0-1 float RGB to hex integer"""
        r, g, b = [int(c * 255) for c in rgb_norm]
        return (r << 16) | (g << 8) | b
    
    def cleanup(self):
        """Cleanup resources"""
        if self.viz is not None:
            try:
                self.viz.close()
            except:
                pass
            self.viz = None
    
    colors = np.array([[5.03830e-02, 2.98030e-02, 5.27975e-01],
                       [1.64070e-01, 2.01710e-02, 5.77478e-01],
                       [2.48032e-01, 1.44390e-02, 6.12868e-01],
                       [3.25150e-01, 6.91500e-03, 6.39512e-01],
                       [3.99411e-01, 8.59000e-04, 6.56133e-01],
                       [4.71457e-01, 5.67800e-03, 6.59897e-01],
                       [5.46157e-01, 3.89540e-02, 6.47010e-01],
                       [6.10667e-01, 9.02040e-02, 6.19951e-01],
                       [6.69845e-01, 1.42992e-01, 5.82154e-01],
                       [7.23444e-01, 1.96158e-01, 5.38981e-01],
                       [7.71958e-01, 2.49237e-01, 4.94813e-01],
                       [8.19651e-01, 3.06812e-01, 4.48306e-01],
                       [8.59750e-01, 3.60588e-01, 4.06917e-01],
                       [8.96131e-01, 4.15712e-01, 3.66407e-01],
                       [9.28329e-01, 4.72975e-01, 3.26067e-01],
                       [9.55470e-01, 5.33093e-01, 2.85490e-01],
                       [9.77856e-01, 6.02051e-01, 2.41387e-01],
                       [9.90681e-01, 6.69558e-01, 2.01642e-01],
                       [9.94495e-01, 7.40880e-01, 1.66335e-01],
                       [9.87621e-01, 8.15978e-01, 1.44363e-01],
                       [9.68443e-01, 8.94564e-01, 1.47014e-01],
                       [9.40015e-01, 9.75158e-01, 1.31326e-01]])


def hold():
    """Keep the program running to view MeshCat"""
    import time
    try:
        print("Press Ctrl+C to exit...")
        print("View the visualization at: http://127.0.0.1:7000/static/")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    import time
    
    # Example usage
    print("Initializing MANO Hand...")
    mano = ManoHand(use_pca=True, n_comp=12, verbose=True)
    mano.reset()
    
    # Random pose test
    # print("\nTesting forward kinematics with random pose...")
    # random_pose = np.random.randn(12) * 0.5
    # mano.forward_kinematic(random_pose)
    
    # # Visualize initial pose
    # print("Visualizing initial pose...")
    # mano.vis_model()
    
    # # Test inverse kinematics
    # print("\nTesting inverse kinematics...")
    # target_mano_keypoints = mano.get_mano_keypoints()
    # print("Visualizing MANO keypoint targets...")
    # mano.vis_model(target_manopoints=target_mano_keypoints)
    # # Perturb targets slightly
    # # target_mano_keypoints += np.random.randn(*target_mano_keypoints.shape) * 0.001
    
    
    # print("Running IK optimization in 10 seconds...")
    # time.sleep(5)  # Wait before starting optimization to view initial state
    # mano.reset()  # Reset to default pose before IK


    mano.vis_model()
    # time.sleep(2)  # Wait to view target keypoints before optimization

    target_mano_keypoints = np.array([
        [ 0.,          0.,         -0.09      ],
        [ 0.,          0.,          0.        ],
        [-0.01949062, -0.00941284,  0.01627053],
        [-0.05942997, -0.03465216,  0.0296706 ],
        [-0.08598454, -0.04365316,  0.02295506],
        [-0.10984702, -0.05040416,  0.0131325 ],
        [-0.08437505, -0.00145769,  0.02750516],
        [-0.12110482, -0.01388912,  0.02527022],
        [-0.13432692, -0.03187465,  0.02315342],
        [-0.14218608, -0.05063087,  0.02111305],
        [-0.08303285,  0.00311293,  0.0043393 ],
        [-0.12272803, -0.01428164, -0.00041428],
        [-0.12504712, -0.04115006, -0.00117505],
        [-0.11099223, -0.05695527,  0.00070275],
        [-0.08018334,  0.00145372, -0.01510685],
        [-0.11467998, -0.01404116, -0.01856331],
        [-0.11460691, -0.03942096, -0.01805448],
        [-0.09862249, -0.0518352,  -0.01635602],
        [-0.07046202, -0.00276888, -0.03168281],
        [-0.09798802, -0.01585541, -0.03459984],
        [-0.10098015, -0.03340237, -0.03324441],
        [-0.08861113, -0.04668252, -0.02994843],
    ])

    mano.vis_model(target_manopoints=target_mano_keypoints)
    time.sleep(5)  # Wait to view target keypoints before optimization

    mano.inverse_kinematic_manopoints(target_mano_keypoints, niter=1000, floating_base=True)
    

    # print("Visualizing IK result...")
    # mano.vis_model(target_mano_keypoints=target_mano_keypoints)
    
    # # Animate random poses
    # print("\nAnimating random poses (5 iterations)...")
    # for i in range(5):
    #     print(f"Iteration {i+1}/5")
    #     random_pose = np.random.randn(mano.dof_tendons)
    #     mano.forward_kinematic(random_pose)
    #     mano.vis_model()
    #     time.sleep(3)
    
    print("\nDone! Keep the window open to view the final pose.")
    hold()

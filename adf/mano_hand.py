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
import torch
import numpy as np
from os import path
from manotorch.manolayer import ManoLayer

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


class ManoHand:
    def __init__(self, dir_assets=None, use_pca=True, n_comp=12, flat_hand=False, 
                 hand_side='right', verbose=True, viz=None, viewer = None):
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
        
        # Initialize MANO layer
        self.hand = ManoLayer(
            mano_assets_root=dir_assets, 
            side=hand_side, 
            center_idx=None,
            flat_hand_mean=flat_hand, 
            rot_mode='axisang',
            use_pca=use_pca, 
            ncomps=n_comp
        )
        
        # Model parameters
        self.dof_tendons = self.n_comp if use_pca else 45
        self.model_nq = 3 + self.dof_tendons  # 3 for root rotation + pose params
        self.model_nv = self.dof_tendons
        
        # State variables
        self.pose = np.zeros(self.model_nq)  # [root_rot(3) + pose_params]
        self.shape = np.zeros(10)
        self.base_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [rx, ry, rz, tx, ty, tz]
        
        # Load calibration if available
        self._load_calibration()
        
        # Visualization
        if viewer is not None:
            self.viz = viewer
        else:
            self.viz = meshcat.Visualizer()
            self.viz.open()
            # Set camera
            self.viz["/Cameras/default/rotated/<object>"].set_property("zoom", 4.0)
        
        if verbose:
            print('-------------------------------------------')
            print(f"Model: {self.name}, Side: {hand_side}")
            print(f"Use PCA: {use_pca}")
            print(f"n_comp: {n_comp if use_pca else 45}")
            print(f"DoF (tendons): {self.dof_tendons}")
            print(f"Config size (nq): {self.model_nq}")
            print(f"Velocity size (nv): {self.model_nv}")
            print('-------------------------------------------')
    
    def _load_calibration(self):
        """Load calibration from file if available"""
        calib_file = path.join(path.dirname(__file__), 'calib_eef.yaml')
        try:
            with open(calib_file, 'r') as f:
                calib = yaml.safe_load(f)
                if calib and self.name in calib:
                    c = calib[self.name]
                    self.base_transform = [
                        c.get('rx', 0.0), c.get('ry', 0.0), c.get('rz', 0.0),
                        c.get('tx', 0.0), c.get('ty', 0.0), c.get('tz', 0.0)
                    ]
                    if self.verbose:
                        print(f"Loaded calibration: {self.base_transform}")
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
            assert self.base_transform is not None, "Base transform is not set"
            assert len(self.base_transform) == 6, "Base transform must have 6 elements"
            base_pos = self.base_transform[3:6]
        if base_rot is None:
            base_rot = self.base_transform[0:3]

        base_pos = [float(v) for v in base_pos]
        base_rot = [float(v) for v in base_rot]
        
        calib[self.name] = dict(
            tx=base_pos[0], ty=base_pos[1], tz=base_pos[2],
            rx=base_rot[0], ry=base_rot[1], rz=base_rot[2],
        )
        
        with open(file_path, "w") as f:
            yaml.safe_dump(calib, f)
        
        self.base_transform = base_rot + base_pos
    
    def reset(self):
        """Reset all joint configurations to initial state"""
        self.pose = np.zeros(self.model_nq)
        self.shape = np.zeros(10)
    
    def forward_kinematic(self, q, use_scheme=True, normalized=False):
        """
        Compute forward kinematics given joint values
        :param q: Joint values (pose parameters)
        :param use_scheme: Not used for MANO (kept for API consistency)
        :param normalized: If True, input q is normalized (0-1)
        """
        q = np.asarray(q)
        
        # Handle different input sizes
        if len(q) == self.dof_tendons:
            # Only pose parameters provided, keep current root rotation
            pose_params = q
            root_rot = self.pose[:3]
        elif len(q) == self.model_nq:
            # Full configuration provided
            root_rot = q[:3]
            pose_params = q[3:]
        else:
            raise ValueError(f"Expected {self.dof_tendons} or {self.model_nq} values, got {len(q)}")
        
        # Denormalize if needed
        if normalized:
            pose_params = self.denormalize_joint(pose_params)
        
        # Update state
        self.pose = np.concatenate([root_rot, pose_params])
    
    def inverse_kinematic(self, pos_anchor, niter=100, lr=1e-2, wd=1e-4, 
                         th_loss=0.0008,min_grad_norm=5e-2, min_param_change=1e-3, floating_base=True, hotstart=True, 
                         visualize=False, focus_tip=False, temporal_smoothing=True):
        """
        Numerical inverse kinematics to find pose from anchor positions
        :param pos_anchor: Target anchor positions [22, 3]
        :param niter: Number of optimization iterations
        :param lr: Learning rate
        :param wd: Weight decay
        :param th_loss: Loss threshold for early stopping
        :param floating_base: Optimize base transform
        :param hotstart: Use previous solution as initialization
        :param visualize: Visualize optimization process
        :param focus_tip: Focus on fingertip accuracy
        """

        # Here are all my hyperparmeters 
        pose_reg_weight = 2.0
        shape_reg_weight = 5.0
        base_reg_weight = 0.5
        calib_reg_weight = 0.1

        palm_parallel_weight = 0.01
        anchor_weight = 1.0
        
        fingertop_weight = 10.0
        palm_weight = 10.0




        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hand.to(device)
        
        
        # Prepare targets
        assert pos_anchor.shape == (22, 3), "pos_anchor must be of shape (22, 3)"
        anchor = torch.from_numpy(np.asarray(pos_anchor)).float().to(device)
        ach_vert = torch.from_numpy(self.ach_vert[:-1]).long().to(device)
        ach_weight = torch.from_numpy(self.ach_weight[:-1]).float().to(device)

        
        base = torch.tensor(self.base_transform, dtype=torch.float32, device=device)
        rrot_calib, rtsl_calib = base[:3].unsqueeze(0), base[3:6].unsqueeze(0)
        
        # Initialize pose and shape
        if hasattr(self, "_prev_pose") and hotstart:
            pose = self._prev_pose.clone().unsqueeze(0).to(device).requires_grad_(True)
            shape = self._prev_shape.clone().unsqueeze(0).to(device).requires_grad_(True)
        else:
            pose = torch.zeros((1, self.dof_tendons), device=device).requires_grad_(True)
            shape = torch.zeros((1, 10), device=device).requires_grad_(True)
        
        # Setup optimizer
        params = []
        if floating_base:
            # Initialize from previous if hotstart, else from calibration
            if hasattr(self, '_prev_rrot') and hotstart:
                rrot = self._prev_rrot.clone().unsqueeze(0).to(device)
                rtsl = self._prev_rtsl.clone().unsqueeze(0).to(device)
            else:
                rrot = rrot_calib.clone()
                rtsl = rtsl_calib.clone()

            rrot.requires_grad_(True)
            rtsl.requires_grad_(True)
            params.append({"params": [rrot, rtsl], "weight_decay": 0, "lr": 0.1 * lr})
        else:
            rrot = rrot_calib.detach()
            rtsl = rtsl_calib.detach()


        params.extend([
            {"params": [pose], "weight_decay": wd, "lr": lr},
            {"params": [shape], "weight_decay": 0, "lr": lr * 0.01},
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
        
        proc_bar = tqdm.tqdm(range(niter)) 
        for iteration in proc_bar:
            optimizer.zero_grad()
            
            # Forward pass
            vertex = self.hand(torch.cat((rrot, pose), dim=1), shape).verts + rtsl
            a = vertex[0, ach_vert[:, 1]] - vertex[0, ach_vert[:, 0]]
            b = vertex[0, ach_vert[:, 2]] - vertex[0, ach_vert[:, 0]]
            anchor_pred = a * ach_weight[:, 0:1] + b * ach_weight[:, 1:2] + vertex[0, ach_vert[:, 0]]
            
            # Weighted Loss between anchors
            loss_huber = torch.nn.functional.huber_loss(anchor_pred, anchor, reduction='none')
            loss = (loss_huber * weights).mean() + 0.0 * torch.sum(shape ** 2)
            
            # Palm parallelism constraint
            palm_axis_pred = anchor_pred[palm_indices[1]] - anchor_pred[palm_indices[0]]
            palm_axis_pred = palm_axis_pred / (torch.norm(palm_axis_pred) + 1e-8)
            palm_axis_target = anchor[palm_indices[1]] - anchor[palm_indices[0]]
            palm_axis_target = palm_axis_target / (torch.norm(palm_axis_target) + 1e-8)
            cosine_sim = torch.abs(torch.dot(palm_axis_pred, palm_axis_target))
            loss_palm_parallel = 1.0 - cosine_sim
            loss = loss + palm_parallel_weight * loss_palm_parallel
            

            # Temporal smoothing of pose and shape
            if hasattr(self, '_prev_pose') and temporal_smoothing:
                pose_reg = pose_reg_weight * torch.nn.functional.mse_loss(pose, self._prev_pose.unsqueeze(0).to(device))
                loss = loss + pose_reg
                shape_reg = shape_reg_weight * torch.nn.functional.mse_loss(shape, self._prev_shape.unsqueeze(0).to(device))
                loss = loss + shape_reg

                # Temporal smoothing of base
                if floating_base:
                    rtsl_reg = base_reg_weight * torch.norm(rtsl - self._prev_rtsl.unsqueeze(0).to(device))
                    rrot_reg = base_reg_weight * torch.norm(rrot - self._prev_rrot.unsqueeze(0).to(device))
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
            if visualize and iteration % 2 == 0:
                self._visualize_ik_step(rrot, rtsl, pose, shape, pos_anchor)
            
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
                proc_bar.set_description(f"loss: {loss.item():.5f} | grad: {grad_norm:.2e} | Δp: {param_change:.2e}")
                
                # Check stopping criteria
                if grad_norm < min_grad_norm:
                    print(f"\nConverged: Vanishing gradients ({grad_norm:.2e})")
                    break
                
                if param_change < min_param_change:
                    print(f"\nConverged: Parameters stable ({param_change:.2e})")
                    break
                
                if loss.item() < th_loss:
                    print(f"\nConverged: Loss threshold ({loss.item():.6f})")
                    break
            # else:
            #     proc_bar.set_description(f"loss: {loss.item():.5f}")

        
        # Store results
        self._prev_pose = pose.detach().cpu().squeeze(0)
        self._prev_shape = shape.detach().cpu().squeeze(0)
        self._prev_rrot = rrot.detach().cpu().squeeze(0)
        self._prev_rtsl = rtsl.detach().cpu().squeeze(0)
        
        self.hand.to('cpu')
        
        # Update internal state
        self.pose = pose.squeeze(0).detach().cpu().numpy()
        self.shape = shape.squeeze(0).detach().cpu().numpy()
        self.base_transform = (rrot.detach().cpu().flatten().tolist() + 
                               rtsl.detach().cpu().flatten().tolist())
        self.rrot = rrot.squeeze(0).detach().cpu().numpy()
        self.rtsl = rtsl.squeeze(0).detach().cpu().numpy()

    def _visualize_ik_step(self, rrot, rtsl, pose, shape, target_anchors):
        """Helper to visualize IK optimization step using MeshCat"""
    
        
        # Move to CPU for visualization
        self.hand.to('cpu')
        
        # Compute current state
        full_pose = torch.cat((rrot.cpu(), pose.cpu()), dim=1).detach().numpy().flatten()
        shape_np = shape.cpu().detach().numpy().flatten()
        temp_base = rrot.detach().cpu().flatten().tolist() + rtsl.detach().cpu().flatten().tolist()
        
        anchors, vertices = self._compute_anchors_vertices(full_pose, shape_np, temp_base)
        
        # Update hand mesh
    
        faces = self.hand.th_faces.detach().numpy().astype(np.uint32)
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
        
        # Update target anchors
        for i, anchor in enumerate(target_anchors):
            self.viz[f"targets/T_{i:02d}"].set_object(
                g.Box([0.002, 0.002, 0.002]),
                g.MeshLambertMaterial(color=self._rgb_to_int(self.colors[i]), opacity=0.5, transparent=True)
            )
            self.viz[f"targets/T_{i:02d}"].set_transform(
                tf.translation_matrix(anchor)
            )
        
        # Move back to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hand.to(device)
    
    def _compute_anchors_vertices(self, pose, shape, base_transform=None):
        """
        Compute anchor points and vertices from pose and shape
        :param pose: MANO pose parameters [3 + n_comp]
        :param shape: MANO shape parameters [10]
        :param base_transform: Optional base transform [rx, ry, rz, tx, ty, tz]
        :return: (anchors [22, 3], vertices [778, 3])
        """
        if base_transform is None:
            base_transform = self.base_transform
        
        # Compute vertices
        pose_tensor = torch.from_numpy(np.asarray(pose)).float().unsqueeze(0)
        shape_tensor = torch.from_numpy(np.asarray(shape)).float().unsqueeze(0)
        
        vertices = self.hand(pose_tensor, shape_tensor).verts[0].numpy()
        
        # Compute anchors from vertices
        a = vertices[self.ach_vert[:, 1]] - vertices[self.ach_vert[:, 0]]
        b = vertices[self.ach_vert[:, 2]] - vertices[self.ach_vert[:, 0]]
        anchors = (a * self.ach_weight[:, 0:1] + 
                  b * self.ach_weight[:, 1:2] + 
                  vertices[self.ach_vert[:, 0]])
        
        # Apply base transform
        translation = np.array(base_transform[3:6]).reshape(1, 3)
        anchors = anchors + translation
        vertices = vertices + translation
        
        return anchors[:-1], vertices  # Return 22 anchors (exclude helper point)
    
    def get_anchor(self, pose=None, shape=None):
        """
        Get anchor positions in world frame
        :param pose: Optional pose override
        :param shape: Optional shape override
        :return: np.ndarray [22, 3]
        """
        if pose is None:
            pose = self.pose
        if shape is None:
            shape = self.shape
        
        anchors, _ = self._compute_anchors_vertices(pose, shape, self.base_transform)
        return anchors
    
    def get_joint(self, all=True, use_scheme=False):
        """
        Get current joint values (pose parameters)
        :param all: Return all parameters (kept for API consistency)
        :param use_scheme: Not used for MANO (kept for API consistency)
        :return: Pose parameters
        """
        if all:
            return self.pose.tolist()
        else:
            return self.pose[3:].tolist()  # Return only pose params (without root rotation)
    
    def get_pose(self):
        """Get the current MANO pose parameters"""
        return self.pose
    
    def get_base(self):
        """
        Get base/end-effector frame transform
        :return: [rx, ry, rz, tx, ty, tz]
        """
        return self.base_transform
    
    def set_base(self, base):
        """
        Set base/end-effector frame transform
        :param base: [rx, ry, rz, tx, ty, tz]
        """
        self.base_transform = [float(x) for x in base]
    
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
    
    def denormalize_tendons(self, tendons):
        """
        Convert normalized tendon values (0-1) to actual joint values
        :param tendons: Normalized tendon values
        :return: Denormalized tendon values
        """
        return self.denormalize_joint(tendons)
    
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
        shape = torch.from_numpy(self.shape).float().unsqueeze(0)  # (1, 10) - shape parameters
        
        # Concatenate root rotation and PCA coefficients
        state = torch.cat((rrot, pose), dim=1)  # (1, 3 + ncomps)
        
        # Forward pass through MANO
        output = self.hand(state, shape)
        joints = output.joints.detach().squeeze(0).cpu().numpy()  # (21, 3)
    
        # Simple approach: Offset wrist by a fixed distance along -Z axis
        wrist = joints[0]
        forearm = wrist.copy()
        forearm[2] -= 0.10  # Move 10cm along negative Z (toward elbow)

        joints_with_forearm = np.vstack([joints, forearm[np.newaxis, :]])
        return joints_with_forearm
        
    def vis_model(self, target_anchors=None, return_image=False):
        """
        Visualize the hand model using MeshCat
        :param target_anchors: Optional target anchor positions to display [22, 3]
        """
  
        # Get current hand state
        rrot_pose = np.concatenate((self.rrot, self.pose))

        anchors, vertices = self._compute_anchors_vertices(
            rrot_pose, self.shape, self.base_transform
        )
        
        # Add hand mesh
        faces = self.hand.th_faces.detach().numpy().astype(np.uint32)
        mesh = g.TriangularMeshGeometry(vertices, faces)
        mesh = g.TriangularMeshGeometry(vertices, faces)
        self.viz["hand"].set_object(mesh, g.MeshLambertMaterial(color=0x505050))
        
        # Add anchors
        for i in range(22):
            self.viz[f"anchors/A_{i:02d}"].set_object(
                g.Sphere(0.005),
                g.MeshLambertMaterial(color=self._rgb_to_int(self.colors[i]))
            )
            self.viz[f"anchors/A_{i:02d}"].set_transform(
                tf.translation_matrix(anchors[i])
            )
        
        # Add target anchors if provided
        if target_anchors is not None:
            for i, anchor in enumerate(target_anchors):
                self.viz[f"targets/T_{i:02d}"].set_object(
                    g.Sphere(0.005),
                    g.MeshLambertMaterial(color=0x00FF00, opacity=0.5, transparent=True)
                )
                self.viz[f"targets/T_{i:02d}"].set_transform(
                    tf.translation_matrix(anchor)
                )
        if return_image:
            # import cv2
            return self.viz.get_image()
            # return  cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)    
        
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
    
    # Random pose test
    print("\nTesting forward kinematics with random pose...")
    random_pose = np.random.randn(mano.dof_tendons) * 0.5
    mano.forward_kinematic(random_pose)
    
    # Visualize initial pose
    print("Visualizing initial pose...")
    mano.vis_model()
    
    # Test inverse kinematics
    print("\nTesting inverse kinematics...")
    target_anchors = mano.get_anchor()
    # Perturb targets slightly
    target_anchors += np.random.randn(*target_anchors.shape) * 0.001
    
    print("Running IK optimization...")
    mano.inverse_kinematic(target_anchors, niter=100, visual=True)
    
    print("Visualizing IK result...")
    mano.vis_model(target_anchors=target_anchors)
    
    # Animate random poses
    print("\nAnimating random poses (5 iterations)...")
    for i in range(5):
        print(f"Iteration {i+1}/5")
        random_pose = np.random.randn(mano.dof_tendons)
        mano.forward_kinematic(random_pose)
        mano.vis_model()
        time.sleep(3)
    
    print("\nDone! Keep the window open to view the final pose.")
    hold()

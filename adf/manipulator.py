
"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import numbers
import sys
from time import time
from xmlrpc import server
import tqdm
import xacro
import tempfile
import numpy as np
from os import path
import yaml
import os
import torch


# Robotics libraries
import pinocchio as pin
from urdf_parser_py.urdf import URDF

from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

# Internal modules (Assumed to exist in your environment)
# sys.path.append(path.dirname(__file__))
from adf.mano_hand import ManoHand

sys.path.append("/home/marco/ros2_ws/src/faive_system/src")  # Adjust this path as needed
from retargeter.retargeter import Retargeter


class Manipulator:
    names = ['robotiq_2f85', 'franka_gripper', 'widowx_gripper',
             'xarm_gripper', 'wsg50_gripper', 'rethink_egripper', 'fetch_gripper',
             'armar_hand_right', 'google_gripper', 'kinova_2f',  'kinova_3f_right',
             'ergocub_hand_right', 'schunk_hand_right', 'allegro_hand_right', 
             'shadow_hand_right', 'leap_hand_right', 'mimic_hand_right', 'orca_v1']

    def __init__(self, model_name, fixed_base=True, verbose=True, headless=False, viewer = None):
        """
        init manipulator class using Pinocchio
        :param model_name: str name of the available model
        :param fixed_base: bool fixed or floating base
        :param verbose: bool verbose model details
        """
        self.verbose = verbose

        if "mimic" in model_name:
            model_name = "mimic_hand_right"
        if "orca" in model_name:
            model_name = "orca_v1"

        print("Loading manipulator model:", model_name)

        assert model_name in Manipulator.names, "{} not found.".format(model_name)
        self.name = model_name
        self.fixed_base = fixed_base

        # --- Load Robot ---
        dir_urdf = path.join(path.dirname(__file__), '../assets', model_name)
        model_urdf_path = path.join(dir_urdf, 'model_klampt.urdf')
        
        # Process Xacro
        self.tmp_urdf_file = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
        urdf_xml = xacro.process_file(model_urdf_path).toprettyxml(indent='  ')
        
        self.tmp_urdf_file.write(urdf_xml)
        self.tmp_urdf_file.close() # Close so other process can read it

        # --- Build Pinocchio Model ---
        if not fixed_base:
            self.model = pin.buildModelFromUrdf(self.tmp_urdf_file.name, pin.JointModelFreeFlyer())
        else:
            self.model = pin.buildModelFromUrdf(self.tmp_urdf_file.name)
        
        self.data = self.model.createData()
        self.collision_model = pin.buildGeomFromUrdf(self.model, self.tmp_urdf_file.name, pin.GeometryType.COLLISION, package_dirs=[dir_urdf])
        self.collision_model.addAllCollisionPairs()
        print("num collision pairs - initial:", len(self.collision_model.collisionPairs))

        self.collision_data = pin.GeometryData(self.collision_model)
        print("Collision model has {} geometries.".format(self.collision_model.ngeoms))
        self.visual_model = pin.buildGeomFromUrdf(self.model, self.tmp_urdf_file.name, pin.GeometryType.VISUAL, package_dirs=[dir_urdf])
        self.visual_data = pin.GeometryData(self.visual_model)
        
        # self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        # self.viz.initViewer(open=True)
        # self.viz.loadViewerModel()
        # self.viz.viewer["/Cameras/default/rotated/<object>"].set_property("zoom", 9.0)


        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        if viewer is not None:
            # Use provided viewer
            self.viz.initViewer(viewer=viewer, open=False)
        else:
            # Create new viewer
            self.viz.initViewer(open=True)
        
        self.viz.loadViewerModel(rootNodeName=model_name)
        
        # Set camera zoom (use viewer directly if provided, otherwise use viz.viewer)
        if viewer is not None:
            viewer["/Cameras/default/rotated/<object>"].set_property("zoom", 9.0)
        else:
            self.viz.viewer["/Cameras/default/rotated/<object>"].set_property("zoom", 9.0)


        # --- Parse URDF for Joint Names Mapping ---
        urdf_parsers = URDF.from_xml_string(urdf_xml)
        self.link_name_to_joint_name = {j.child: j.name for j in urdf_parsers.joints}

        self.base_path = dir_urdf

        # Load scheme if available
        scheme_path = path.join(path.dirname(__file__), '../assets', model_name, 'scheme.yaml')
        if path.exists(scheme_path):
            with open(scheme_path, 'r') as f:
                self.scheme = yaml.safe_load(f)
            self.dof_tendons = len(self.scheme.get('gc_tendons', []))
            print(f"Loaded scheme for {model_name} with {self.dof_tendons} tendons. There are {self.model.nv} DoF in total.")
        else:
            self.scheme = None
            self.dof_tendons = self.model.nv # Default to system DoF and assume direct mapping
            print(f"No scheme found for {model_name}, using direct joint mapping.")
        
        self.q = pin.neutral(self.model)
        
        # Map joint indices to names excluding Universe
        self.joint_names = [n for n in self.model.names if n != "universe"]
        
        self.retargeter = Retargeter(
                    urdf_filepath=self.tmp_urdf_file.name,
                    hand_scheme=scheme_path,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    optimizer='RMSprop'
                )


        # colls = self.get_collision_pairs()
        # print(f"Model {self.name} has {len(colls)} collision pairs.")

        # self.ignored_collision_pairs = colls                

        # verbose
        if verbose:
            print('-------------------------------------------')
            print(f"Model: {self.name}, FixedBase: {fixed_base}")
            print(f"n_q (config size): {self.model.nq}")
            print(f"n_v (velocity size): {self.model.nv}")
            for i, name in enumerate(self.joint_names):
                print(f"Joint {i}: {name}")
            print('-------------------------------------------')

    def get_collision_pairs(self):
        """Return list of collision pairs"""
        pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, self.q, False)
        pairs = []
        for k in range(len(self.collision_model.collisionPairs)):
            cp = self.collision_model.collisionPairs[k]
            cr = self.collision_data.collisionResults[k]
            print(f"Collision pair {k}: {cp.first} - {cp.second}, isCollision: {cr.isCollision()}")
            if self.collision_data.collisionResults[k].isCollision() == 'Yes':
                pairs.append((cp.first, cp.second)) 
        return pairs

    def cleanup(self):
        """cleanup temporary files"""
        if path.exists(self.tmp_urdf_file.name):
            os.remove(self.tmp_urdf_file.name)

    def save_calib_eef(self, base_pos, base_rot):
        file_path = path.join(path.dirname(__file__), "calib_eef.yaml")
        try:
            with open(file_path, "r") as f:
                calib = yaml.safe_load(f) or {}
        except FileNotFoundError:
            calib = {}
        
        base_pos = [float(v) for v in base_pos]
        # Assuming input is Euler/Rotation vector, ensuring list of floats
        base_rot = [float(v) for v in base_rot] 

        calib[self.name] = dict(
            tx=base_pos[0], ty=base_pos[1], tz=base_pos[2],
            rx=base_rot[0], ry=base_rot[1], rz=base_rot[2],
        )

        with open(file_path, "w") as f:
            yaml.safe_dump(calib, f)

    def reset(self):
        """
        reset all joint configuration to init state
        """
        self.q = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def _get_joint_id(self, joint_name):
        if self.model.existJointName(joint_name):
            return self.model.getJointId(joint_name)
        return None

    def forward_kinematic(self, q, use_scheme=True, normalized=True):
        """
        set driver joint values and compute kinematics
        :param q: list of joint values or tendons
        :param use_scheme: bool use scheme mapping if available
        :param normalized: bool if true, input q is normalized (0-1)
        """
        config_vec = self.q.copy()

        # Handle floating base offset
        q_start_idx = 0
        if not self.fixed_base:
            q_start_idx = 7 # 3 pos + 4 quat

        if use_scheme and self.scheme is not None:
            gc_tendons = self.scheme.get('gc_tendons', {}) 
            assert len(q) == len(gc_tendons), (
                'q expected {}, given {}'.format(len(gc_tendons), len(q)))

            
            for tendon_num, (tendon_name, mapping) in enumerate(gc_tendons.items()):
                value = float(q[tendon_num])
                
                # Set the main tendon joint
                jid = self._get_joint_id(tendon_name)
                if jid:
                    idx_q = self.model.joints[jid].idx_q
                    config_vec[idx_q] = value
                
                # Handle coupled joints (mimics)
                if mapping:
                    for mapped_joint, ratio in mapping.items():
                        jid_map = self._get_joint_id(mapped_joint)
                        if jid_map:
                            idx_q_map = self.model.joints[jid_map].idx_q
                            config_vec[idx_q_map] = value * ratio
                            
        else:
            # Direct mapping. Assuming q provided matches model.nv (minus freeflyer if applicable)
            n_actuated = self.model.nv - (6 if not self.fixed_base else 0)
            assert len(q) == n_actuated, f"Expected {n_actuated} joints, got {len(q)}"
            
            for i in range(n_actuated):
                # +q_start_idx handles skipping base params in config_vec
                # logic valid for simple revolute/prismatic joints
                config_vec[q_start_idx + i] = float(q[i])

        self.q = config_vec

        if normalized:
            self.q = self.denormalize_joint(config_vec)
        

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data, self.q)
        # colls = self.get_collision_pairs()
        # print(f"After FK, {len(colls)} collision pairs detected.")

    def get_links_transform(self, link_source, link_target):
        # Update frames first
        pin.updateFramePlacements(self.model, self.data)
        
        try:
            id_source = self.model.getFrameId(link_source)
            id_target = self.model.getFrameId(link_target)
            
            oMsource = self.data.oMf[id_source]
            oMtarget = self.data.oMf[id_target]
            
            # source to target: inv(source) * target
            sourceMtarget = oMsource.actInv(oMtarget)
            return sourceMtarget.homogeneous
        except Exception as e:
            print(f"Error getting link transform: {e}")
            return np.eye(4)

    def get_anchor(self):
        """
        get anchors position A_00 to A_21
        :return: np.ndarray [22, 3]
        """
        pin.updateFramePlacements(self.model, self.data)
        anchors = []
        for i in range(22):
            name = 'A_{:02d}'.format(i)
            if self.model.existFrame(name):
                fid = self.model.getFrameId(name)
                anchors.append(self.data.oMf[fid].translation)
            else:
                # Fallback or error if anchor doesn't exist
                anchors.append(np.zeros(3))
        return np.array(anchors)

    def get_mano(self):
        """
        get mano joint values
        """
        if not hasattr(self, "mano") and ManoHand:
            self.mano = ManoHand()
        if self.mano:
            print("Getting MANO pose for", self.name)
            self.mano.inverse_kinematic(self.get_anchor())
            return self.mano.get_pose()
        return None

    def mano_to_joints(self, mano_pose):
        if not hasattr(self, "mano") and ManoHand:
            self.mano = ManoHand()
        if self.mano:
            print("Getting MANO keypoints for", self.name)
            anchors = self.mano.get_anchor(pose=mano_pose)
            self.inverse_kinematic(anchors)
            return self.get_joint(all=False, use_scheme=True)
        return []

    def get_joint(self, all=True, use_scheme=False):
        """
        get joint values
        """
        offset = 7 if not self.fixed_base else 0
        # Get raw actuated joints
        q_raw = self.q[offset:].tolist()

        if all:
            if self.scheme is not None and use_scheme:
                gc_tendons = self.scheme.get('gc_tendons', {})
                q_mapped = np.zeros(len(gc_tendons))
                
                # Map back from raw joints to scheme tendons
                # This is an approximation/inverse of the Scheme logic
                for tendon_num, tendon_name in enumerate(gc_tendons.keys()):
                    values = []
                    
                    # Find the index of the joint in the raw q vector
                    joint_id = self._get_joint_id(tendon_name)
                    if joint_id:
                        # Map joint Id to q index
                        q_idx = self.model.joints[joint_id].idx_q - offset
                        if 0 <= q_idx < len(q_raw):
                            values.append(q_raw[q_idx])
                    
                    # Add mimics
                    mapping = gc_tendons[tendon_name]
                    for mapped_joint, ratio in mapping.items():
                        m_id = self._get_joint_id(mapped_joint)
                        if m_id:
                            m_q_idx = self.model.joints[m_id].idx_q - offset
                            if 0 <= m_q_idx < len(q_raw) and ratio != 0:
                                values.append(q_raw[m_q_idx] / ratio)
                    
                    if values:
                        q_mapped[tendon_num] = np.mean(values)
                
                return q_mapped.tolist()
            else:
                return q_raw
        else:
            return q_raw

    def get_base(self):
        """
        get base/eef frame [tx, ty, tz, rx, ry, rz]
        Only really meaningful if not fixed base.
        """
        if self.fixed_base:
            return [0.0]*6
        
        # q consists of [x, y, z, qx, qy, qz, qw] for FreeFlyer
        pos = self.q[:3]
        quat = pin.Quaternion(self.q[6], self.q[3], self.q[4], self.q[5]) # scalar is last in pinocchio input
        
        # Convert quat to rpy ? Original code seemed to return config mapping?
        # The original code: [q0, q1, q2, q5, q4, q3] -> Pos, then swapped rotation params
        # NOTE: Original Code assumed specific index mapping. 
        # Standard convention: return SE3 pos+euler
        
        # Assuming original code returned Position and specific euler mapping
        # We will return standard Position + Euler angles (RPY)
        rpy = pin.utils.matrixToRpy(quat.matrix())
        return np.concatenate((pos, rpy)).tolist()

    def set_base(self, q):
        """
        set base eef frame
        q: [tx, ty, tz, rx, ry, rz]
        """
        if self.fixed_base:
            return

        pos = np.array(q[:3])
        rpy = np.array(q[3:6])
        rot = pin.utils.rpyToMatrix(rpy[0], rpy[1], rpy[2])
        quat = pin.Quaternion(rot)
        
        self.q[0:3] = pos
        self.q[3] = quat.x
        self.q[4] = quat.y
        self.q[5] = quat.z
        self.q[6] = quat.w
        
        pin.forwardKinematics(self.model, self.data, self.q)

    def inverse_kinematic_legacy(self, pos_anchor, focus_tip=False, dt=0.5, damp=1e-12, niter=200, th_loss=1e-5, base_pos_weight = 1.0,base_rot_weight = 1.0, visualize=False):
        """
        Numerical Inverse Kinematics using Jacobian Pseudo-Inverse (Isolating Pinocchio logic)
        :param pos_anchor: list of anchor position [x,y,z] (size 22)
        """
        
        if visualize:
            self.vis_model(target_anchors=pos_anchor)


        patience = 5
        patience_counter = 0
        best_loss = float('inf')

        # Define frame IDs for anchors
        frame_ids = []
        target_positions = []
        
        indices_to_track = list(range(len(pos_anchor)))
        
        # Logic from original code: specific weighting for tips vs others
        if focus_tip:
            indices_to_track = [0, 4, 8, 12, 16, 20]  # Tips of each finger + palm base

        for i in indices_to_track:
            name = 'A_{:02d}'.format(i)
            assert self.model.existFrame(name), f"Frame {name} does not exist in the model."
            
            frame_ids.append(self.model.getFrameId(name))
            target_positions.append(np.array(pos_anchor[i]))

        n_targets = len(frame_ids)
        if n_targets == 0:
            return

        # Optimization loop
        q = self.q.copy()
        
        offset = 7 if not self.fixed_base else 0
        n_joints = self.model.nv - offset # joints to optimize (excluding base if mobile)

        proc_bar = tqdm.tqdm(range(niter), desc=f"Fitting {self.name}", unit="it") 

        for it in proc_bar:
        # for _ in range(iters):        
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # Stack Jacobian and Error
            J_stack = []
            err_stack = []
            
            for i in range(n_targets):
                fid = frame_ids[i]
                target = target_positions[i]
                
                curr_pos = self.data.oMf[fid].translation
                err = target - curr_pos # 3D error
                
                # Get Frame Jacobian (linear part only)
                J = pin.computeFrameJacobian(self.model, self.data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_linear = J[:3, :] # Take top 3 rows (translation)
                
                J_stack.append(J_linear)
                err_stack.append(err)
            
            J_all = np.vstack(J_stack)
            err_all = np.hstack(err_stack)
            
            # Add base Constraints so that the base does move more slowly
            if not self.fixed_base:
                base_pos_curr = q[:3]
                base_pos_start = self.q[:3]
                base_rot_curr = q[3:7]    # quaternion in config
                base_rot_start = self.q[3:7]

                M_curr = pin.SE3(pin.Quaternion(base_rot_curr), base_pos_curr)
                M_start = pin.SE3(pin.Quaternion(base_rot_start), base_pos_start)

                # se3 6-vector error: log6( M_curr.inverse() * M_start ).vector
                err_base6 = base_pos_weight * pin.log6(M_curr.inverse() * M_start).vector  # length 6

                J_base = np.zeros((6, self.model.nv))
                J_base[:, :6] = np.eye(6)   # base velocity lives in first 6 velocity DOFs
                J_all = np.vstack([J_all, J_base])
                err_all = np.hstack([err_all, err_base6])
                    
            loss = np.linalg.norm(err_all)
             
            if loss < th_loss:
                print("IK optimization converged.")
                break
             # Early stopping
            if loss + 1e-5 < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
                if patience_counter >= patience:
                    print("Impatient")
                    # break
            
            
            JJt = J_all @ J_all.T
            damp_mat = (damp if np.isscalar(damp) else float(damp)) * np.eye(JJt.shape[0])
            rhs = np.linalg.solve(JJt + damp_mat, err_all)   # solve (J J^T + λI) x = err
            dq = J_all.T @ rhs                                # nq-length

            # integrate and clamp
            q = pin.integrate(self.model, q, dq * dt)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)


            self.vis_model(q=q) if visualize else None
            
            proc_bar.set_description(f"Fitting {self.name} loss: {loss:.5f}")

        self.q = q
        # Ensure final update
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def inverse_kinematic(self, mano_keypoints, visualize=False, 
                                   opt_steps=100, warm=True):
        """
        PyTorch-based IK using retargeter
        pos_anchor: (22, 3) array of anchor positions
        """

        print("Starting PyTorch IK with retargeter...")
        if visualize:
            self.vis_model(target_anchors=mano_keypoints) # ATTENTION THESE ARE NOT THE SAME AS ANCHORS
        
    

        # Convert to numpy if tensor
        if isinstance(mano_keypoints, torch.Tensor):
            mano_keypoints = mano_keypoints.cpu().numpy()
        
        # Use retargeter to solve
        joint_angles, _ = self.retargeter.retarget(
            joints=mano_keypoints,
            debug_dict={}
        )

        joint_angles = [np.deg2rad(ja) for ja in joint_angles]  # Convert to radians if needed

      
        print("Retargeter found joint angles:", joint_angles)
        # Update configuration
        self.forward_kinematic(joint_angles, use_scheme=True, normalized=False)
        
        if visualize:
            self.vis_model(target_anchors=mano_keypoints)
        
        # Compute final error
        final_anchors = self.get_anchor()
        error = np.linalg.norm(mano_keypoints - final_anchors, axis=1)
        print(f"PyTorch IK - Mean error: {error.mean()*1000:.3f}mm, "
              f"Max error: {error.max()*1000:.3f}mm")
        
        return joint_angles

    def denormalize_joint(self, qn):
        """
        convert normalized values (0-1) to actual joint values based on limits 
        """
        offset = 0 if self.fixed_base else 7
        n_actuated = self.model.nq - offset

        qn = np.asarray(qn)
        assert qn.shape[0] == n_actuated, "Wrong number of joint values"

        lower = np.asarray(self.model.lowerPositionLimit[offset:])
        upper = np.asarray(self.model.upperPositionLimit[offset:])

        q = lower + (upper - lower) * qn
        return q    
    
    def denormalize_tendons(self, tendons):
        """
        convert normalized tendon values (0-1) to actual joint values based on scheme limits 
        """
        assert self.scheme is not None, "Scheme not defined for this model."
        gc_tendons = self.scheme.get('gc_tendons', {})
        assert len(tendons) == len(gc_tendons), (
            'tendons expected {}, given {}'.format(len(gc_tendons), len(tendons)))
        
        q = []
        for tendon_num, (tendon_name, mapping) in enumerate(gc_tendons.items()):
            jid = self._get_joint_id(tendon_name)
            if jid:
                idx_q = self.model.joints[jid].idx_q
                lower = self.model.lowerPositionLimit[idx_q]
                upper = self.model.upperPositionLimit[idx_q]
                val = lower + (upper - lower) * tendons[tendon_num]
                q.append(val)
            else:
                q.append(0.0) # Default if joint not found
        return q


    def vis_model(self, q=None, target_anchors=None, return_image=False):
        """
        Visualize model using Meshcat
        """
        
        if q is not None:
            q_viz = q
        else:
            q_viz = self.q

            
        
        self.viz.display(q_viz)

        pin.forwardKinematics(self.model, self.data, q_viz)
        pin.updateFramePlacements(self.model, self.data)
        # pin.updateGeometryPlacements(self.model, self.visual_model, self.data, self.viz)
        
        # Visualize Anchors
        try:
            for i in range(22):
                name = "A_{:02d}".format(i)
                if self.model.existFrame(name):
                    fid = self.model.getFrameId(name)
                    # print(f"Visualizing anchor {name} at frame id {fid}")
                    pos = self.data.oMf[fid].translation
                    # print(f"Anchor {name} position: {pos}")
                    self.viz.viewer[f"anchors/{name}"].set_object(
                        g.Sphere(0.005), 
                        g.MeshLambertMaterial(color=self._rgb_to_int(self.colors[i]))
                    )
                    self.viz.viewer[f"anchors/{name}"].set_transform(tf.translation_matrix(pos))
        except Exception as e:
            print(f"Error visualizing anchors: {e}")

        # Visualize Target Anchors
        if target_anchors is not None:
            for idx, anchor in enumerate(target_anchors):
                obj_name = f"target_anchors/{idx}"
                self.viz.viewer[obj_name].set_object(
                    g.Box([0.003, 0.003, 0.003]),
                    g.MeshLambertMaterial(color=self._rgb_to_int(self.colors[idx]), opacity=0.5, transparent=True)
                )
                self.viz.viewer[obj_name].set_transform(tf.translation_matrix(anchor))
        if return_image:
            # import cv2
            return self.viz.viewer.get_image()
            # return  cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

    def _rgb_to_int(self, rgb_norm):
        """Helper to convert 0-1 float RGB to hex integer"""
        r, g, b = [int(c * 255) for c in rgb_norm]
        return (r << 16) | (g << 8) | b

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
    """Keep the program running to view Meshcat"""
    import time
    try:
        print("Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    manip = Manipulator(model_name="mimic_hand_right")
    import time
    img = manip.vis_model(return_image=True)
    img.save("manipulator_view.png")
    # save img 
    # with open("manipulator_view.png", "wb") as f:
    #     f.write(img)
    


    while True:
        tendons = np.random.uniform(low=0.0, high=1.0, size=manip.dof_tendons)
        manip.forward_kinematic(tendons, use_scheme=True, normalized=True)
        manip.vis_model()
        time.sleep(1)



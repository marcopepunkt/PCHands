"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause

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
palm'                 |  22
-----------------------------------------
"""
import sys
import tqdm
import yaml
import torch
import numpy as np
from os import path
from klampt.math import se3
from manotorch.manolayer import ManoLayer
from klampt import vis, TriangleMesh, GeometricPrimitive
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import invert_transform, transform_from, vectors_to_points, transform
sys.path.append(path.dirname(__file__))


def rotmat_to_euler_xyz(R):
    # R: 3x3 rotation matrix
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return x,y,z

class ManoHand:
    def __init__(self, dir_assets=None, use_pca=True, n_comp=12, flat_hand=False, hand_side='right', calibrated=True):
        """
        init mano hand class
        :param dir_assets: mano assets directory: assets/mano_hand/models/MANO_RIGHT.pkl
        :param use_pca: is pose in pca-space, otherwise joint-space
        :param n_comp: number of pca components
        :param flat_hand: if flat hand is mode used
        """
        torch.set_num_threads(1)

        if dir_assets is None:
            dir_assets = path.join(path.dirname(__file__), '../assets', 'mano_hand')
        self.use_pca = use_pca
        self.n_comp = n_comp

        # instances
        self.ach_vert = np.loadtxt(path.join(dir_assets, "anchor/anchor_vertex.txt"), dtype=int)
        self.ach_weight = np.loadtxt(path.join(dir_assets, "anchor/anchor_weight.txt"))
        self.hand = ManoLayer(mano_assets_root=dir_assets, side=hand_side, center_idx=None,
                              flat_hand_mean=flat_hand, rot_mode='axisang',
                              use_pca=use_pca, ncomps=n_comp)
        self.name = 'mano_hand'


        try:
            if calibrated:
                calib_eef = yaml.safe_load(open(path.join(path.dirname(__file__), 'calib_eef.yaml'), 'r'))['mano_hand']
                self.calib_eef = [calib_eef['rx'], calib_eef['ry'], calib_eef['rz'], calib_eef['tx'], calib_eef['ty'], calib_eef['tz']]
            else:
                self.calib_eef = [0, 0, 0, 0, 0, 0]
        except:
            print('No calib_eef.yaml loaded!')
            self.calib_eef = [0, 0, 0, 0, 0, 0]

        # rendering
        self.mesh = TriangleMesh()
        self.mesh.setIndices(self.hand.th_faces.detach().numpy().copy().astype(np.int32))
    

    

    def save_calib_eef(self, base_pos, base_rot):
        file_path = path.join(path.dirname(__file__), "calib_eef.yaml")

        # load existing yaml
        try:
            with open(file_path, "r") as f:
                calib = yaml.safe_load(f) or {}
        except FileNotFoundError:
            calib = {}
        
        base_pos = [float(v) for v in base_pos]
        base_rot = [float(v) for v in base_rot]

        # update only this key
        calib[self.name] = dict(
            tx=base_pos[0], ty=base_pos[1], tz=base_pos[2],
            rx=base_rot[0], ry=base_rot[1], rz=base_rot[2],
        )

        # write back
        with open(file_path, "w") as f:
            yaml.safe_dump(calib, f)
            
    def pose_to_anchor(self, pose, shape=None, apply_calib_eef=False):
        """
        convert mano pose to anchor
        :param pose: np.ndarray [48] or [3 + n_comp]
        :param shape: np.ndarray [10]
        :param return_vert: bool return vertex
        :param return_palm_frame: bool anchor in palm frame
        :return: np.ndarray [22, 3] anchors, and/or np.ndarray [778, 3] vertex
        """
        if self.use_pca:
            assert pose.shape[0] == 3 + self.n_comp
        else:
            assert pose.shape[0] == 48
        if shape is None:
            shape = np.zeros(10)

        # pose to vertex
        v = self.hand(torch.from_numpy(pose)[None].float(),
                      torch.from_numpy(shape)[None].float()).verts[0].numpy()

        # vertex to anchor
        a = self.vertex_to_anchor(v)

        
        if apply_calib_eef:
            t = np.array(self.calib_eef[3:6]).reshape(1, 3)  # shape (1,3)
            a = a + t
            v = v + t
        return a, v


    def pose_to_anchor_absolute(self, pose, shape, tsl):
        """
        convert pose to anchor in frame of original palm frame
        Args:
            pose torch.tensor [b, 22, 3]: mano pose
            shape torch.tensor [b, 10]: mano shape
            tsl torch.tensor [b, 1, 3]: mano translation

        Returns:
            np.array [b, 22, 3]: anchor in the original palm frame
        """
        v = (self.hand(pose, shape).verts + tsl).numpy()
        a = v[:, self.ach_vert[:, 1]] - v[:, self.ach_vert[:, 0]]
        b = v[:, self.ach_vert[:, 2]] - v[:, self.ach_vert[:, 0]]
        anc = a * self.ach_weight[:, 0:1][None] + b * self.ach_weight[:, 1:2][None] + v[:, self.ach_vert[:, 0]]
        a0 = []
        for a_ in anc:
            tf1 = self.anchor_transform(a_, return_tf=True)
            a1 = transform(invert_transform(tf1), vectors_to_points(a_[:22]))
            a0_ = transform(np.matmul(self.palm_tf0_inv, tf1), a1)[:, :3]
            a0.append(a0_)
        return np.stack(a0)

    def pose_to_anchor_world(self, pose, shape, c2h_t=None, w2c=None):
        """
        convert mano pose to anchor in world frame
        :param pose: np.ndarray [48] or [3 + n_comp] hand pose
        :param shape: np.ndarray [10] hand shape
        :param c2h_t: np.ndarray [3] camera to hand translation
        :param w2c: np.ndarray [4, 4] world tp camera transformation, or None (camera is world frame)
        :return: np.ndarray [22, 3] anchor in world frame, np.ndarray [778, 3] vectex,
        np.ndarray [4, 4] world to palm frame
        """
        # world anchor and vertex
        a, v = self.pose_to_anchor(pose, shape, return_vert=True, return_palm_frame=False)
        # camera frame translation
        if c2h_t is not None:
            a = a + c2h_t
            v = v + c2h_t
        # world frame transformation
        if w2c is not None:
            a = np.matmul(w2c, np.vstack((a.T, np.ones((1, a.shape[0]))))).T[:, :-1]
            v = np.matmul(w2c, np.vstack((v.T, np.ones((1, v.shape[0]))))).T[:, :-1]
        return a, v, self.anchor_transform(a, return_tf=True)

    def vertex_to_anchor(self, vert):
        """
        convert mano right hand vertex to anchor points
        :param vert: np.ndarray [778, 3] vertex
        :return: np.ndarray [22, 3] anchor
        """
        a = vert[self.ach_vert[:, 1]] - vert[self.ach_vert[:, 0]]
        b = vert[self.ach_vert[:, 2]] - vert[self.ach_vert[:, 0]]
        anchor = a * self.ach_weight[:, 0:1] + b * self.ach_weight[:, 1:2] + vert[self.ach_vert[:, 0]]
        return anchor

    def anchor_transform(self, anchor, vert=None, return_tf=False):
        """
        forming a frame with anchor20,21,22:
        pinkie     index
            21 . . 20
             .    .
              .  .
               22
              wrist
        then transform anchor from mesh center frame to the new fixed vertex frame
        :param anchor: np.ndarray [23, 3]
        :param vert: np.ndarray [778, 3] or None
        :param return_tf: bool to only return world to palm frame
        :return: transformed anchor np.ndarray [22, 3], and/or transformed vert np.ndarray [778, 3],
        or palm frame np.ndarray [4, 4]
        """
        # rotation
        xa = np.cross(anchor[20] - anchor[22], anchor[21] - anchor[22])
        xa = xa / np.linalg.norm(xa)
        za = anchor[20] - anchor[21]
        za = za / np.linalg.norm(za)
        ya = np.cross(za, xa)
        # translation
        ori = 0.5 * (anchor[20] + anchor[21])
        # transformation 4x4
        tf = np.eye(4)
        tf[:3, 3] = ori
        tf[:3, 0] = xa
        tf[:3, 1] = ya
        tf[:3, 2] = za
        # z-rot correction
        zrot = np.eye(4)
        zrot[:3, :3] = matrix_from_euler((0, 0, -np.deg2rad(0)), 0, 1, 2, True)
        w2palm = np.matmul(tf, zrot)
        # calib eef
        w2eef = np.matmul(w2palm, invert_transform(self.calib_eef))  # world2palm * inv(eef2palm)
        if return_tf:
            return w2eef

        # transform anchor from world to eef frame: inv(world2eef) * world2anc
        eef2w = invert_transform(w2eef)
        anchor_ = transform(eef2w, vectors_to_points(anchor[:22]))[:, :3]
        if vert is not None:
            vert_ = transform(eef2w, vectors_to_points(vert))[:, :3]
            return anchor_, vert_
        else:
            return anchor_

    def inverse_kinematic(self,*args, **kwargs):
         
        self.pose, self.shape, self.calib_eef =self.anchor_to_pose(*args,**kwargs)

    def anchor_to_pose(self, anchor, niter=100, lr=5e-2, wd=1e-4, th_loss=0.00008, visual=False, floating_base=False, hotstart=True, **kwargs):
        """
        retrieve pose and shape from anchor
        :param anchor: np.ndarray [22x3]
        :param niter: number of iteration
        :param lr: learning rate
        :param wd: weight decay
        :param visual: visualization of the learning
        :return: pose: numpy.ndarray [?]
                 shape: numpy.ndarray [10]
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hand.to(device)


        if visual:
            vis.setWindowTitle("Visualization")
            vp = vis.getViewport()
            vp.camera.dist = 0.5
            vis.setViewport(vp)
            # Add the target anchors to the visualizer
            for i, single_anchor in enumerate(anchor):
                name = "target_anchor_{}".format(i)
                anc = GeometricPrimitive()
                anc.setSphere(single_anchor, 0.001)
                vis.add(name, anc, hide_label=True)
                vis.setColor(name, *self.colors[i])
            
        # setting
        ach_vert = torch.from_numpy(self.ach_vert[:-1]).long().to(device)
        ach_weight = torch.from_numpy(self.ach_weight[:-1]).float().to(device)
        anchor = torch.from_numpy(anchor).float().to(device)
        # parameters
        if floating_base:
            if hasattr(self, '_prev_rtsl') and hotstart:
                rtsl = self._prev_rtsl.clone().to(device).requires_grad_(True)
                rrot = self._prev_rrot.clone().to(device).requires_grad_(True)

            else:
                rrot = torch.randn((1, 3)).to(device) * 1e-3
                rtsl = anchor.mean(dim=0, keepdim=True).to(device)
                rrot.requires_grad_(True)
                rtsl.requires_grad_(True)

        else:
            base = torch.tensor(self.calib_eef, dtype=torch.float32, device=device)
            rrot, rtsl = base[:3].unsqueeze(0), base[3:6].unsqueeze(0)  # shape [1,3]        
        # Warm Start from previous optimization
        if hasattr(self, "_prev_pose") and hotstart:
            pose = self._prev_pose.clone().to(device).requires_grad_(True)
        else:
            pose = torch.randn(1, self.n_comp if self.use_pca else 45).to(device) * 1e-3
            # pose is 15*3 = 45 (15 Joints with 3 DoF each) for non-pca
            # Or just the n_comp for pca

        shape = torch.randn(1, 10).to(device) * 1e-3

        pose.requires_grad_(True)
        shape.requires_grad_(True)
        # optim
        optim = torch.optim.AdamW([
            {"params": [rrot, rtsl], "weight_decay": 0},
            {"params": [pose], "weight_decay": wd, "lr": lr},
            {"params": [shape], "weight_decay": wd, "lr": lr * 0.1},
        ], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, factor=0.5, patience=25, verbose=False
        )
        # iterate
        proc_bar = tqdm.tqdm(range(niter))

        fingertip_indices = [7, 11, 15, 19]
        thumb_tip_indices = [3]
        palm_indices = [20, 21]

        patience = 10
        best_loss = float('inf')
        counter = 0

        for _ in proc_bar:
        # for _ in range(niter):
            optim.zero_grad()
            vertex = self.hand(torch.cat((rrot, pose), dim=1), shape).verts + rtsl # This  computes all vertices for the hand
            a = vertex[0, ach_vert[:, 1]] - vertex[0, ach_vert[:, 0]]
            b = vertex[0, ach_vert[:, 2]] - vertex[0, ach_vert[:, 0]]
            anchor_ = a * ach_weight[:, 0:1] + b * ach_weight[:, 1:2] + vertex[0, ach_vert[:, 0]]

            assert anchor.shape == anchor_.shape

            weights = torch.zeros_like(anchor_)
            weights[fingertip_indices] = 10.0  # Higher weight for fingertips
            weights[palm_indices] = 15.0   # Higher weight for palm points
            weights[thumb_tip_indices] = 5.0  # Higher weight for thumb tip

            lossl1 = torch.nn.functional.smooth_l1_loss(anchor_, anchor, reduction='none') # + 0.0001 * torch.sum(pose ** 2)
            loss = (lossl1 * weights).mean() + 0.01 * torch.sum(shape ** 2) #+ 0.00001 * torch.sum(pose ** 2)
            
            # temporal regularization to smooth the result
            if hasattr(self, '_prev_rtsl'):
                rtsl_diff = torch.norm(rtsl - self._prev_rtsl.to(device))
                rrot_diff = torch.norm(rrot - self._prev_rrot.to(device))
                rtsl_reg = 0.1 * rtsl_diff
                rrot_reg = 0.1 * rrot_diff
                loss = loss + rtsl_reg + rrot_reg

            loss.backward()
            optim.step()
            scheduler.step(loss.item())
            proc_bar.set_description(f"loss: {loss.item():.5f} rtsl: {rtsl.detach().cpu().numpy().flatten()} ")
            # print(f"uptated rtsl : {rtsl}")
            # rendering
            if visual:
                self.hand.to('cpu')
                self.calib_eef = rrot.detach().cpu().flatten().tolist() + rtsl.detach().cpu().flatten().tolist()
                a, v = self.pose_to_anchor(torch.cat((rrot.cpu(), pose.cpu()), dim=1).detach().numpy().flatten(),
                                           shape.cpu().detach().numpy().flatten(), apply_calib_eef=True)
                self.mesh.setVertices(v)
                vis.add('robot', self.mesh)
                vis.hideLabel('robot')
                vis.setBackgroundColor(0.7, 0.7, 0.7)
                for i in range(22):
                    name = "A_{:02d}".format(i)
                    anc = GeometricPrimitive()
                    anc.setSphere(a[i], 0.004)
                    vis.add(name, anc)
                    vis.setColor(name, *self.colors[i])
                    vis.hideLabel(name)
                vp = vis.getViewport()
                vp.w = 800
                vp.h = 800
                vis.setViewport(vp)
                vis.setColor('robot', 0.7, 0.7, 0.7)
                vis.show()
                self.hand.to(device)
            # if loss.item() < th_loss:
            #     break
            if loss.item() < best_loss:
                best_loss = loss.item()
                counter = 0
            else:
                counter += 1
            
            if counter >= patience:
                print("Stopping early: model on plateau")
                break
        
        self._prev_pose = pose.detach().cpu()
        self._prev_shape = shape.detach().cpu()
        self._prev_rtsl = rtsl.detach().cpu()
        self._prev_rrot = rrot.detach().cpu()
        self.hand.to('cpu')

        base = rrot.detach().cpu().flatten().tolist() + rtsl.detach().cpu().flatten().tolist()
        return torch.cat((rrot, pose), dim=1)[0].detach().cpu().numpy(), shape[0].detach().cpu().numpy(), base

    def get_pose(self):
        """
        get the current mano pose
        :return: np.ndarray [?]
        """
        return self.pose

    def joint_to_pose(self, joint, params=None, niter=1000, lr=1e-1, wd=1e-6, visual=False):
        """
        retrieve pose and shape from joint
        :param joint: np.ndarray [21x3]
        :param niter: number of iteration
        :param lr: learning rate
        :param wd: weight decay
        :param visual: visualization of the learning
        :return: pose: numpy.ndarray [48]
                 shape: numpy.ndarray [10]
        """

        if visual:
            vis.setWindowTitle("Visualization")
            vp = vis.getViewport()
            vp.camera.dist = 0.5
            vis.setViewport(vp)
        # parameters
        joint = torch.from_numpy(joint).float()
        if params is None:
            rrot = torch.randn((1, 3)) * 1e-1
            rtsl = joint.mean(dim=0, keepdim=True)
            # rrot = torch.tensor([[-0.9709,  1.4544, -0.2404]])
            # rtsl = torch.tensor([[-0.1382, -0.3717,  0.2979]])
            pose = torch.randn(1, self.n_comp if self.use_pca else 45) * 1e-3
            shape = torch.randn(1, 10) * 1e-2
        else:
            rrot, rtsl, pose, shape = params
        rrot.requires_grad_(True)
        rtsl.requires_grad_(True)
        pose.requires_grad_(True)
        shape.requires_grad_(True)
        # optim
        optim = torch.optim.AdamW([
            {"params": [rrot, rtsl],
             "weight_decay": 0,
             "lr": 0.1},
            {"params": [pose, shape],
             "weight_decay": wd,
             "lr": lr}
        ])

        # iterate
        proc_bar = tqdm.tqdm(range(niter))
        for _ in proc_bar:
            optim.zero_grad()
            joint_ = self.hand(torch.cat((rrot, pose), dim=1), shape).joints[0] + rtsl
            loss = torch.nn.functional.smooth_l1_loss(joint_, joint)
            loss.backward()
            optim.step()
            proc_bar.set_description(f"loss: {loss.item():.5f}")

            # rendering
            if loss.item() < 1e-4:
                if visual:
                    a, v = self.pose_to_anchor(torch.cat((rrot, pose), dim=1).detach().numpy().flatten(),
                                               shape.detach().numpy().flatten(), return_vert=True,
                                               return_palm_frame=True)
                    self.mesh.setVertices(v)
                    vis.add('robot', self.mesh)
                    vis.hideLabel('robot')
                    vis.setBackgroundColor(0.7, 0.7, 0.7)
                    for i in range(22):
                        name = "A_{:02d}".format(i)
                        anc = GeometricPrimitive()
                        anc.setSphere(a[i], 0.004)
                        vis.add(name, anc)
                        vis.setColor(name, *self.colors[i])
                        vis.hideLabel(name)
                    vp = vis.getViewport()
                    vp.w = 800
                    vp.h = 800
                    vis.setViewport(vp)
                    vis.setColor('robot', 0.7, 0.7, 0.7)
                    vis.show()
                break
        print('ik residual:', loss.item())
        return (rrot, rtsl, pose, shape)

    def get_base(self):
        """
        get the base transformation from world to palm frame
        :return: [tx,ty,tz,rx,ry,rz]
        """
        
        return self.calib_eef

    def get_anchor(self, pose=None, shape=None):
        """
        get anchor in world frame
        :return: np.ndarray [22, 3] anchor
        """
        if pose is None and not hasattr(self, "pose"):
            raise ValueError("Please provide pose for getting anchor!")
        if shape is None and not hasattr(self, "shape"):
            shape = np.zeros(10)

        a, _ = self.pose_to_anchor(pose if pose is not None else self.pose, shape if shape is not None else self.shape, apply_calib_eef=True)
        return a[:-1]

    def vis_model(self, pose=None, shape=None, save=None, cam_t=None, cam_r=[0, -1.57, -1.57], cam_dist=0.6, target_anchors=None):
        """
        visualize hand and anchor
        :param pose: None or np.ndarray [48] or [3 + n_comp] pose of Mano
        :param shape: None or np.ndarray [10] shape of Mano
        :param c2h_t: np.ndarray [3] camera to hand translation
        :param w2c: np.ndarray [4, 4] world tp camera transformation, or None (camera is world frame)
        :param save: image saving path
        :param cam_t: list of float [x, y, z] visual translation
        :param cam_r: list of float [r, p, y] visual rotation
        :param cam_dist: float visual distance
        :return: None
        """
        # init
        try:
            if pose is None:
                pose = self.pose
            if shape is None:
                shape = self.shape
        except:
            raise ValueError("Please provide pose and shape for visualization!")    

        # anchor and vertex
        a, v = self.pose_to_anchor(pose, shape, apply_calib_eef=True)
        # rendering
        vis.setWindowTitle("Visualization")
        vis.setBackgroundColor(1, 1, 1)
        vis.add('world', se3.identity(), fancy=True, length=0.05, width=0.004, hide_label=True)
        self.mesh.setVertices(v)
        vis.add('robot', self.mesh, hide_label=True)
        vis.setColor('robot', 0.7, 0.6, 0.6)
        for i in range(22):
            name = "A_{:02d}".format(i)
            anc = GeometricPrimitive()
            anc.setSphere(a[i], 0.005)
            vis.add(name, anc, hide_label=True)
            vis.setColor(name, *self.colors[i])
        
        if target_anchors is not None:
            for anchor in target_anchors:
                name = "target_anchor_{}".format(np.random.randint(0, 1e6))
                anc = GeometricPrimitive()
                anc.setSphere(anchor, 0.005)
                vis.add(name, anc, hide_label=True)
                vis.setColor(name, 0, 1, 0, 0.5)

        vp = vis.getViewport()
        vp.camera.ori = ['z', 'x', 'y']
        vp.camera.dist = cam_dist
        if cam_t is not None:
            vp.camera.tgt = cam_t
        if cam_r is not None:
            vp.camera.rot = cam_r
        vp.w = 800
        vp.h = 800
        vis.setViewport(vp)
        if save is None:
            vis.dialog()
        elif save == 'return':
            vis.show()
            return vis.screenshot('numpy')
        else:
            vis.show()
            vis.spin(1.0)
            vis.screenshot('Image').save(save)

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

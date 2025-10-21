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
from cvae import AnchorAE


class ManoHand:
    def __init__(self, dir_assets=None, use_pca=True, n_comp=30, flat_hand=False, hand_side='right'):
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
            f_stats = path.join(path.dirname(__file__), 'stats.npy')
            self.stats = np.load(f_stats, allow_pickle=True).item()['mano_hand']
            self.pca = AnchorAE(load_model='pca.pth')
        except:
            print('PCA data not available!')
            self.pca = None
            self.stats = None

        try:
            calib_eef = yaml.safe_load(open(path.join(path.dirname(__file__), 'calib_eef.yaml'), 'r'))['mano_hand']
            self.calib_eef = transform_from(
                matrix_from_euler((calib_eef['rx'], calib_eef['ry'], calib_eef['rz']), 0, 1, 2, True),
                (calib_eef['tx'], calib_eef['ty'], calib_eef['tz']))
        except:
            print('No calib_eef.yaml loaded!')
            self.calib_eef = np.eye(4)

        # rendering
        self.mesh = TriangleMesh()
        self.mesh.setIndices(self.hand.th_faces.detach().numpy().copy().astype(np.int32))

    def pose_to_anchor(self, pose, shape=None, return_vert=False, return_palm_frame=True):
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
        # anchor global frame to palm frame
        if return_vert and return_palm_frame:
            return self.anchor_transform(a, v)
        elif return_vert:
            return a, v
        elif return_palm_frame:
            return self.anchor_transform(a)
        else:
            return a

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

    def anchor_to_pose(self, anchor, niter=4000, lr=1e-3, wd=1e-4, th_loss=0.0001, visual=False):
        """
        retrieve pose and shape from anchor
        :param anchor: np.ndarray [22x3]
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
        # setting
        ach_vert = torch.from_numpy(self.ach_vert[:-1]).long()
        ach_weight = torch.from_numpy(self.ach_weight[:-1]).float()
        anchor = torch.from_numpy(anchor).float()
        # parameters
        rrot = torch.randn((1, 3)) * 1e-3
        rtsl = anchor.mean(dim=0, keepdim=True)
        pose = torch.randn(1, self.n_comp if self.use_pca else 45) * 1e-3
        shape = torch.randn(1, 10) * 1e-3
        rrot.requires_grad_(True)
        rtsl.requires_grad_(True)
        pose.requires_grad_(True)
        shape.requires_grad_(True)
        # optim
        optim = torch.optim.AdamW([
            {"params": [rrot, rtsl],
             "weight_decay": 0},
            {"params": [pose, shape],
             "weight_decay": wd}],
            lr=lr)

        # iterate
        # proc_bar = tqdm.tqdm(range(niter))
        # for _ in proc_bar:
        for _ in range(niter):
            optim.zero_grad()
            vertex = self.hand(torch.cat((rrot, pose), dim=1), shape).verts + rtsl
            a = vertex[0, ach_vert[:, 1]] - vertex[0, ach_vert[:, 0]]
            b = vertex[0, ach_vert[:, 2]] - vertex[0, ach_vert[:, 0]]
            anchor_ = a * ach_weight[:, 0:1] + b * ach_weight[:, 1:2] + vertex[0, ach_vert[:, 0]]
            loss = torch.nn.functional.smooth_l1_loss(anchor_, anchor, reduction='none')
            loss = loss.mean()
            loss.backward()
            optim.step()
            # proc_bar.set_description(f"loss: {loss.item():.5f}")

            # rendering
            if visual:
                a, v = self.pose_to_anchor(torch.cat((rrot, pose), dim=1).detach().numpy().flatten(),
                                           shape.detach().numpy().flatten(), return_vert=True, return_palm_frame=True)
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
            if loss.item() < th_loss:
                break
        return torch.cat((rrot, pose), dim=1)[0].detach().numpy(), shape[0].detach().numpy()

    def anchor_to_pose_batch(self, anchor, niter=1000, lr=1e-1, wd=1e-3):
        """
        retrieve pose and shape from anchor
        :param anchor: np.ndarray [b, 22, 3]
        :param niter: number of iteration
        :param lr: learning rate
        :param wd: weight decay
        :param visual: visualization of the learning
        :return: pose: numpy.ndarray [b, 48]
                 shape: numpy.ndarray [b, 10]
        """
        # setting
        ach_vert = torch.from_numpy(self.ach_vert[:-1]).long()
        ach_weight = torch.from_numpy(self.ach_weight[:-1]).float()
        anchor = torch.from_numpy(anchor).float()
        n_b = anchor.shape[0]
        # parameters
        rrot = torch.randn((n_b, 3)) * 1e-2
        rtsl = anchor.mean(dim=1, keepdim=True)
        pose = torch.randn(n_b, self.n_comp if self.use_pca else 45) * 1e-3
        shape = torch.randn(n_b, 10) * 1e-2
        rrot.requires_grad_(True)
        rtsl.requires_grad_(True)
        pose.requires_grad_(True)
        shape.requires_grad_(True)
        # optim
        optim = torch.optim.AdamW([
            {"params": [rrot, rtsl],
             "weight_decay": 0,
             "lr": lr},
            {"params": [pose, shape],
             "weight_decay": wd,
             "lr": lr}])

        # iterate
        for _ in range(niter):
            optim.zero_grad()
            vertex = self.hand(torch.cat((rrot, pose), dim=1), shape).verts + rtsl
            a = vertex[:, ach_vert[:, 1]] - vertex[:, ach_vert[:, 0]]
            b = vertex[:, ach_vert[:, 2]] - vertex[:, ach_vert[:, 0]]
            anchor_ = a * ach_weight[:, 0:1][None] + b * ach_weight[:, 1:2][None] + vertex[:, ach_vert[:, 0]]
            loss = torch.nn.functional.smooth_l1_loss(anchor_, anchor)
            loss.backward()
            optim.step()
            if loss.item() < 0.00001:
                break
        return torch.cat((rrot, pose), dim=1).detach(), shape.detach(), rtsl.detach()

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

    def anchor_to_pc(self, anchor, n_pc=None):
        """
        convert anchor into principal components
        :param anchor: np.ndarray [22x3]
        :param n_pc: int number of principal components to use; None = use all
        :return: np.ndarray [66] principal components
        """
        anchor = (anchor - self.stats['means']) / self.stats['stds']
        pc = self.pca.transform(anchor.flatten()[None], 'mano_hand')[0]
        if n_pc is not None:
            pc[n_pc:] = 0
        return pc

    def pc_to_anchor(self, pc):
        """
        convert principal components to anchors
        :param pc: np.ndarray[66] principal components
        :return: np.ndarray[22x3] anchors
        """
        pc = np.hstack((pc, np.zeros(self.pca.num_components - pc.shape[0])))
        anchor = self.pca.inverse_transform(pc[None], 'mano_hand').reshape(-1, 3)
        anchor = anchor * self.stats['stds'] + self.stats['means']
        return anchor

    def vis_model(self, pose=None, shape=None, save=None, cam_t=None, cam_r=[0, -1.57, -1.57], cam_dist=0.6):
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
        if pose is None:
            pose = np.zeros((3 + self.n_comp) if self.use_pca else 48)
        if shape is None:
            shape = np.zeros(10)

        # anchor and vertex
        a, v = self.pose_to_anchor(pose, shape, return_vert=True, return_palm_frame=True)
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

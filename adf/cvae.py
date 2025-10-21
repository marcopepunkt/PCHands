"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
"""
import wandb
import torch
import pickle
import datetime
from os import path
import torch.nn as nn
from torch.optim import AdamW
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import ReduceLROnPlateau


class AnchorAE:
    def __init__(self, dim_outer=66, dim_inner=(128, 10), n_lyr=2,
                 lr=1e-3, wd=5e-3, a_kl=1e-5, num_epoch=1500, size_batch=2048,
                 labels=None, load_model=None, reuse_model=False):
        torch.set_num_threads(1)

        self.lr = lr
        self.wd = wd
        self.a_kl = a_kl
        self.num_epoch = num_epoch
        self.size_batch = size_batch
        self.num_components = dim_inner[-1]
        self.means, self.stds = None, None
        self.labels = torch.load(path.join(path.dirname(__file__), load_model),
                                 map_location='cpu', weights_only=False)['labels'] if labels is None else labels

        # reconstruct weight
        self.w_rcs = torch.ones(22, 3)
        for idx in [0, 1, 2, 3]:  # thumb
            self.w_rcs[idx] += 2.0
        for idx in [3, 7, 11, 15, 19]:  # tips
            self.w_rcs[idx] += 2.0
        for idx in [20, 21]:  # palm
            self.w_rcs[idx] -= 0.5
        self.w_rcs = self.w_rcs.div(self.w_rcs.sum()).mul(66).flatten()

        # create/load model
        self.create_model(dim_outer, dim_inner, n_lyr)
        if load_model is not None:
            self.load_dict(path.join(path.dirname(__file__), load_model))
        if reuse_model:
            try:
                self.load_dict(path.join(path.dirname(__file__), 'pca.pth'))
                print('Model reused!')
            except:
                print('Failed to reuse model!')

    def create_model(self, dim_outer, dim_inner, n_lyr):
        self.encoder = nn.Sequential(
            nn.Linear(dim_outer + len(self.labels), dim_inner[0]),
            ResMlp(dim_inner[0], n_lyr=n_lyr, drop=0))
        self.fc_mu = nn.Linear(dim_inner[0], dim_inner[-1])
        self.fc_var = nn.Linear(dim_inner[0], dim_inner[-1])
        self.decoder = nn.Sequential(
            nn.Linear(dim_inner[-1] + len(self.labels), dim_inner[0]),
            ResMlp(dim_inner[0], n_lyr=n_lyr, drop=0),
            nn.Linear(dim_inner[0], dim_outer))

        # pca
        self.pca = PCA()

    def save_dict(self, f_dir):
        """
        :param f_dir: file name
        :return: None
        """
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'fc_mu': self.fc_mu.state_dict(),
            'fc_var': self.fc_var.state_dict(),
            'means': self.means,
            'stds': self.stds,
            'labels': self.labels},
            f_dir)
        pickle.dump(self.pca, open(f_dir.replace('.pth', '.pkl'), 'wb'))

    def load_dict(self, f_dir):
        """
        :param f_dir: file name
        :return: None
        """
        state_dict = torch.load(f_dir, map_location='cpu', weights_only=False)
        self.encoder.load_state_dict(state_dict=state_dict['encoder'])
        self.decoder.load_state_dict(state_dict=state_dict['decoder'])
        self.encoder.eval()
        self.decoder.eval()
        self.fc_mu.load_state_dict(state_dict=state_dict['fc_mu'])
        self.fc_var.load_state_dict(state_dict=state_dict['fc_var'])
        self.fc_mu.eval()
        self.fc_var.eval()
        self.means = state_dict['means']
        self.stds = state_dict['stds']
        self.labels = state_dict['labels']
        self.pca = pickle.load(open(f_dir.replace('.pth', '.pkl'), 'rb'))

    def encode(self, x):
        z = self.encoder(x)
        mu = self.fc_mu(z)
        return mu
        # logvar = self.fc_var(z)
        # std = torch.exp(0.5 * logvar)
        # pc = torch.randn_like(std) * std + mu
        # return pc

    @torch.inference_mode()
    def transform(self, x, y):
        """
        :param x: numpy ndarray [nx66]
        :param y: list of int label or string label
        :return: numpy ndarray [nxd]
        """
        if isinstance(y, str):
            y = torch.nn.functional.one_hot(torch.tensor([self.labels.index(y)], dtype=torch.long), len(self.labels)).cpu()
        else:
            y = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), len(self.labels)).cpu()
        x = (torch.from_numpy(x).float() - self.means) / self.stds
        z = self.encode(torch.cat((x, y), dim=1)).numpy()
        z = self.pca.transform(z)
        return z

    @torch.inference_mode()
    def inverse_transform(self, x, y):
        """
        :param x: numpy ndarray [nxd]
        :param y: list of int label or string label
        :return: numpy ndarray [nx66]
        """
        x = self.pca.inverse_transform(x)
        if isinstance(y, str):
            y = torch.nn.functional.one_hot(torch.tensor([self.labels.index(y)], dtype=torch.long), len(self.labels)).cpu()
        else:
            y = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), len(self.labels)).cpu()
        x = torch.cat((torch.from_numpy(x).float(), y), dim=1)
        x = self.decoder(x)
        return (x * self.stds + self.means).numpy()

    def train(self, x, y):
        """
        :param x: input anchor position numpy.ndarray [mxl, d]
        :param y: input label numpy.ndarray [mxl]
        :return: None
        """
        torch.set_num_threads(16)

        # prepare data
        x = torch.from_numpy(x).float()
        self.means = torch.mean(x)
        self.stds = torch.std(x)
        x = (x - self.means) / self.stds
        y = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), len(self.labels))
        x = x.to('cuda')
        y = y.to('cuda')
        self.w_rcs = self.w_rcs.to('cuda')

        # prepare model
        self.encoder = self.encoder.to('cuda')
        self.decoder = self.decoder.to('cuda')
        self.fc_mu = self.fc_mu.to('cuda')
        self.fc_var = self.fc_var.to('cuda')
        self.encoder.train()
        self.decoder.train()
        self.fc_mu.train()
        self.fc_var.train()
        print('#enc param:', sum([p.data.nelement() for p in self.encoder.parameters()]))
        print('#dec param:', sum([p.data.nelement() for p in self.decoder.parameters()]))
        print('#mu  param:', sum([p.data.nelement() for p in self.fc_mu.parameters()]))
        print('#var param:', sum([p.data.nelement() for p in self.fc_var.parameters()]))
        optim = AdamW(lr=self.lr, weight_decay=self.wd, params=[
            {'params': self.encoder.parameters()}, {'params': self.decoder.parameters()},
            {'params': self.fc_mu.parameters()}, {'params': self.fc_var.parameters()}
        ])
        sched = ReduceLROnPlateau(optim, factor=0.5, threshold=0.01,
                                  min_lr=1e-6, patience=self.num_epoch // 8)

        # mini-batch loop
        wandb.init(
            project='CVAE Dim-Reduction',
            name=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        )
        for i in range(self.num_epoch):
            sched_mrt = []
            idx = torch.randperm(x.shape[0])
            for j in range(x.shape[0] // self.size_batch):
                x_ = x[idx[(j * self.size_batch):((j + 1) * self.size_batch)]]
                y_ = y[idx[(j * self.size_batch):((j + 1) * self.size_batch)]]
                optim.zero_grad()
                z = self.encoder(torch.cat((x_, y_), dim=1))
                mu = self.fc_mu(z)
                logvar = self.fc_var(z)
                std = torch.exp(0.5 * logvar)
                pc = torch.randn_like(std) * std + mu

                _x_ = self.decoder(torch.cat((pc, y_), dim=1))
                l_rcs = (_x_ - x_).abs().mul(self.w_rcs[None]).mean()
                l_kld = -torch.mean(1 + logvar - mu ** 2 - logvar.exp())
                loss = l_rcs + self.a_kl * l_kld

                loss.backward()
                optim.step()
                sched_mrt.append(l_rcs.item())

            sched.step(sum(sched_mrt) / len(sched_mrt))
            if i % 10 == 0:
                wandb.log(step=i * (x.shape[0] // self.size_batch), commit=True, data={
                    'loss/rcs': l_rcs.item(),
                    'loss/kld': l_kld.item(),
                    'lr': sched.get_last_lr()[0]})
        wandb.finish()

        # pca
        x = x.to('cpu')
        y = y.to('cpu')
        self.encoder = self.encoder.to('cpu')
        self.fc_mu = self.fc_mu.to('cpu')
        self.fc_var = self.fc_var.to('cpu')
        self.encoder.eval()
        self.fc_mu.eval()
        self.fc_var.eval()
        with torch.inference_mode():
            z = self.encode(torch.cat((x, y), dim=1)).detach().numpy()
        self.pca.fit(z)
        print('PCA-variance_ratio', self.pca.explained_variance_ratio_)


class ResMlp(nn.Module):
    def __init__(self, in_features, n_lyr, drop=0.0):
        """
        Create residual mlp model
        Args:
            in_features int: input length
            n_lyr int: number of layer

        Returns:
            nn.Module: residual mlp model
        """
        super().__init__()

        self.model = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_features),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(in_features, in_features)
            )
            for _ in range(n_lyr)])

    def forward(self, x):
        """
        forward pass
        Args:
            x torch.tensor[b, m]: input tensor

        Returns:
            nn.tensor [b, n]
        """
        for lyr in self.model:
            x = lyr(x) + x
        return x

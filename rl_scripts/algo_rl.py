"""
SPDX-FileCopyrightText: 2025 Humanoid Sensing and Perception, Istituto Italiano di Tecnologia
SPDX-License-Identifier: BSD-3-Clause
Code adapted from https://github.com/aravindr93/mjrl/ under Apache 2.0 license
"""
import os
import sys
import time
import torch
import pickle
import numpy as np
import torch.nn as nn
from collections import defaultdict
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from gymnasium.wrappers.utils import RunningMeanStd
sys.path.append(os.path.dirname(__file__))
from base_sampler import sample_paths_parallel


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPBaseline:
    def __init__(self, dim_obs, lr=1e-3, wd=0.0, batch_size=64, epochs=4, max_grad_norm=0.5):
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = nn.Sequential(
            layer_init(nn.Linear(dim_obs + 4, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.loss_func = torch.nn.MSELoss()

    def _features(self, paths):
        obs = np.concatenate([path['observations'] for path in paths])
        feat_mat = np.ones((obs.shape[0], obs.shape[1] + 4))
        feat_mat[:,:obs.shape[1]] = obs
        k = 0  # start from this row
        for i in range(len(paths)):
            l = len(paths[i]['rewards'])
            al = np.arange(l) / 500.0
            for j in range(4):
                feat_mat[k:k+l, -4+j] = al**(j+1)
            k += l
        return feat_mat

    def fit(self, paths, return_errors=False):
        featmat = torch.from_numpy(self._features(paths)).float()
        returns = torch.from_numpy(np.concatenate([path['returns'] for path in paths])[:, None]).float()
        num_samples = returns.shape[0]

        if return_errors:
            self.model.eval()
            with torch.no_grad():
                errors = returns - self.model(featmat)
                error_before = (errors.pow(2).sum() / returns.pow(2).sum().add(1e-8)).item()

        # fitting
        self.model.train()
        b_inds = np.arange(num_samples)
        for _ in range(self.epochs):
            np.random.shuffle(b_inds)
            for mb in range(num_samples // self.batch_size):
                mb_inds = b_inds[(mb * self.batch_size):((mb + 1) * self.batch_size)]
                self.optimizer.zero_grad()
                yhat = self.model(featmat[mb_inds])
                loss = self.loss_func(yhat, returns[mb_inds])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        if return_errors:
            self.model.eval()
            with torch.no_grad():
                errors = returns - self.model(featmat)
                error_after = (errors.pow(2).sum() / returns.pow(2).sum().add(1e-8)).item()
            return error_before, error_after

    def predict(self, path):
        self.model.eval()
        with torch.no_grad():
            featmat = torch.from_numpy(self._features([path])).float()
            prediction = self.model(featmat).flatten().numpy()
        return prediction


class MLP:
    def __init__(self, dim_obs, dim_act,
                 hidden_sizes=(64,64),
                 rpo_alpha=0,
                 min_log_std=-3,
                 init_log_std=0):
        self.dim_act = dim_act
        self.min_log_std = min_log_std
        self.rpo_alpha = rpo_alpha

        # Policy network
        self.model = nn.Sequential(
            layer_init(nn.Linear(dim_obs, hidden_sizes[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], dim_act), std=0.01)
        )
        self.log_std = nn.Parameter(torch.ones(dim_act) * init_log_std)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        # Old Policy network
        self.old_model = nn.Sequential(
            nn.Linear(dim_obs, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], dim_act),
        )
        self.old_log_std = nn.Parameter(torch.ones(dim_act) * init_log_std)
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]

    def zero_grad(self):
        self.model.zero_grad()
        self.old_model.zero_grad()

    def get_param_values(self):
        params = torch.cat([param.data.flatten() for param in self.trainable_params])
        return params.clone()

    def set_param_values(self, new_params, is_new):
        params = self.trainable_params if is_new else self.old_params
        current_idx = 0
        for idx, param in enumerate(params):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            param.data.copy_(vals)
            current_idx += self.param_sizes[idx]
        # clip std at minimum value
        params[-1].data.clamp_(self.min_log_std)

    def get_action(self, observation):
        with torch.no_grad():
            obs = torch.from_numpy(observation[None]).float()
            mean = self.model(obs)[0].numpy()
            # action = Normal(mean, self.log_std.exp().expand_as(mean)).sample()
        std = np.exp(self.log_std.data.detach().numpy())
        action = mean + std * np.random.randn(self.dim_act)
        return action, {'mean': mean, 'std': std}

    def action_prob(self, obs, act=None, is_new=True, with_rpo=False):
        model = self.model if is_new else self.old_model
        log_std = self.log_std if is_new else self.old_log_std
        mean = model(obs)
        if with_rpo and self.rpo_alpha > 0:
            mean = mean + torch.empty_like(mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
        prob = Normal(mean, log_std.exp()[None].expand_as(mean))
        if act is None:
            return prob
        else:
            logprob = prob.log_prob(act).sum(1)
            return logprob


class NPG_REINFORCE:
    def __init__(self, env_spec, policy, baseline, max_kl, damping, seed):
        self.env_spec = env_spec
        self.baseline = baseline
        self.damping = damping
        self.policy = policy
        self.max_kl = max_kl
        self.iter_count = 0
        self.seed = seed
        self.log = defaultdict(list)
        self.return_rms = RunningMeanStd()
        self.demo_size = None

    def normalize_reward(self, paths):
        for path in paths:
            self.return_rms.update(path['rewards'])
        for path in paths:
            path['rewards'] = path['rewards'] / np.sqrt(self.return_rms.var + 1e-8)

    @staticmethod
    def compute_returns(paths, gamma):
        for path in paths:
            path['returns'] = NPG_REINFORCE.discount_sum(path['rewards'], gamma)

    @staticmethod
    def compute_advantages(paths, baseline, gamma, gae_lambda):
        # compute and store returns, advantages, and baseline
        for path in paths:
            b = path['baseline'] = baseline.predict(path)
            b1 = np.append(path['baseline'], 0.0 if path['terminated'] else b[-1])
            td_deltas = path['rewards'] + gamma * b1[1:] - b1[:-1]
            path['advantages'] = NPG_REINFORCE.discount_sum(td_deltas, gamma * gae_lambda)

    @staticmethod
    def discount_sum(x, gamma, terminal=0.0):
        y = []
        run_sum = terminal
        for t in range(len(x) - 1, -1, -1):
            run_sum = x[t] + gamma * run_sum
            y.append(run_sum)
        return np.array(y[::-1])

    def get_grad(self, x, **kwargs):
        grads = torch.autograd.grad(x, self.policy.trainable_params, **kwargs)
        return torch.cat([grad.flatten() for grad in grads])

    def cg_solve(self, pg_grad, obs, niters=10, residual_tol=1e-10):
        x = torch.zeros_like(pg_grad)
        r, p = pg_grad.clone(), pg_grad.clone()
        rdotr = r.dot(r)
        for _ in range(niters):
            z = self.hvp(p, obs).detach()
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def hvp(self, x, obs):
        x.detach_()
        kl_grad = self.get_grad(self.kl_old_new(obs), create_graph=True)
        kl_grad_grad = self.get_grad((kl_grad * x).sum()).detach()
        self.policy.zero_grad()
        return kl_grad_grad + x * self.damping

    def actor_loss(self, obs, act, adv, with_rpo=False):
        logprob_new = self.policy.action_prob(obs, act, is_new=True, with_rpo=with_rpo)
        logprob_old = self.policy.action_prob(obs, act, is_new=False, with_rpo=with_rpo)
        loss = (logprob_new - logprob_old).exp() * adv
        return loss.mean()

    def kl_old_new(self, obs):
        with torch.no_grad():
            prob_old = self.policy.action_prob(obs, is_new=False)
        prob_new = self.policy.action_prob(obs, is_new=True)
        kld = kl_divergence(prob_old, prob_new).mean()
        return kld

    def train_step(self, num_ep, gamma=0.995, gae_lambda=0.98, num_cpu=1):
        tick = time.time()
        paths = sample_paths_parallel(num_ep=num_ep, policy=self.policy, seed=self.seed,
                                      num_cpu=num_cpu, env_spec=self.env_spec)
        self.log['time_sampling'].append(time.time() - tick)
        self.seed = self.seed + num_ep
        self.log_rollout_statistics(paths)
        self.iter_count += 1

        tick = time.time()
        # normalize reward
        self.normalize_reward(paths)
        # compute returns
        self.compute_returns(paths, gamma)
        # compute advantages
        self.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        self.train_from_paths(paths)
        self.log['time_pi'].append(time.time() - tick)

        # fit baseline
        tick = time.time()
        error_before, error_after = self.baseline.fit(paths, return_errors=True)
        self.log['time_VF'].append(time.time() - tick)
        self.log['VF_error_before'].append(error_before)
        self.log['VF_error_after'].append(error_after)

    def log_rollout_statistics(self, paths):
        path_returns = [sum(p['rewards']) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.log['stoc_pol_mean'].append(mean_return)
        self.log['stoc_pol_std'].append(std_return)
        self.log['stoc_pol_max'].append(max_return)
        self.log['stoc_pol_min'].append(min_return)


class TRPO(NPG_REINFORCE):
    def __init__(self, env_spec, policy, baseline, max_kl=1e-2, damping=1e-4, seed=0):
        super().__init__(env_spec, policy, baseline, max_kl, damping, seed)

    def train_from_paths(self, paths):
        # Concatenate env traj
        obs = torch.from_numpy(np.concatenate([path['observations'] for path in paths])).float()
        act = torch.from_numpy(np.concatenate([path['actions'] for path in paths])).float()
        adv = torch.from_numpy(np.concatenate([path['advantages'] for path in paths])).float()
        adv = adv.sub(adv.mean()).div(adv.std().add(1e-6))

        # Optimization algorithm
        self.policy.zero_grad()
        with torch.no_grad():
            surr_before = self.actor_loss(obs, act, adv).item()
        vpg_grad = self.get_grad(self.actor_loss(obs, act, adv, with_rpo=True)).detach()
        self.policy.zero_grad()
        npg_grad = self.cg_solve(vpg_grad, obs)

        # Policy update
        full_step = npg_grad * (2 * self.max_kl / (npg_grad * self.hvp(npg_grad, obs)).sum()).pow(0.5)
        expected_improve = (vpg_grad * full_step).sum()
        curr_params = self.policy.get_param_values()
        alpha = 1
        with torch.no_grad():
            for i in range(16):
                new_params = curr_params + alpha * full_step
                self.policy.set_param_values(new_params, is_new=True)
                surr_after = self.actor_loss(obs, act, adv).item()
                actual_improve = surr_after - surr_before
                kld = self.kl_old_new(obs).item()
                if ((actual_improve > 0) and
                    (kld < self.max_kl) and
                    ((actual_improve / expected_improve / alpha) > 0.1)):
                    break
                alpha = 0.8 * alpha
            self.policy.set_param_values(new_params, is_new=False)

        # Log information
        self.log['alpha'].append(alpha)
        self.log['kl_dist'].append(kld)
        self.log['surr_improvement'].append(surr_after - surr_before)


class DAPG(NPG_REINFORCE):
    def __init__(self, env_spec, policy, baseline, f_demo, f_norm,
                 lam0=1.0, lam1=0.95, max_kl=1e-2, damping=1e-4, seed=0):
        super().__init__(env_spec, policy, baseline, max_kl, damping, seed)
        self.lam0 = lam0
        self.lam1 = lam1
        # norm stat
        demo = pickle.load(open(f_demo, 'rb'))
        norm_stat = np.load(f_norm)
        obs_mean, obs_std = norm_stat['obs_mean'][None], norm_stat['obs_std'][None]
        act_min, act_max = norm_stat['act_min'][None], norm_stat['act_max'][None]
        # norm demo
        demo_obs = np.concatenate([path['observations'] for path in demo])
        demo_obs = (demo_obs - obs_mean) / obs_std
        demo_act = np.concatenate([path['actions'] for path in demo])
        demo_act = (demo_act - act_min) / (act_max - act_min) * 2 - 1
        self.demo_obs = torch.from_numpy(demo_obs).float()
        self.demo_act = torch.from_numpy(demo_act).float()
        self.demo_adv = torch.full((self.demo_obs.shape[0],), self.lam0)

    def train_from_paths(self, paths):
        # Concatenate env traj
        obs = torch.from_numpy(np.concatenate([path['observations'] for path in paths])).float()
        act = torch.from_numpy(np.concatenate([path['actions'] for path in paths])).float()
        adv = torch.from_numpy(np.concatenate([path['advantages'] for path in paths])).float()
        adv = adv.sub(adv.mean()).div(adv.std().add(1e-6))

        # concatenate all
        all_obs = torch.concat((obs, self.demo_obs), dim=0)
        all_act = torch.concat((act, self.demo_act), dim=0)
        all_adv = torch.concat((adv, self.demo_adv * self.lam1 ** self.iter_count), dim=0)
        all_adv = all_adv.sub(all_adv.mean()).div(all_adv.std().add(1e-6))

        # Optimization algorithm
        self.policy.zero_grad()
        with torch.no_grad():
            surr_before = self.actor_loss(all_obs, all_act, all_adv).item()
        vpg_grad = self.get_grad(self.actor_loss(all_obs, all_act, all_adv, with_rpo=True)).detach()
        self.policy.zero_grad()
        npg_grad = self.cg_solve(vpg_grad, all_obs)

        # Policy update
        full_step = npg_grad * (2 * self.max_kl / (npg_grad * self.hvp(npg_grad, all_obs)).sum()).pow(0.5)
        expected_improve = (vpg_grad * full_step).sum()
        curr_params = self.policy.get_param_values()
        alpha = 1
        with torch.no_grad():
            for i in range(16):
                new_params = curr_params + alpha * full_step
                self.policy.set_param_values(new_params, is_new=True)
                surr_after = self.actor_loss(all_obs, all_act, all_adv).item()
                actual_improve = surr_after - surr_before
                kld = self.kl_old_new(all_obs).item()
                if ((actual_improve > 0) and
                    (kld < self.max_kl) and
                    ((actual_improve / expected_improve / alpha) > 0.1)):
                    break
                alpha = 0.8 * alpha
            self.policy.set_param_values(new_params, is_new=False)

        # Log information
        self.log['alpha'].append(alpha)
        self.log['kl_dist'].append(kld)
        self.log['surr_improvement'].append(surr_after - surr_before)

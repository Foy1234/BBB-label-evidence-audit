from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

def _sigmoid(x):
    return torch.sigmoid(x)

class LatentHEM:

    def __init__(self, n_molecules: int, n_em_iters: int=50, m_epochs: int=100, m_lr: float=0.01, l2_prior: float=1.0, freeze_delta: bool=False):
        self.n_molecules = int(n_molecules)
        self.n_em_iters = int(n_em_iters)
        self.m_epochs = int(m_epochs)
        self.m_lr = m_lr
        self.l2_prior = l2_prior
        self.freeze_delta = bool(freeze_delta)
        self._e_substeps = 30
        self._e_lr = 0.5
        self._init_params()

    def _init_params(self):
        self.mu = np.zeros(self.n_molecules)
        self.sigma = np.full(self.n_molecules, 1.0 / np.sqrt(self.l2_prior))
        self.delta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        self.a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        self.b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        self.log_sigma = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    def _to_arrays(self, idx, labels):
        self.b3db_idx = np.asarray(idx, dtype=int)
        self.b3db_labels = np.asarray(labels, dtype=float)
        self.has_bbbp = False
        self.bbbp_idx = np.zeros(0, dtype=int)
        self.bbbp_labels = np.zeros(0, dtype=float)
        self.has_logbb = False
        self.logbb_idx = np.zeros(0, dtype=int)
        self.logbb_vals = np.zeros(0, dtype=float)

    def set_observations(self, b3db_idx, b3db_labels, bbbp_idx=None, bbbp_labels=None, logbb_idx=None, logbb_vals=None):
        self._to_arrays(b3db_idx, b3db_labels)
        if bbbp_idx is not None and len(bbbp_idx):
            self.has_bbbp = True
            self.bbbp_idx = np.asarray(bbbp_idx, dtype=int)
            self.bbbp_labels = np.asarray(bbbp_labels, dtype=float)
        if logbb_idx is not None and len(logbb_idx):
            self.has_logbb = True
            self.logbb_idx = np.asarray(logbb_idx, dtype=int)
            self.logbb_vals = np.asarray(logbb_vals, dtype=float)
            within = []
            for i in np.unique(self.logbb_idx):
                vals = self.logbb_vals[self.logbb_idx == i]
                if len(vals) >= 2:
                    within.append(float(np.var(vals)))
            if within:
                sigma_init = float(np.sqrt(np.mean(within)))
            else:
                sigma_init = float(np.std(self.logbb_vals))
            sigma_init = float(max(sigma_init, 0.05))
            self.log_sigma = torch.tensor(float(np.log(sigma_init)), dtype=torch.float64, requires_grad=True)

    def fit(self, b3db_idx, b3db_labels, bbbp_idx=None, bbbp_labels=None, logbb_idx=None, logbb_vals=None, verbose: bool=False):
        self.set_observations(b3db_idx, b3db_labels, bbbp_idx, bbbp_labels, logbb_idx, logbb_vals)
        for it in range(self.n_em_iters):
            self._e_step()
            self._m_step()
            if verbose and it % 10 == 0:
                print(f'  EM it {it}: elbo={self._elbo():.3f} delta={float(self.delta):.3f}')
        return self

    def _e_step(self):
        d = float(self.delta.detach())
        a = float(self.a.detach())
        b = float(self.b.detach())
        sigma = float(torch.exp(self.log_sigma).detach())
        n = self.n_molecules
        sum_v = np.zeros(n)
        count = np.zeros(n, dtype=int)
        if self.has_logbb:
            for i, v in zip(self.logbb_idx, self.logbb_vals):
                sum_v[i] += v
                count[i] += 1
        b3db_membership = np.zeros(n, dtype=bool)
        b3db_membership[self.b3db_idx] = True
        b3db_label_map = np.full(n, np.nan)
        for i, y in zip(self.b3db_idx, self.b3db_labels):
            b3db_label_map[i] = y
        bbbp_membership = np.zeros(n, dtype=bool)
        bbbp_label_map = np.full(n, np.nan)
        if self.has_bbbp:
            bbbp_membership[self.bbbp_idx] = True
            for i, y in zip(self.bbbp_idx, self.bbbp_labels):
                bbbp_label_map[i] = y
        lr = self._e_lr
        for _ in range(self._e_substeps):
            mu = self.mu
            grad = self.l2_prior * mu
            hess = np.full(n, self.l2_prior)
            p = 1.0 / (1.0 + np.exp(-mu))
            grad += np.where(b3db_membership, p - b3db_label_map, 0.0)
            hess += np.where(b3db_membership, p * (1 - p), 0.0)
            if self.has_bbbp:
                z = mu + d
                pz = 1.0 / (1.0 + np.exp(-z))
                grad += np.where(bbbp_membership, pz - bbbp_label_map, 0.0)
                hess += np.where(bbbp_membership, pz * (1 - pz), 0.0)
            if self.has_logbb:
                has = count > 0
                mu_sel = mu[has]
                n_obs = count[has]
                sumv_sel = sum_v[has]
                grad[has] += a * (n_obs * (a * mu_sel + b) - sumv_sel) / sigma ** 2
                hess[has] += n_obs * a ** 2 / sigma ** 2
            hess = np.maximum(hess, 0.1)
            self.mu = np.clip(mu - lr * grad / hess, -5.0, 5.0)
        mu = self.mu
        hess = np.full(n, self.l2_prior)
        p = 1.0 / (1.0 + np.exp(-mu))
        hess += np.where(b3db_membership, p * (1 - p), 0.0)
        if self.has_bbbp:
            z = mu + d
            pz = 1.0 / (1.0 + np.exp(-z))
            hess += np.where(bbbp_membership, pz * (1 - pz), 0.0)
        if self.has_logbb:
            has = count > 0
            hess[has] += count[has] * a ** 2 / sigma ** 2
        self.sigma = 1.0 / np.sqrt(np.maximum(hess, 1e-08))

    def _m_step(self):
        if self.freeze_delta:
            with torch.no_grad():
                self.delta.fill_(0.0)
            self.a.requires_grad_(True)
            self.b.requires_grad_(True)
            mu = torch.tensor(self.mu, dtype=torch.float64)
            optim = torch.optim.Adam([self.a, self.b], lr=self.m_lr)
        else:
            self.delta.requires_grad_(True)
            self.a.requires_grad_(True)
            self.b.requires_grad_(True)
            mu = torch.tensor(self.mu, dtype=torch.float64)
            optim = torch.optim.Adam([self.delta, self.a, self.b], lr=self.m_lr)
        for _ in range(self.m_epochs):
            optim.zero_grad()
            loss = self._m_loss(mu)
            loss.backward()
            optim.step()
        if self.has_logbb:
            within = []
            idx = self.logbb_idx
            for i in np.unique(idx):
                vals = self.logbb_vals[idx == i]
                if len(vals) >= 2:
                    within.append(float(np.var(vals)))
            if within:
                var = float(np.mean(within))
            else:
                var = float(np.var(self.logbb_vals))
            var = float(np.clip(var, 0.0025, 25.0))
            with torch.no_grad():
                self.log_sigma.fill_(float(0.5 * np.log(var)))
        self._freeze_global()

    def _m_loss(self, mu):
        loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        b3db_lab = torch.tensor(self.b3db_labels, dtype=torch.float64)
        loss = loss + torch.nn.functional.binary_cross_entropy(_sigmoid(mu[self.b3db_idx]), b3db_lab)
        if self.has_bbbp:
            bb_lab = torch.tensor(self.bbbp_labels, dtype=torch.float64)
            loss = loss + torch.nn.functional.binary_cross_entropy(_sigmoid(mu[self.bbbp_idx] + self.delta), bb_lab)
        if self.has_logbb:
            sigma = float(torch.exp(self.log_sigma).detach())
            mu_sel = mu[self.logbb_idx]
            vals = torch.tensor(self.logbb_vals, dtype=torch.float64)
            loss = loss + torch.sum((vals - (self.a * mu_sel + self.b)) ** 2) / (2 * sigma ** 2)
        return loss

    def _elbo(self) -> float:
        mu = torch.tensor(self.mu, dtype=torch.float64)
        loss = self._m_loss(mu)
        loss = loss + 0.5 * self.l2_prior * torch.sum(mu ** 2)
        return float(loss)

    def _latent_numpy(self) -> np.ndarray:
        return self.mu

    def predict(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=int)
        return 1.0 / (1.0 + np.exp(-self.mu[idx]))

    def predict_with_uncertainty(self, idx) -> dict:
        idx = np.asarray(idx, dtype=int)
        p = self.predict(idx)
        return {'mean': p, 'latent': self.mu[idx], 'std_latent': self.sigma[idx]}

    def source_contributions(self, mol_ref_matrix, mol_labels) -> dict:
        mol_ref_matrix = np.asarray(mol_ref_matrix, dtype=float)
        if mol_ref_matrix.shape[0] != self.n_molecules:
            raise ValueError(f'mol_ref_matrix rows ({mol_ref_matrix.shape[0]}) != n_molecules ({self.n_molecules})')
        labels = np.asarray(mol_labels, dtype=float)
        if len(labels) != self.n_molecules:
            raise ValueError('mol_labels length != n_molecules')
        valid = ~np.isnan(labels)
        weight = 1.0 / np.maximum(self.sigma, 1e-06)
        score = np.full(self.n_molecules, np.nan)
        sign = np.sign(self.mu)
        score[valid] = (2 * labels[valid] - 1) * sign[valid] * weight[valid]
        n_sources = mol_ref_matrix.shape[1]
        w = np.zeros(n_sources)
        cnt = np.zeros(n_sources)
        for r in range(n_sources):
            mask = (mol_ref_matrix[:, r] > 0) & valid
            if mask.any():
                w[r] = float(np.nansum(score[mask]) / np.nansum(weight[mask]))
                cnt[r] = int(mask.sum())
        return {'reliability': w, 'count': cnt}

    def save(self, path: str):
        obj = {'n_molecules': self.n_molecules, 'mu': self.mu.tolist(), 'sigma': self.sigma.tolist(), 'delta': float(np.asarray(self.delta)), 'a': float(np.asarray(self.a)), 'b': float(np.asarray(self.b)), 'log_sigma': float(np.asarray(self.log_sigma)), 'l2_prior': self.l2_prior}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'LatentHEM':
        with open(path, encoding='utf-8') as f:
            obj = json.load(f)
        m = cls(n_molecules=obj['n_molecules'], l2_prior=obj['l2_prior'])
        m.mu = np.array(obj['mu'], dtype=float)
        m.sigma = np.array(obj['sigma'], dtype=float)
        m.delta = torch.tensor(obj['delta'], dtype=torch.float64)
        m.a = torch.tensor(obj['a'], dtype=torch.float64)
        m.b = torch.tensor(obj['b'], dtype=torch.float64)
        m.log_sigma = torch.tensor(obj['log_sigma'], dtype=torch.float64)
        return m

    def _freeze_global(self):
        self.delta.requires_grad_(False)
        self.a.requires_grad_(False)
        self.b.requires_grad_(False)
        self.log_sigma.requires_grad_(False)

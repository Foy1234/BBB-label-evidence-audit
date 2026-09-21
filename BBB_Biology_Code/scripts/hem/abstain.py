from __future__ import annotations
import numpy as np

def abstain_mask(std_latent: np.ndarray, tau: float) -> np.ndarray:
    std = np.asarray(std_latent, dtype=float)
    return std > tau

def coverage_risk(std_latent: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, tau_grid=None) -> dict:
    std = np.asarray(std_latent, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    yp = (np.asarray(y_pred, dtype=float) >= 0.5).astype(float)
    if tau_grid is None:
        tau_grid = np.linspace(0.0, std.max() + 1e-06, 21)
    coverage = []
    risk = []
    for tau in tau_grid:
        abst = std > tau
        covered = ~abst
        n_cov = int(covered.sum())
        if n_cov == 0:
            coverage.append(0.0)
            risk.append(1.0)
        else:
            err = float(np.mean(yt[covered] != yp[covered]))
            coverage.append(float(n_cov / len(std)))
            risk.append(err)
    return {'tau': np.asarray(tau_grid), 'coverage': np.asarray(coverage), 'risk': np.asarray(risk)}

def abstain_summary(std_latent: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> dict:
    abst = abstain_mask(std_latent, tau)
    covered = ~abst
    n = len(std_latent)
    n_abstain = int(abst.sum())
    yt = np.asarray(y_true, dtype=float)
    yp = (np.asarray(y_pred, dtype=float) >= 0.5).astype(float)
    if covered.sum() == 0:
        covered_acc = float('nan')
        covered_risk = float('nan')
    else:
        covered_acc = float(np.mean(yt[covered] == yp[covered]))
        covered_risk = 1.0 - covered_acc
    return {'n_total': n, 'n_abstain': n_abstain, 'abstain_ratio': n_abstain / n, 'n_covered': int(covered.sum()), 'covered_accuracy': covered_acc, 'covered_risk': covered_risk}

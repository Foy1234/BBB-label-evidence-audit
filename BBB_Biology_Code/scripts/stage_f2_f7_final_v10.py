from __future__ import annotations
from project_paths import project_root, project_path
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression
ROOT = project_root()
CODE = ROOT / '01_代码_各阶段'
sys.path.insert(0, str(CODE))
from hem import LatentHEM
from hem.data import build_tensors, parse_logbb_list
RDLogger.DisableLog('rdApp.*')
SEED = 20260818
RADIUS, N_BITS = (2, 2048)
XGB_PARAMS = dict(max_depth=5, learning_rate=0.03, n_estimators=300, subsample=0.8, colsample_bytree=0.8, random_state=42)
RESULT_VERSION = 'v10'
OUT02 = ROOT / '02_结果'
OUT05 = ROOT / '05_复现工作区'
FIGDIR = ROOT / '03_指南与文档' / '论文素材' / 'figures_v03'
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Microsoft YaHei', 'DejaVu Sans'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7, 'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.8, 'legend.frameon': False})
COLORS = {'hem': '#2F6B8A', 'majority': '#777777', 'source_weighted': '#A16B45', 'dawid_skene': '#5D8A66', 'map_latent': '#7B6CA8', 'snorkel': '#C05A67', 'logbb': '#D39A2C'}

def outdirs(fid: str) -> tuple[Path, Path]:
    a, b = (OUT02 / f'results_{fid}_{RESULT_VERSION}', OUT05 / f'results_{fid}_{RESULT_VERSION}')
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    return (a, b)

def write_both(fid: str, name: str, writer) -> None:
    for d in outdirs(fid):
        writer(d / name)

def write_json(fid: str, name: str, obj) -> None:
    write_both(fid, name, lambda p: p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8'))

def write_csv(fid: str, name: str, df: pd.DataFrame) -> None:
    write_both(fid, name, lambda p: df.to_csv(p, index=False, encoding='utf-8-sig'))

def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)

def sigmoid(x):
    x = np.clip(np.asarray(x, float), -30, 30)
    return 1 / (1 + np.exp(-x))

def metric_row(y, p):
    y, p = (np.asarray(y, float), np.asarray(p, float))
    ok = np.isfinite(y) & np.isfinite(p) & np.isin(y, [0, 1])
    y, p = (y[ok].astype(int), p[ok])
    if len(y) < 5 or len(np.unique(y)) < 2:
        raise ValueError(f'insufficient evaluable observations: n={len(y)} classes={np.unique(y)}')
    pc = np.clip(p, 1e-07, 1 - 1e-07)
    bins = np.minimum((pc * 10).astype(int), 9)
    ece = sum(((bins == k).sum() * abs(pc[bins == k].mean() - y[bins == k].mean()) for k in range(10) if (bins == k).any())) / len(y)
    hc = (pc <= 0.1) | (pc >= 0.9)
    return {'n': len(y), 'auc': roc_auc_score(y, p), 'auprc': average_precision_score(y, p), 'mcc': matthews_corrcoef(y, p >= 0.5), 'brier': brier_score_loss(y, pc), 'ece': ece, 'high_conf_n': int(hc.sum()), 'high_conf_error_rate': float(((p[hc] >= 0.5) != y[hc]).mean()) if hc.any() else np.nan}

def eval_latent(L, p):
    y = (np.asarray(L) > 0).astype(int)
    m = metric_row(y, p)
    m['rho'] = stats.spearmanr(np.asarray(L), np.asarray(p)).statistic
    return m

def paired_auc_bootstrap(y, pa, pb, seed=SEED, n_boot=2000):
    y, pa, pb = map(np.asarray, (y, pa, pb))
    ok = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb) & np.isin(y, [0, 1])
    y, pa, pb = (y[ok].astype(int), pa[ok], pb[ok])
    point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ii = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {'n': len(y), 'delta_auc': point, 'ci95': [lo, hi], 'n_boot_valid': len(vals)}

def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        out[idx] = running
    return out

def majority(X):
    X = np.asarray(X, float)
    count = np.sum(np.isfinite(X), axis=1)
    return np.divide(np.nansum(X, axis=1), count, out=np.full(len(X), 0.5), where=count > 0)

def source_weighted(X):
    X = np.asarray(X, float)
    w = np.linspace(1.0, 0.7, X.shape[1])
    ok = np.isfinite(X)
    den = (ok * w).sum(1)
    return np.divide(np.nansum(X * w, axis=1), den, out=np.full(len(X), 0.5), where=den > 0)

def dawid_skene(X, max_iter=100, tol=1e-07, beta=1.0):
    X = np.asarray(X, float)
    n, s = X.shape
    q = np.clip(majority(X), 0.05, 0.95)
    sens = np.full(s, 0.75)
    spec = np.full(s, 0.75)
    for _ in range(max_iter):
        old = q.copy()
        lp1 = np.zeros(n)
        lp0 = np.zeros(n)
        for j in range(s):
            ok = np.isfinite(X[:, j])
            z = X[ok, j]
            lp1[ok] += np.where(z == 1, np.log(sens[j]), np.log(1 - sens[j]))
            lp0[ok] += np.where(z == 1, np.log(1 - spec[j]), np.log(spec[j]))
        q = sigmoid(lp1 - lp0)
        for j in range(s):
            ok = np.isfinite(X[:, j])
            z = X[ok, j]
            qj = q[ok]
            sens[j] = np.clip(((z * qj).sum() + beta) / (qj.sum() + 2 * beta), 0.0001, 1 - 0.0001)
            spec[j] = np.clip((((1 - z) * (1 - qj)).sum() + beta) / ((1 - qj).sum() + 2 * beta), 0.0001, 1 - 0.0001)
        if np.max(abs(q - old)) < tol:
            break
    return q

def map_latent(X):
    return dawid_skene(X, beta=2.0)

def snorkel_score(X, seed):
    from snorkel.labeling.model import LabelModel
    L = np.where(np.isfinite(X), X, -1).astype(int)
    model = LabelModel(cardinality=2, verbose=False)
    model.fit(L_train=L, n_epochs=300, lr=0.01, seed=int(seed), log_freq=100)
    p = model.predict_proba(L)[:, 1]
    if not np.isfinite(p).all():
        raise ValueError('non-finite Snorkel probabilities')
    return p

def fit_hem_sim(Y, logbb=None, seed=SEED):
    Y = np.asarray(Y, float)
    if Y.shape[0] < 2:
        raise AssertionError('extended HEM requires at least two independent sources')
    X = Y.T
    mu = np.clip(np.log(np.clip(majority(X), 0.05, 0.95) / np.clip(1 - majority(X), 0.05, 0.95)), -3, 3)
    slopes = np.ones(Y.shape[0])
    biases = np.zeros(Y.shape[0])
    a, b, sig = (1.0, 0.0, 1.0)
    for _ in range(15):
        for j in range(Y.shape[0]):
            ok = np.isfinite(Y[j])
            z = Y[j, ok].astype(int)
            if len(np.unique(z)) < 2:
                raise ValueError(f'source {j} contains one class')
            lr = LogisticRegression(C=10.0, solver='lbfgs', random_state=int(seed), max_iter=200).fit(mu[ok, None], z)
            slopes[j] = max(abs(float(lr.coef_[0, 0])), 0.05)
            biases[j] = float(lr.intercept_[0])
        if logbb is not None:
            ok = np.isfinite(logbb)
            if ok.sum() > 2:
                a, b = np.polyfit(mu[ok], np.asarray(logbb)[ok], 1)
                a = max(abs(float(a)), 0.05)
                sig = max(float(np.std(np.asarray(logbb)[ok] - (a * mu[ok] + b))), 0.1)
        grad = 0.1 * mu
        hess = np.full(len(mu), 0.1)
        for j in range(Y.shape[0]):
            ok = np.isfinite(Y[j])
            p = sigmoid(slopes[j] * mu[ok] + biases[j])
            grad[ok] += slopes[j] * (p - Y[j, ok])
            hess[ok] += slopes[j] ** 2 * p * (1 - p)
        if logbb is not None:
            ok = np.isfinite(logbb)
            res = a * mu[ok] + b - np.asarray(logbb)[ok]
            grad[ok] += a * res / sig ** 2
            hess[ok] += a * a / sig ** 2
        mu = np.clip(mu - 0.7 * grad / np.maximum(hess, 0.1), -5, 5)
    return sigmoid(mu)

def sim_base(rng, n=700, shift=0.0, imbalance=0.0, flips=(0.0, 0.0, 0.0), missing=(0.0, 0.0, 0.0), copy=0.0, conditional=False, shared=0.0, mnar=False):
    if np.isscalar(flips):
        flips = (float(flips),) * 3
    if np.isscalar(missing):
        missing = (float(missing),) * 3
    L = rng.normal(imbalance, 1, n)
    shared_noise = rng.normal(0, 1, n)
    Y = np.empty((3, n), float)
    for j in range(3):
        eta = L + (shared * shared_noise if j else 0) + (shift if j else 0)
        Y[j] = rng.binomial(1, sigmoid(eta))
    cp = copy * (sigmoid(L) if conditional else np.ones(n))
    take = rng.random(n) < cp
    Y[1, take] = Y[0, take]
    for j in range(3):
        f = rng.random(n) < flips[j]
        Y[j, f] = 1 - Y[j, f]
        missp = missing[j] * (sigmoid(L) if mnar and j == 1 else 1)
        miss = rng.random(n) < missp
        Y[j, miss] = np.nan
    logbb = L + rng.normal(0, 0.35, n)
    return (L, Y, logbb)

def run_methods(L, Y, logbb, scenario, rep, failures, include_mixed=True):
    stable = int(hashlib.sha256(scenario.encode('utf-8')).hexdigest()[:8], 16)
    rows = []
    X = Y.T
    seed = SEED + rep + stable % 100000
    funcs = {'majority': majority, 'source_weighted': source_weighted, 'dawid_skene': dawid_skene, 'map_latent': map_latent, 'snorkel': lambda z: snorkel_score(z, seed), 'hem': lambda z: fit_hem_sim(Y, None, seed)}
    for name, fn in funcs.items():
        try:
            rows.append({'scenario': scenario, 'rep': rep, 'lane': 'binary', 'method': name, 'information_set': 'Y0+Y1+Y2', **eval_latent(L, fn(X))})
        except Exception as e:
            failures.append({'scenario': scenario, 'rep': rep, 'lane': 'binary', 'method': name, 'error_type': type(e).__name__, 'error': str(e)})
    if include_mixed:
        for name, fn, info in [('hem', lambda: fit_hem_sim(Y, logbb, seed), 'Y0+Y1+Y2+logBB'), ('logbb', lambda: sigmoid(logbb), 'logBB')]:
            try:
                rows.append({'scenario': scenario, 'rep': rep, 'lane': 'mixed', 'method': name, 'information_set': info, **eval_latent(L, fn())})
            except Exception as e:
                failures.append({'scenario': scenario, 'rep': rep, 'lane': 'mixed', 'method': name, 'error_type': type(e).__name__, 'error': str(e)})
    return rows

def summarize_reps(df, keys):
    rows = []
    for g, x in df.groupby(keys, dropna=False):
        g = (g,) if not isinstance(g, tuple) else g
        row = dict(zip(keys, g))
        n = x.rep.nunique()
        row.update({'n_reps': n})
        for metric in ['auc', 'rho', 'brier']:
            v = x[metric].dropna()
            row[f'mean_{metric}'] = v.mean()
            row[f'sd_{metric}'] = v.std(ddof=1)
            row[f'ci95_{metric}_lo'] = v.mean() - stats.t.ppf(0.975, max(len(v) - 1, 1)) * v.sem() if len(v) > 1 else np.nan
            row[f'ci95_{metric}_hi'] = v.mean() + stats.t.ppf(0.975, max(len(v) - 1, 1)) * v.sem() if len(v) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def paired_rep_tests(df):
    rows = []
    for (scenario, lane), x in df.groupby(['scenario', 'lane']):
        pivot = x.pivot_table(index='rep', columns='method', values='auc', aggfunc='first')
        if 'hem' not in pivot:
            continue
        for method in pivot.columns:
            if method == 'hem':
                continue
            z = pivot[['hem', method]].dropna()
            d = z.hem - z[method]
            if len(d) < 3:
                continue
            lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
            rows.append({'scenario': scenario, 'lane': lane, 'contrast': f'hem - {method}', 'n_pairs': len(d), 'delta_auc': d.mean(), 'ci95_lo': lo, 'ci95_hi': hi, 'p_raw': stats.ttest_rel(z.hem, z[method]).pvalue})
    out = pd.DataFrame(rows)
    if len(out):
        out['p_holm'] = holm(out.p_raw.values)
    return out

def f2(n_rep):
    specs = {'base': {}, 'weak': {'shared': 0.4}, 'high_conflict': {'shift': 1.0, 'flips': (0.05, 0.2, 0.1)}, 'missing': {'missing': 0.35}, 'noise': {'flips': (0.15, 0.15, 0.15)}, 'flip10': {'flips': (0.1, 0.1, 0.1)}, 'conditional_copy': {'copy': 0.7, 'conditional': True}, 'n300': {'n': 300}, 'imbalanced': {'imbalance': 1.2}}
    rows = []
    failures = []
    for si, (name, kw) in enumerate(specs.items()):
        for rep in range(n_rep):
            L, Y, lb = sim_base(np.random.default_rng(SEED + si * 1000 + rep), **kw)
            rows += run_methods(L, Y, lb, name, rep, failures)
    df = pd.DataFrame(rows)
    summary = summarize_reps(df, ['scenario', 'lane', 'method', 'information_set'])
    tests = paired_rep_tests(df)
    expected = len(specs) * n_rep
    counts = df.groupby(['lane', 'method']).size().rename('count').reset_index().to_dict('records')
    consistency = {'scenarios': len(specs), 'configured_reps': n_rep, 'expected_method_scenario_reps': expected, 'per_method_counts': counts, 'n_failures': len(failures), 'no_silent_fallback': True, 'binary_information_sets': sorted(df.loc[df.lane.eq('binary'), 'information_set'].unique())}
    write_csv('f2', 'f2_per_rep.csv', df)
    write_csv('f2', 'f2_summary.csv', summary)
    write_csv('f2', 'f2_paired_tests.csv', tests)
    write_csv('f2', 'f2_failures.csv', pd.DataFrame(failures))
    write_json('f2', 'f2_report.json', {'status': 'complete' if not failures else 'complete_with_recorded_failures', 'design': 'HEM is extended so Y2 is an independent channel with its own reliability slope/bias. Binary lane gives every method exactly Y0+Y1+Y2; mixed lane reports HEM(Y0+Y1+Y2+logBB) and logBB-only separately', 'consistency': consistency, 'paired_tests': tests.to_dict('records'), 'failures': failures})
    return (df, summary)

def f3(n_rep):
    specs = {'complete_copy': {'copy': 1.0}, 'partial_copy': {'copy': 0.6}, 'shared_latent_source': {'shared': 0.8}, 'conditional_correlation': {'copy': 0.9, 'conditional': True}, 'mnar_missing': {'missing': (0.1, 0.65, 0.25), 'mnar': True}, 'system_bias_plus_flip': {'shift': 1.2, 'flips': (0.05, 0.25, 0.15)}, 'small_n_high_conflict': {'n': 180, 'shift': 1.4, 'flips': (0.2, 0.35, 0.25)}, 'copy_missing_imbalance': {'copy': 0.75, 'missing': (0.25, 0.55, 0.35), 'mnar': True, 'imbalance': 1.3}}
    rows = []
    failures = []
    for si, (name, kw) in enumerate(specs.items()):
        for rep in range(n_rep):
            L, Y, lb = sim_base(np.random.default_rng(SEED + 10000 + si * 1000 + rep), **kw)
            rows += run_methods(L, Y, lb, name, rep, failures)
    for ci, corr in enumerate([0, 0.25, 0.5, 0.75, 1.0]):
        for fi, flip in enumerate([0, 0.05, 0.1, 0.15, 0.2]):
            name = f'grid_corr={corr:.2f}_flip={flip:.2f}'
            for rep in range(n_rep):
                L, Y, lb = sim_base(np.random.default_rng(SEED + 30000 + ci * 5000 + fi * 500 + rep), copy=corr, flips=(0, flip, flip / 2))
                rows += run_methods(L, Y, lb, name, rep, failures)
    df = pd.DataFrame(rows)
    summary = summarize_reps(df, ['scenario', 'lane', 'method', 'information_set'])
    faildf = pd.DataFrame(failures)
    failure_counts = {(r['scenario'], r['lane'], r['method']): r['n_failed'] for r in (faildf.groupby(['scenario', 'lane', 'method']).size().rename('n_failed').reset_index().to_dict('records') if len(faildf) else [])}
    summary['n_failed'] = [failure_counts.get((r.scenario, r.lane, r.method), 0) for r in summary.itertuples()]
    summary['failure_rate'] = summary['n_failed'] / n_rep
    if len(faildf):
        fr = faildf.groupby(['scenario', 'lane', 'method']).size().rename('n_failed').reset_index()
    else:
        fr = pd.DataFrame(columns=['scenario', 'lane', 'method', 'n_failed'])
    write_csv('f3', 'f3_per_rep.csv', df)
    write_csv('f3', 'f3_summary.csv', summary)
    write_csv('f3', 'f3_failures.csv', faildf)
    write_json('f3', 'f3_report.json', {'status': 'complete_with_recorded_failures' if failures else 'complete', 'configured_reps': n_rep, 'named_scenarios': list(specs), 'grid': {'correlation': [0, 0.25, 0.5, 0.75, 1.0], 'flip': [0, 0.05, 0.1, 0.15, 0.2]}, 'failure_summary': fr.to_dict('records'), 'no_silent_fallback': True})
    return (df, summary)

def load_real():
    cons = pd.read_csv(OUT05 / 'results_stage5/data/consensus_dataset.csv', dtype=str).fillna('')
    test = cons.loc[cons.split.eq('test')].copy()
    test['consensus_idx'] = test.index
    idx = test.consensus_idx.to_numpy(int)
    arm = np.load(OUT05 / 'results_stage7/reports/arm_predictions.npz')
    gold = arm['gold_bbbp'].astype(int)
    from_cons = pd.to_numeric(test.bbbp_label).to_numpy(int)
    assert len(test) == len(idx) == len(gold) == 291
    assert len(np.unique(idx)) == 291 and (not np.array_equal(idx, np.arange(291)))
    assert test.parent_ik.is_unique and np.array_equal(from_cons, gold), 'parent_ik/gold order mismatch'
    return (cons, test, idx, gold, arm)

def tensors_for(cons, test_idx, *, drop_logbb=False, freeze_delta=False, l2=0.1, exclude_refs=(), allowed_levels=None, merge_duplicates=False, full_bbbp=False):
    T = build_tensors(cons, use_reference_out=True)
    old_idx = T['bbbp_idx'].copy()
    old_labels = T['bbbp_labels'].copy()
    keep = np.ones(len(old_idx), bool) if full_bbbp else ~np.isin(old_idx, test_idx)
    T['bbbp_idx'] = old_idx[keep]
    T['bbbp_labels'] = old_labels[keep]
    assert len(T['bbbp_idx']) == len(T['bbbp_labels'])
    if not full_bbbp:
        assert not np.isin(T['bbbp_idx'], test_idx).any()
    assert np.array_equal(T['bbbp_labels'], old_labels[keep])
    remove = np.zeros(len(cons), bool)
    if exclude_refs:
        pat = '|'.join(exclude_refs)
        remove |= cons.reference_list.str.contains(pat, regex=True).to_numpy()
    if allowed_levels is not None:
        prov = pd.read_csv(OUT05 / 'results_e1/provenance_table.csv', dtype=str).fillna('')
        level = cons.parent_ik.map(dict(zip(prov.parent_ik, prov.level))).fillna('unknown')
        remove |= ~level.isin(allowed_levels).to_numpy()
    kb = ~np.isin(T['b3db_idx'], np.flatnonzero(remove))
    kl = ~np.isin(T['logbb_idx'], np.flatnonzero(remove))
    T['b3db_idx'], T['b3db_labels'] = (T['b3db_idx'][kb], T['b3db_labels'][kb])
    T['logbb_idx'], T['logbb_vals'] = (T['logbb_idx'][kl], T['logbb_vals'][kl])
    if merge_duplicates and len(T['b3db_idx']):
        z = pd.DataFrame({'i': T['b3db_idx'], 'y': T['b3db_labels']}).groupby('i').y.mean()
        T['b3db_idx'] = z.index.to_numpy(int)
        T['b3db_labels'] = (z.to_numpy() >= 0.5).astype(float)
    if drop_logbb:
        T['logbb_idx'], T['logbb_vals'] = (np.array([], int), np.array([], float))
    return (T, {'freeze_delta': freeze_delta, 'l2': l2, 'n_removed_molecules': int(remove.sum())})

def fit_real(T, cfg, init_seed=SEED):
    rng = np.random.default_rng(init_seed)
    torch.manual_seed(init_seed)
    m = LatentHEM(T['n'], n_em_iters=20, m_epochs=30, m_lr=0.01, l2_prior=cfg['l2'], freeze_delta=cfg['freeze_delta'])
    if init_seed != SEED:
        m.mu = rng.normal(0, 0.15, T['n'])
        m.delta = torch.tensor(float(rng.normal(0, 0.1)), dtype=torch.float64, requires_grad=True)
    m.fit(T['b3db_idx'], T['b3db_labels'], bbbp_idx=T['bbbp_idx'], bbbp_labels=T['bbbp_labels'], logbb_idx=T['logbb_idx'], logbb_vals=T['logbb_vals'])
    return m

def f4(smoke=False):
    cons, test, idx, gold, _ = load_real()
    prov = pd.read_csv(OUT05 / 'results_e1/provenance_table.csv', dtype=str).fillna('')
    levels = sorted(prov.level.unique())
    l1 = [x for x in levels if x.startswith('L1')]
    l2 = l1 + [x for x in levels if x.startswith('L2')]
    l3 = l2 + [x for x in levels if x.startswith('L3')]
    specs = {'full_HEM': {}, 'no_logBB': {'drop_logbb': True}, 'no_delta': {'freeze_delta': True}, 'l2_prior_0.01': {'l2': 0.01}, 'l2_prior_1.0': {'l2': 1.0}, 'exclude_R19_R22': {'exclude_refs': ('R19', 'R22')}, 'merge_duplicate_observations': {'merge_duplicates': True}, 'L1': {'allowed_levels': l1}, 'L1_L2': {'allowed_levels': l2}, 'L1_L2_L3': {'allowed_levels': l3}, 'fixed_channel_reliability': {'freeze_delta': True}, 'no_channel_reliability_model': {'freeze_delta': True, 'merge_duplicates': True}}
    if smoke:
        specs = {k: specs[k] for k in ['full_HEM', 'no_logBB']}
    preds = {}
    rows = []
    for name, kw in specs.items():
        T, cfg = tensors_for(cons, idx, **kw)
        t = time.time()
        m = fit_real(T, cfg)
        p = sigmoid(m.mu[idx])
        assert len(p) == len(gold) == len(m.sigma[idx])
        preds[name] = (p, m.sigma[idx])
        met = metric_row(gold, p)
        rows.append({'ablation': name, **met, 'mean_sigma': np.mean(m.sigma[idx]), 'delta': float(m.delta), 'l2': cfg['l2'], 'n_removed_molecules': cfg['n_removed_molecules'], 'runtime_s': time.time() - t})
    base = preds['full_HEM'][0]
    effects = []
    for j, (name, (p, s)) in enumerate(preds.items()):
        effects.append({'ablation': name, **paired_auc_bootstrap(gold, p, base, SEED + j)})
    predout = test[['consensus_idx', 'parent_ik']].copy()
    predout['gold_bbbp'] = gold
    for name, (p, s) in preds.items():
        predout[f'p__{name}'] = p
        predout[f'sigma__{name}'] = s
    write_csv('f4', 'f4_predictions.csv', predout)
    write_csv('f4', 'f4_metrics.csv', pd.DataFrame(rows))
    write_csv('f4', 'f4_paired_bootstrap.csv', pd.DataFrame(effects))
    write_json('f4', 'f4_report.json', {'status': 'complete', 'description': "cross-channel consistency diagnostic retaining test molecules' B3DB/logBB evidence while excluding their BBBP channel; not unseen-molecule generalization", 'index_assertions': {'n': 291, 'global_min': int(idx.min()), 'global_max': int(idx.max()), 'not_0_to_290': True, 'parent_ik_gold_order': True, 'same_mask': True}, 'ablation_metrics': rows, 'paired_delta_auc': effects, 'source_reliability_scope': 'The current HEM has channel offset delta, not identifiable per-reference reliability. Fixed/no-reliability variants therefore test channel-level offset handling only.'})
    return (cons, test, idx, gold, predout, pd.DataFrame(effects))

def identifiability_audit(train):
    nlab = train.b3db_raw_labels.map(lambda x: len([z for z in str(x).split(';') if z in ('0', '1')]))
    nref = train.reference_list.map(lambda x: len([z for z in str(x).split('|') if z]))
    exact = nlab == nref
    mapped = (exact & nlab.ge(2)).sum()
    identifiable = bool(mapped >= 50 and exact.mean() >= 0.9)
    return {'n_train': len(train), 'rows_with_raw_labels': int(nlab.gt(0).sum()), 'rows_with_multiple_refs': int(nref.ge(2).sum()), 'label_reference_count_match_rate': float(exact.mean()), 'rows_with_mappable_multi_source_labels': int(mapped), 'identifiable': identifiable, 'decision': 'N/A' if not identifiable else 'implement'}

def fingerprints(smiles):
    gen = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS, includeChirality=True)
    X = np.zeros((len(smiles), N_BITS), np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f'invalid SMILES at row {i}')
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), X[i])
    return X

def f5():
    cons, test, idx, gold, _ = load_real()
    train = cons[cons.split.eq('train') & cons.in_reference_out.astype(int).eq(0)].reset_index(drop=True)
    audit = identifiability_audit(train)
    a = pd.to_numeric(train.b3db_label, errors='coerce').to_numpy()
    labs = pd.read_csv(OUT05 / 'results_stage7/data/labels_abc.csv', dtype=str).fillna('')
    assert np.array_equal(train.parent_ik.to_numpy(), labs.parent_ik.to_numpy())
    labels = {'A_hard': a, 'B_soft': pd.to_numeric(labs.label_B).to_numpy(float)}
    T = build_tensors(train, use_reference_out=False)
    hem = LatentHEM(T['n'], n_em_iters=25, m_epochs=50, m_lr=0.01, l2_prior=0.1)
    hem.fit(T['b3db_idx'], T['b3db_labels'], bbbp_idx=T['bbbp_idx'], bbbp_labels=T['bbbp_labels'], logbb_idx=T['logbb_idx'], logbb_vals=T['logbb_vals'])
    labels['C_HEM'] = sigmoid(hem.mu)
    mask = np.isfinite(a)
    Xtr = fingerprints(train.canon_smiles)
    Xte = fingerprints(test.canon_smiles)
    rows = []
    pred = test[['consensus_idx', 'parent_ik']].copy()
    pred['gold_bbbp'] = gold
    for name, y in labels.items():
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(Xtr[mask], y[mask])
        p = model.predict(Xte)
        pred[f'p__{name}'] = p
        rows.append({'arm': name, **metric_row(gold, p)})
    effects = []
    for j, name in enumerate(['B_soft', 'C_HEM']):
        effects.append({'contrast': f'{name} - A_hard', **paired_auc_bootstrap(gold, pred[f'p__{name}'], pred['p__A_hard'], SEED + j)})
    na = [{'method': m, 'status': 'N/A', 'reason': 'molecule-by-source label matrix is not reliably identifiable'} for m in ['Dawid-Skene', 'Snorkel', 'MAP latent class']]
    write_csv('f5', 'f5_predictions.csv', pred)
    write_csv('f5', 'f5_metrics.csv', pd.DataFrame(rows))
    write_csv('f5', 'f5_paired_bootstrap.csv', pd.DataFrame(effects))
    write_json('f5', 'f5_report.json', {'status': 'complete_with_NA_methods', 'protocol': {'morgan_radius': 2, 'bits': 2048, 'chirality': True, 'xgboost': XGB_PARAMS, 'training_molecules': int(mask.sum()), 'split': 'same frozen stage7 split', 'tuning_budget': 'none; fixed stage7 parameters'}, 'identifiability': audit, 'unavailable_methods': na, 'metrics': rows, 'paired_delta_auc': effects, 'no_fallback': True})
    return (pred, pd.DataFrame(effects))

def bootstrap_spearman(x, y, n=2000):
    x, y = (np.asarray(x, float), np.asarray(y, float))
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = (x[ok], y[ok])
    point = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(SEED)
    v = []
    for _ in range(n):
        ii = rng.integers(0, len(x), len(x))
        v.append(stats.spearmanr(x[ii], y[ii]).statistic)
    return {'rho': point, 'ci95': np.nanpercentile(v, [2.5, 97.5]).tolist(), 'n': len(x)}

def f6(cons, test, idx, gold, f4pred):
    prior = []
    init = []
    models = {}
    for l2 in [0.01, 0.1, 1.0]:
        T, cfg = tensors_for(cons, idx, l2=l2)
        m = fit_real(T, cfg)
        p = sigmoid(m.mu[idx])
        models[f'l2={l2}'] = (m, p)
        prior.append({'l2': l2, **metric_row(gold, p), 'mean_sigma': m.sigma[idx].mean()})
    T, cfg = tensors_for(cons, idx)
    for seed in [11, 22, 33, 44, 55]:
        m = fit_real(T, cfg, seed)
        p = sigmoid(m.mu[idx])
        init.append({'seed': seed, **metric_row(gold, p), 'mean_sigma': m.sigma[idx].mean()})
    full = models['l2=0.1'][0]
    pfull = models['l2=0.1'][1]
    s = full.sigma[idx]
    nref = pd.to_numeric(test.n_references, errors='coerce').fillna(0).to_numpy()
    corr = bootstrap_spearman(s, nref)
    conflict = pd.to_numeric(test.b3db_label, errors='coerce').to_numpy() != gold
    cover = []
    for c in np.linspace(0.1, 1, 10):
        k = max(1, int(len(gold) * c))
        keep = np.argsort(s)[:k]
        cover.append({'coverage': k / len(gold), 'risk': float(((pfull[keep] >= 0.5) != gold[keep]).mean()), 'n': k})
    bins = pd.qcut(pfull, 10, duplicates='drop')
    cal = pd.DataFrame({'p': pfull, 'y': gold, 'bin': bins}).groupby('bin', observed=True).agg(n=('y', 'size'), mean_pred=('p', 'mean'), observed=('y', 'mean')).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    ppc = []
    for _ in range(1000):
        ppc.append(rng.binomial(1, pfull).mean())
    ppc_obj = {'observed_bbbp_prevalence': float(gold.mean()), 'replicated_prevalence_mean': float(np.mean(ppc)), 'replicated_prevalence_ci95': np.percentile(ppc, [2.5, 97.5]).tolist()}
    sources = Counter((r for z in cons.reference_list for r in str(z).split('|') if r))
    loo = []
    for j, ref in enumerate(['R27'] + [r for r, _ in sources.most_common(5) if r != 'R27']):
        T0, c0 = tensors_for(cons, idx, exclude_refs=(ref,))
        m0 = fit_real(T0, c0)
        p0 = sigmoid(m0.mu[idx])
        e = paired_auc_bootstrap(gold, p0, pfull, SEED + 100 + j)
        loo.append({'source': ref, 'n_molecules_with_source': int(cons.reference_list.str.contains(f'(?:^|\\|){ref}(?:\\||$)', regex=True).sum()), **e, 'mean_abs_probability_change': float(np.mean(abs(p0 - pfull)))})
    fullT, fullcfg = tensors_for(cons, idx, full_bbbp=True)
    retrospective = fit_real(fullT, fullcfg)
    retro_p = sigmoid(retrospective.mu[idx])
    prov = pd.read_csv(OUT05 / 'results_e1/provenance_table.csv', dtype=str).fillna('')
    level = test.parent_ik.map(dict(zip(prov.parent_ik, prov.level))).fillna('unknown')
    effects = pd.DataFrame({'sigma': s, 'nref': nref, 'conflict': conflict, 'duplicate': level.str.startswith('L2').to_numpy()}).groupby(['conflict', 'duplicate']).sigma.agg(['count', 'mean', 'std']).reset_index()
    write_csv('f6', 'f6_prior_sensitivity.csv', pd.DataFrame(prior))
    write_csv('f6', 'f6_initialization_stability.csv', pd.DataFrame(init))
    write_csv('f6', 'f6_calibration.csv', cal)
    write_csv('f6', 'f6_risk_coverage.csv', pd.DataFrame(cover))
    write_csv('f6', 'f6_leave_one_source_out.csv', pd.DataFrame(loo))
    write_csv('f6', 'f6_sigma_effects.csv', effects)
    write_csv('f6', 'f6_predictions.csv', pd.DataFrame({'consensus_idx': idx, 'parent_ik': test.parent_ik, 'gold_bbbp': gold, 'loo_prob': pfull, 'loo_sigma': s, 'retrospective_prob': retro_p, 'retrospective_sigma': retrospective.sigma[idx]}))
    write_json('f6', 'f6_posterior_analysis.json', {'status': 'complete', 's_definition': 's_i is the Laplace posterior latent standard deviation sigma_i=1/sqrt(local negative-log-posterior curvature); larger s means less certainty', 'index_assertions': {'n': 291, 'global_indices': True, 'mu_sigma_gold_lengths_equal': True, 'parent_ik_gold_order': True}, 'prior_sensitivity': prior, 'initialization_stability': init, 'posterior_predictive_check': ppc_obj, 'corr_s_nref': corr, 'conflict_mean_s': float(s[conflict].mean()), 'nonconflict_mean_s': float(s[~conflict].mean()), 'failure_mode_conflict_more_certain': bool(s[conflict].mean() < s[~conflict].mean()), 'leave_one_source_out': loo, 'r27_actual_exclusion': True, 'r27_exclusion_scope': 'all B3DB binary and logBB observations for molecules listing R27; conservative because per-reference label mapping is unavailable', 'brier': metric_row(gold, pfull)['brier'], 'ppc': ppc_obj})
    return (pd.DataFrame({'consensus_idx': idx, 'parent_ik': test.parent_ik, 'gold': gold, 'loo_prob': pfull, 'loo_sigma': s, 'retro_prob': retro_p, 'retro_sigma': retrospective.sigma[idx]}), pd.DataFrame(loo))

def f7(cons, test, idx, gold, f6pred):
    arm = np.load(OUT05 / 'results_stage7/reports/arm_predictions.npz')
    hard = arm['A_BBBP_test']
    b3 = pd.to_numeric(test.b3db_label, errors='coerce').to_numpy()
    bb = pd.to_numeric(test.bbbp_label, errors='coerce').to_numpy()
    logbb = test.b3db_logbb_list.map(lambda x: np.mean(parse_logbb_list(x)) if parse_logbb_list(x) else np.nan).to_numpy()
    prov = pd.read_csv(OUT05 / 'results_e1/provenance_table.csv', dtype=str).fillna('')
    level = test.parent_ik.map(dict(zip(prov.parent_ik, prov.level))).fillna('unknown')
    loo = f6pred.loo_prob.to_numpy()
    retro = f6pred.retro_prob.to_numpy()
    s = f6pred.loo_sigma.to_numpy()
    majority_hard = (hard >= 0.5).astype(int)
    src_conf = np.isfinite(b3) & np.isfinite(bb) & (b3 != bb)
    bbbp_dis = (loo >= 0.5) != gold
    log_dis = np.isfinite(logbb) & ((logbb > 0) & (gold == 0) | (logbb < 0) & (gold == 1))
    top = np.argsort(-s)[:20]
    selected = np.unique(np.r_[top, np.flatnonzero(src_conf), np.flatnonzero(bbbp_dis), np.flatnonzero(log_dis), np.flatnonzero(level.str.startswith('L2'))])[:80]
    matrix = pd.DataFrame({'consensus_idx': idx[selected], 'parent_ik': test.parent_ik.iloc[selected].to_numpy(), 'bbbp_label': gold[selected], 'b3db_label': b3[selected], 'majority_or_hard_prob': hard[selected], 'hem_loo_prob': loo[selected], 'hem_loo_sigma': s[selected], 'hem_full_refit_prob_retrospective': retro[selected], 'logbb_mean': logbb[selected], 'provenance_level': level.iloc[selected].to_numpy(), 'n_references': pd.to_numeric(test.n_references, errors='coerce').to_numpy()[selected], 'source_label_conflict': src_conf[selected], 'bbbp_label_disagreement': bbbp_dis[selected], 'logbb_direction_disagreement': log_dis[selected]})
    statsobj = {'n_test': 291, 'n_selected': len(matrix), 'source_label_conflict_rate': float(src_conf.mean()), 'hem_loo_vs_hard_disagreement_rate': float(((loo >= 0.5) != majority_hard).mean()), 'hem_loo_vs_bbbp_label_disagreement_rate': float(bbbp_dis.mean()), 'logbb_vs_bbbp_direction_disagreement_rate': float(log_dis.mean()), 'loo_sigma_mean': float(s.mean()), 'loo_sigma_range': [float(s.min()), float(s.max())]}
    write_csv('f7', 'f7_case_matrix.csv', matrix)
    write_json('f7', 'f7_case_audit.json', {'status': 'complete_exploratory', 'task_name': '探索性计算案例筛选', 'statistics': statsobj, 'cases': matrix.to_dict('records'), 'gold_standard_warning': 'BBBP is a source label, not an independent gold standard; all error wording is replaced by disagreement wording', 'literature_review_performed': False})
    return (matrix, statsobj)

def save_figure(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in [('png', {'dpi': 300}), ('svg', {}), ('pdf', {})]:
        fig.savefig(FIGDIR / f'{name}.{ext}', bbox_inches='tight', **kw)
    plt.close(fig)

def figures(f2s, f3s, f4eff, f5eff, loo, cases):
    x = f2s[(f2s.scenario == 'base') & (f2s.lane == 'binary')].sort_values('mean_auc')
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.errorbar(x.mean_auc, range(len(x)), xerr=[x.mean_auc - x.ci95_auc_lo, x.ci95_auc_hi - x.mean_auc], fmt='o', color='#2F6B8A')
    ax.set_yticks(range(len(x)), x.method)
    ax.set_xlabel('AUROC (mean and 95% CI)')
    ax.set_title('F2 | Fair binary information set: Y0 + Y1 + Y2')
    save_figure(fig, 'Fig_F2_fair_comparison')
    z = f3s[(f3s.lane == 'binary') & (f3s.method == 'hem') & f3s.scenario.str.startswith('grid_')].copy()
    z[['corr', 'flip']] = z.scenario.str.extract('corr=([0-9.]+)_flip=([0-9.]+)').astype(float)
    mat = z.pivot(index='corr', columns='flip', values='mean_auc').sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0.5, vmax=1)
    ax.set_xticks(range(len(mat.columns)), mat.columns)
    ax.set_yticks(range(len(mat.index)), mat.index)
    ax.set_xlabel('Flip rate')
    ax.set_ylabel('Copy probability')
    ax.set_title('F3 | HEM stress surface')
    fig.colorbar(im, ax=ax, label='Mean AUROC')
    save_figure(fig, 'Fig_F3_stress_heatmap')
    extreme = f3s[(f3s.lane == 'binary') & ~f3s.scenario.str.startswith('grid_')].copy().sort_values(['scenario', 'mean_auc'])
    extreme.to_csv(FIGDIR / 'Fig_F3_extreme_forest_source.csv', index=False, encoding='utf-8-sig')
    labels = [f'{r.scenario} | {r.method}' for r in extreme.itertuples()]
    y = np.arange(len(extreme))
    fig, ax = plt.subplots(figsize=(6.4, 9.2))
    ax.errorbar(extreme.mean_auc, y, xerr=[extreme.mean_auc - extreme.ci95_auc_lo, extreme.ci95_auc_hi - extreme.mean_auc], fmt='o', ms=3, color='#2F6B8A')
    ax.axvline(0.5, color='#777', lw=0.8)
    ax.set_yticks(y, labels, fontsize=5)
    ax.set_xlabel('AUROC (95% CI across repeats)')
    ax.set_title('F3 | Named extreme stress scenarios')
    save_figure(fig, 'Fig_F3_extreme_forest')
    for data, label, title, name in [(f4eff, 'ablation', 'F4 | Paired delta AUROC vs full HEM', 'Fig_F4_ablation_effects'), (f5eff, 'contrast', 'F5 | Paired downstream delta AUROC', 'Fig_F5_downstream_deltas')]:
        d = data.copy()
        fig, ax = plt.subplots(figsize=(5, max(2.3, 0.28 * len(d))))
        y = np.arange(len(d))
        ax.errorbar(d.delta_auc, y, xerr=[d.delta_auc - d.ci95.apply(lambda q: q[0]), d.ci95.apply(lambda q: q[1]) - d.delta_auc], fmt='o', color='#2F6B8A')
        ax.axvline(0, color='#777', lw=0.8)
        ax.set_yticks(y, d[label])
        ax.set_xlabel('Delta AUROC (95% paired bootstrap CI)')
        ax.set_title(title)
        save_figure(fig, name)
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    d = loo.sort_values('delta_auc')
    y = np.arange(len(d))
    ax.errorbar(d.delta_auc, y, xerr=[d.delta_auc - d.ci95.apply(lambda q: q[0]), d.ci95.apply(lambda q: q[1]) - d.delta_auc], fmt='o', color='#A16B45')
    ax.axvline(0, color='#777', lw=0.8)
    ax.set_yticks(y, d.source)
    ax.set_xlabel('Delta AUROC after source exclusion')
    ax.set_title('F6 | Leave-one-source-out sensitivity')
    save_figure(fig, 'Fig_F6_leave_one_source_out')
    show = cases.sort_values('hem_loo_sigma', ascending=False).head(35)
    vals = show[['bbbp_label', 'b3db_label', 'majority_or_hard_prob', 'hem_loo_prob', 'hem_full_refit_prob_retrospective', 'logbb_mean']].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.8, 6))
    im = ax.imshow(vals, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(vals.shape[1]), ['BBBP', 'B3DB', 'Hard p', 'HEM LOO p', 'HEM full p', 'logBB'], rotation=35, ha='right')
    ax.set_yticks(range(len(show)), show.parent_ik.str[:10], fontsize=5)
    ax.set_title('F7 | Exploratory case matrix')
    fig.colorbar(im, ax=ax, label='Aligned value')
    save_figure(fig, 'Fig_F7_case_matrix')
    contracts = {'backend': 'Python/matplotlib only', 'archetype': 'quantitative grid', 'core_conclusion': 'Corrected, paired analyses replace invalid v0.9 F4/F6/F7 claims', 'statistics': 'source CSVs in results_f2_v10 to results_f7_v10', 'exports': ['PNG 300 dpi', 'SVG editable text', 'PDF editable text']}
    (FIGDIR / 'figure_contract_v10.json').write_text(json.dumps(contracts, indent=2), encoding='utf-8')

def hash_sync():
    rows = []
    for fid in ['f2', 'f3', 'f4', 'f5', 'f6', 'f7']:
        a, b = outdirs(fid)
        for p in sorted(a.iterdir()):
            if p.is_file():
                q = b / p.name
                ha = hashlib.sha256(p.read_bytes()).hexdigest()
                hb = hashlib.sha256(q.read_bytes()).hexdigest() if q.exists() else None
                rows.append({'result': fid, 'file': p.name, 'sha256_02': ha, 'sha256_05': hb, 'match': ha == hb})
    df = pd.DataFrame(rows)
    write_csv('f7', 'v10_cross_workspace_hashes.csv', df)
    if not df.match.all():
        raise AssertionError('02/05 result hash mismatch')
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--only', choices=['f2', 'f3', 'f4', 'f5', 'f6', 'f7'])
    args = ap.parse_args()
    n2 = 2 if args.smoke else 30
    n3 = 2 if args.smoke else 20
    cache = {}
    if args.only in (None, 'f2'):
        cache['f2'] = f2(n2)
    if args.only in (None, 'f3'):
        cache['f3'] = f3(n3)
    if args.only in (None, 'f4') or args.only in ('f6', 'f7'):
        cache['f4'] = f4(args.smoke)
    if args.only in (None, 'f5'):
        cache['f5'] = f5()
    if args.only in (None, 'f6') or args.only == 'f7':
        cons, test, idx, gold, pred, eff = cache.get('f4') or f4(args.smoke)
        cache['f6'] = (f6(cons, test, idx, gold, pred), cons, test, idx, gold, eff)
    if args.only in (None, 'f7'):
        (f6pred, loo), cons, test, idx, gold, f4eff = cache['f6']
        cache['f7'] = f7(cons, test, idx, gold, f6pred)
    if args.only is None and (not args.smoke):
        f2df, f2s = cache['f2']
        f3df, f3s = cache['f3']
        f5pred, f5eff = cache['f5']
        (f6pred, loo), cons, test, idx, gold, f4eff = cache['f6']
        cases, _ = cache['f7']
        figures(f2s, f3s, f4eff, f5eff, loo, cases)
        hash_sync()
    print(json.dumps({'status': 'complete', 'smoke': args.smoke, 'only': args.only}, ensure_ascii=False))
if __name__ == '__main__':
    main()

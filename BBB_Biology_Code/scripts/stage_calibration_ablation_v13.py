from __future__ import annotations
from project_paths import project_root, project_path
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
ROOT = project_root()
CODE = ROOT / '01_代码_各阶段'
WORK = ROOT / '05_复现工作区'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(WORK))
from hem import LatentHEM
from hem.data import build_tensors
from stage_external_science_v13 import metrics, bootstrap_metrics, write_csv_both, write_json_both, result_dirs, sha256
SEED = 20260818

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(np.asarray(x, float), -30, 30)))

def load_aligned():
    cons = pd.read_csv(WORK / 'results_stage5' / 'data' / 'consensus_dataset.csv', dtype=str).fillna('')
    test = cons.loc[cons['split'].eq('test')].copy()
    test['consensus_idx'] = test.index
    idx = test['consensus_idx'].to_numpy(int)
    pred = pd.read_csv(ROOT / '02_结果' / 'results_f6_v10' / 'f6_predictions.csv')
    assert len(test) == len(idx) == len(pred) == 291
    assert len(np.unique(idx)) == 291 and (not np.array_equal(idx, np.arange(291)))
    assert np.array_equal(test['parent_ik'].to_numpy(), pred['parent_ik'].to_numpy())
    gold = pd.to_numeric(test['bbbp_label']).to_numpy(int)
    assert np.array_equal(gold, pred['gold_bbbp'].to_numpy(int))
    return (cons, test, idx, gold, pred)

def ece_bins(y, p, model, n_bins=10):
    p = np.clip(np.asarray(p, float), 0, 1)
    y = np.asarray(y, int)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for k in range(n_bins):
        take = b == k
        if take.any():
            rows.append({'model': model, 'bin': k + 1, 'lower': k / n_bins, 'upper': (k + 1) / n_bins, 'n': int(take.sum()), 'mean_pred': float(p[take].mean()), 'observed_rate': float(y[take].mean())})
    return rows

def crossfit_platt(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-06, 1 - 1e-06)
    x = np.log(p / (1 - p))[:, None]
    out = np.full(len(y), np.nan)
    params = []
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(x, y), 1):
        m = LogisticRegression(C=1000000.0, solver='lbfgs', max_iter=1000).fit(x[tr], y[tr])
        out[te] = m.predict_proba(x[te])[:, 1]
        params.append({'fold': fold, 'n_train': len(tr), 'n_test': len(te), 'intercept': float(m.intercept_[0]), 'slope': float(m.coef_[0, 0])})
    assert np.isfinite(out).all()
    return (out, pd.DataFrame(params))

def calibration_analysis():
    cons, test, idx, gold, old = load_aligned()
    raw = old['loo_prob'].to_numpy(float)
    sigma = old['loo_sigma'].to_numpy(float)
    calibrated, fold_params = crossfit_platt(gold, raw)
    prediction = test[['consensus_idx', 'parent_ik']].copy()
    prediction['gold_bbbp'] = gold
    prediction['loo_probability_raw'] = raw
    prediction['loo_probability_crossfit_platt'] = calibrated
    prediction['loo_sigma'] = sigma
    prediction['n_references'] = pd.to_numeric(test['n_references'], errors='coerce').fillna(0).astype(int).to_numpy()
    b3 = pd.to_numeric(test['b3db_label'], errors='coerce').to_numpy(float)
    prediction['conflict_status'] = np.where(~np.isfinite(b3), 'B3DB_missing', np.where(b3.astype(float) != gold, 'conflict', 'nonconflict'))
    prediction['source_count_stratum'] = pd.cut(prediction['n_references'], bins=[-1, 0, 5, 10, np.inf], labels=['0', '1-5', '6-10', '>10']).astype(str)
    write_csv_both('calibration', 'calibration_predictions.csv', prediction)
    write_csv_both('calibration', 'crossfit_calibration_parameters.csv', fold_params)
    metric_df = pd.DataFrame([{'model': 'raw_LOO', **metrics(gold, raw)}, {'model': 'crossfit_platt', **metrics(gold, calibrated)}])
    write_csv_both('calibration', 'calibration_metrics.csv', metric_df)
    rel = pd.DataFrame(ece_bins(gold, raw, 'raw_LOO') + ece_bins(gold, calibrated, 'crossfit_platt'))
    write_csv_both('calibration', 'reliability_bins.csv', rel)
    strata = []
    for variable in ['source_count_stratum', 'conflict_status']:
        for level, x in prediction.groupby(variable):
            y = x['gold_bbbp'].to_numpy(int)
            if len(x) >= 10 and len(np.unique(y)) == 2:
                for col, name in [('loo_probability_raw', 'raw_LOO'), ('loo_probability_crossfit_platt', 'crossfit_platt')]:
                    strata.append({'stratifier': variable, 'level': level, 'model': name, **metrics(y, x[col].to_numpy(float)), 'mean_sigma': float(x['loo_sigma'].mean())})
            else:
                strata.append({'stratifier': variable, 'level': level, 'model': 'N/A', 'n': len(x), 'reason': 'fewer than 10 molecules or one outcome class'})
    strata_df = pd.DataFrame(strata)
    write_csv_both('calibration', 'calibration_strata.csv', strata_df)
    risk = []
    order = np.argsort(sigma)
    for coverage in np.linspace(0.1, 1, 19):
        n = max(1, int(round(len(gold) * coverage)))
        keep = order[:n]
        risk.append({'coverage': n / len(gold), 'n': n, 'classification_risk': float(np.mean((raw[keep] >= 0.5) != gold[keep])), 'brier_risk': float(np.mean((raw[keep] - gold[keep]) ** 2)), 'mean_sigma': float(sigma[keep].mean())})
    risk_df = pd.DataFrame(risk)
    write_csv_both('calibration', 'risk_coverage.csv', risk_df)
    rng = np.random.default_rng(SEED)
    replicated = np.array([rng.binomial(1, raw).mean() for _ in range(5000)])
    ppc = {'observed_positive_rate': float(gold.mean()), 'replicated_positive_rate_mean': float(replicated.mean()), 'replicated_positive_rate_ci95': np.percentile(replicated, [2.5, 97.5]).tolist(), 'observed_inside_interval': bool(np.percentile(replicated, 2.5) <= gold.mean() <= np.percentile(replicated, 97.5))}
    prior = pd.read_csv(ROOT / '02_结果' / 'results_f6_v10' / 'f6_prior_sensitivity.csv')
    init = pd.read_csv(ROOT / '02_结果' / 'results_f6_v10' / 'f6_initialization_stability.csv')
    loo = pd.read_csv(ROOT / '02_结果' / 'results_f6_v10' / 'f6_leave_one_source_out.csv')
    r27 = loo.loc[loo['source'].eq('R27')].to_dict('records')
    old_report = json.loads((ROOT / '02_结果' / 'results_f6_v10' / 'f6_posterior_analysis.json').read_text(encoding='utf-8'))
    report = {'status': 'complete_with_calibration_failure', 'index_assertions': {'n': 291, 'global_index_min': int(idx.min()), 'global_index_max': int(idx.max()), 'not_zero_to_290': True, 'parent_ik_gold_order': True}, 's_definition': old_report['s_definition'], 'raw_metrics': metric_df.iloc[0].to_dict(), 'crossfit_calibration_metrics': metric_df.iloc[1].to_dict(), 'crossfit_scope': 'five-fold post-hoc calibration within the 291 evaluation molecules; demonstrates correctability, not prospective external calibration', 'posterior_predictive_check': ppc, 'calibration_failure_preserved': not ppc['observed_inside_interval'], 'corr_s_nref': old_report['corr_s_nref'], 'prior_sensitivity_rows': len(prior), 'initialization_rows': len(init), 'r27_leave_one_source_out': r27, 'r27_exclusion_actually_refit': bool(r27 and abs(float(r27[0]['mean_abs_probability_change'])) > 0), 'interpretation': 'posterior uncertainty tracks evidence quantity, but raw posterior probabilities remain insufficiently calibrated'}
    write_json_both('calibration', 'calibration_report.json', report)
    return report

def prepare_tensors(cons, idx):
    T = build_tensors(cons, use_reference_out=True)
    old_idx = T['bbbp_idx'].copy()
    old_labels = T['bbbp_labels'].copy()
    keep = ~np.isin(old_idx, idx)
    T['bbbp_idx'] = old_idx[keep]
    T['bbbp_labels'] = old_labels[keep]
    assert len(T['bbbp_idx']) == len(T['bbbp_labels'])
    assert not np.isin(T['bbbp_idx'], idx).any()
    assert np.array_equal(T['bbbp_labels'], old_labels[keep])
    return T

def fit_variant(base, variant, idx):
    b3i, b3y = (base['b3db_idx'].copy(), base['b3db_labels'].copy())
    bbi, bby = (base['bbbp_idx'].copy(), base['bbbp_labels'].copy())
    lbi, lbv = (base['logbb_idx'].copy(), base['logbb_vals'].copy())
    note = 'HEM'
    if variant in {'no_logBB', 'no_continuous_channel'}:
        lbi, lbv = (np.array([], int), np.array([], float))
    elif variant == 'no_BBBP':
        bbi, bby = (np.array([], int), np.array([], float))
    elif variant == 'no_B3DB':
        b3i, b3y = (bbi, bby)
        bbi, bby = (np.array([], int), np.array([], float))
        note = 'BBBP-anchored HEM reparameterization'
    elif variant == 'permuted_logBB':
        rng = np.random.default_rng(SEED)
        lbv = rng.permutation(lbv)
    elif variant == 'train_only_standardized_logBB':
        train_vals = lbv[~np.isin(lbi, idx)]
        mean, sd = (float(train_vals.mean()), float(train_vals.std(ddof=0)))
        if not sd > 0:
            raise ValueError('zero training logBB standard deviation')
        lbv = (lbv - mean) / sd
        note = f'HEM; logBB standardized using non-test observations only (mean={mean:.6g}, sd={sd:.6g})'
    model = LatentHEM(base['n'], n_em_iters=20, m_epochs=30, m_lr=0.01, l2_prior=0.1)
    torch.manual_seed(SEED)
    model.fit(b3i, b3y, bbbp_idx=bbi, bbbp_labels=bby, logbb_idx=lbi, logbb_vals=lbv)
    p = sigmoid(model.mu[idx])
    assert len(p) == len(idx) and np.isfinite(p).all()
    return (p, model.sigma[idx], note)

def logbb_only_baseline(cons, test, idx, gold):
    means = cons['b3db_logbb_list'].map(lambda z: np.mean([float(v) for v in str(z).split(';') if v.strip()]) if str(z).strip() else np.nan).to_numpy(float)
    labels = pd.to_numeric(cons['bbbp_label'], errors='coerce').to_numpy(float)
    train = np.isfinite(means) & np.isfinite(labels) & ~np.isin(np.arange(len(cons)), idx)
    if train.sum() < 30 or len(np.unique(labels[train])) < 2:
        raise ValueError('logBB-only mapping lacks non-test training pairs')
    mean, sd = (means[train].mean(), means[train].std(ddof=0))
    model = LogisticRegression(C=1000000.0, solver='lbfgs', max_iter=1000).fit(((means[train] - mean) / sd)[:, None], labels[train].astype(int))
    observed = np.isfinite(means[idx])
    p = np.full(len(idx), labels[train].mean())
    p[observed] = model.predict_proba(((means[idx][observed] - mean) / sd)[:, None])[:, 1]
    return (p, np.full(len(idx), np.nan), f'logBB-only logistic channel baseline; mapping trained on {train.sum()} non-test BBBP/logBB pairs; missing test logBB ({(~observed).sum()}) assigned training prevalence')

def ablation_analysis():
    cons, test, idx, gold, _ = load_aligned()
    base = prepare_tensors(cons, idx)
    variants = ['full', 'no_logBB', 'logBB_only', 'no_BBBP', 'no_B3DB', 'no_continuous_channel', 'permuted_logBB', 'train_only_standardized_logBB']
    predictions, sigmas, notes, failures = ({}, {}, {}, [])
    for variant in variants:
        try:
            if variant == 'logBB_only':
                p, s, note = logbb_only_baseline(cons, test, idx, gold)
            else:
                p, s, note = fit_variant(base, variant, idx)
            predictions[variant] = p
            sigmas[variant] = s
            notes[variant] = note
        except Exception as exc:
            failures.append({'variant': variant, 'error_type': type(exc).__name__, 'error': str(exc)})
    write_csv_both('ablation', 'ablation_failures.csv', pd.DataFrame(failures, columns=['variant', 'error_type', 'error']))
    if 'full' not in predictions:
        raise RuntimeError('full HEM failed')
    pred = test[['consensus_idx', 'parent_ik']].copy()
    pred['gold_bbbp'] = gold
    for name, p in predictions.items():
        pred[f'p__{name}'] = p
        pred[f'sigma__{name}'] = sigmas[name]
    write_csv_both('ablation', 'ablation_predictions.csv', pred)
    metric_df = pd.DataFrame([{'variant': name, **metrics(gold, p), 'definition': notes[name]} for name, p in predictions.items()])
    write_csv_both('ablation', 'ablation_metrics.csv', metric_df)
    ci, diffs = bootstrap_metrics(gold, predictions, 'full', seed=SEED + 1)
    write_csv_both('ablation', 'ablation_metric_bootstrap_ci.csv', ci)
    write_csv_both('ablation', 'ablation_paired_bootstrap.csv', diffs)
    prov = pd.read_csv(WORK / 'results_e1' / 'provenance_table.csv', dtype=str).fillna('')
    level_map = dict(zip(prov['parent_ik'], prov['level']))
    strata = pd.DataFrame({'gold': gold, 'provenance': test['parent_ik'].map(level_map).fillna('unknown').to_numpy(), 'conflict': np.where(pd.to_numeric(test['b3db_label'], errors='coerce').isna(), 'B3DB_missing', np.where(pd.to_numeric(test['b3db_label']) != gold, 'conflict', 'nonconflict'))})
    strata_rows = []
    for stratifier in ['provenance', 'conflict']:
        for level, rows in strata.groupby(stratifier).groups.items():
            take = np.asarray(list(rows), int)
            y = gold[take]
            if len(take) < 15 or len(np.unique(y)) < 2:
                continue
            for name, p in predictions.items():
                strata_rows.append({'stratifier': stratifier, 'level': level, 'variant': name, **metrics(y, p[take])})
    write_csv_both('ablation', 'ablation_stratified_metrics.csv', pd.DataFrame(strata_rows))
    report = {'status': 'complete_with_recorded_failures' if failures else 'complete', 'description': 'cross-channel consistency diagnostic on the same 291 molecules; BBBP test labels are excluded from HEM fitting, while available B3DB/logBB evidence is retained unless ablated', 'index_assertions': {'n': 291, 'global_min': int(idx.min()), 'global_max': int(idx.max()), 'not_zero_to_290': True, 'same_BBBP_mask': True, 'parent_ik_gold_order': True}, 'variants_completed': list(predictions), 'failures': failures, 'no_silent_fallback': True, 'variant_definitions': notes, 'important_boundary': 'channel contribution and cross-channel consistency, not unseen-molecule structural generalization', 'metrics': metric_df.to_dict('records')}
    write_json_both('ablation', 'ablation_report.json', report)
    return report

def sync_and_verify():
    target = WORK / '01_代码_各阶段'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), target / Path(__file__).name)
    rows = []
    for name in ['calibration', 'ablation']:
        a, b = result_dirs(name)
        for p in sorted(a.iterdir()):
            if p.is_file():
                q = b / p.name
                rows.append({'group': name, 'filename': p.name, 'sha256_02': sha256(p), 'sha256_05': sha256(q) if q.exists() else None, 'identical': q.exists() and sha256(p) == sha256(q)})
    out = pd.DataFrame(rows)
    write_csv_both('calibration', '02_05_sha256_manifest.csv', out)
    assert out['identical'].all()

def main():
    calibration_analysis()
    ablation_analysis()
    sync_and_verify()
    print(json.dumps({'status': 'complete'}))
if __name__ == '__main__':
    main()

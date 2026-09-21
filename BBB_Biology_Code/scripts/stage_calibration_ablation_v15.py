from __future__ import annotations
from project_paths import project_root, project_path
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
ROOT = project_root()
CODE = ROOT / '01_代码_各阶段'
WORK = ROOT / '05_复现工作区'
sys.path.insert(0, str(CODE))
import stage_external_science_v13 as io_base
io_base.VERSION = 'v15'
import stage_calibration_ablation_v13 as old
import stage_external_science_v15 as external_v15
SEED = 20260818
N_BOOT = 2000

def observed_logbb(cons: pd.DataFrame) -> np.ndarray:

    def mean_or_nan(value):
        vals = [float(x) for x in str(value).split(';') if x.strip()]
        return float(np.mean(vals)) if vals else np.nan
    return cons['b3db_logbb_list'].map(mean_or_nan).to_numpy(float)

def logbb_mapping(cons, idx):
    values = observed_logbb(cons)
    labels = pd.to_numeric(cons['bbbp_label'], errors='coerce').to_numpy(float)
    train = np.isfinite(values) & np.isfinite(labels) & ~np.isin(np.arange(len(cons)), idx)
    if train.sum() < 30 or len(np.unique(labels[train])) < 2:
        raise ValueError('logBB-only mapping lacks non-test BBBP/logBB pairs')
    mean, sd = (float(values[train].mean()), float(values[train].std(ddof=0)))
    if not sd > 0:
        raise ValueError('zero non-test logBB standard deviation')
    model = LogisticRegression(C=1000000.0, solver='lbfgs', max_iter=1000)
    model.fit(((values[train] - mean) / sd)[:, None], labels[train].astype(int))
    observed = np.isfinite(values[idx])
    p = model.predict_proba(((values[idx][observed] - mean) / sd)[:, None])[:, 1]
    return (observed, p, {'mapping_train_n': int(train.sum()), 'train_mean': mean, 'train_sd': sd, 'missing_test_logbb_n': int((~observed).sum()), 'missing_value_imputation': 'none'})

def metric_simple(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p)), 'brier': float(brier_score_loss(y, p)), 'ece': io_base.ece(y, p)}

def paired_distribution(y, left, right, left_name, right_name, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED + 17)
    rows = []
    valid = 0
    while valid < n_boot:
        take = rng.integers(0, len(y), len(y))
        yy = y[take]
        if len(np.unique(yy)) < 2:
            continue
        lm, rm = (metric_simple(yy, left[take]), metric_simple(yy, right[take]))
        for metric in ['auc', 'auprc', 'brier', 'ece']:
            rows.append({'bootstrap': valid + 1, 'contrast': f'{left_name} - {right_name}', 'metric': metric, 'difference': lm[metric] - rm[metric]})
        valid += 1
    return pd.DataFrame(rows)

def ablation_analysis():
    cons, test, idx, gold, _ = old.load_aligned()
    base_tensors = old.prepare_tensors(cons, idx)
    variants = ['full', 'no_logBB', 'no_BBBP', 'no_B3DB', 'no_continuous_channel', 'permuted_logBB', 'train_only_standardized_logBB']
    predictions, sigmas, notes, failures = ({}, {}, {}, [])
    for variant in variants:
        try:
            p, sigma, note = old.fit_variant(base_tensors, variant, idx)
            predictions[variant] = p
            sigmas[variant] = sigma
            notes[variant] = note
        except Exception as exc:
            failures.append({'analysis': 'main_ablation', 'variant': variant, 'error_type': type(exc).__name__, 'error': str(exc)})
    if 'full' not in predictions:
        raise RuntimeError('full HEM failed')
    pred = test[['consensus_idx', 'parent_ik']].copy()
    pred['gold_bbbp'] = gold
    for name, p in predictions.items():
        pred[f'p__{name}'] = p
        pred[f'sigma__{name}'] = sigmas[name]
    io_base.write_csv_both('ablation', 'ablation_predictions.csv', pred)
    metrics = pd.DataFrame([{'variant': name, **io_base.metrics(gold, p), 'definition': notes[name]} for name, p in predictions.items()])
    io_base.write_csv_both('ablation', 'ablation_metrics.csv', metrics)
    ci, diffs = io_base.bootstrap_metrics(gold, predictions, 'full', n_boot=N_BOOT, seed=SEED + 1)
    io_base.write_csv_both('ablation', 'ablation_metric_bootstrap_ci.csv', ci)
    io_base.write_csv_both('ablation', 'ablation_paired_bootstrap.csv', diffs)
    observed, p_logbb, mapping = logbb_mapping(cons, idx)
    complete = test.loc[observed, ['consensus_idx', 'parent_ik']].copy()
    complete['gold_bbbp'] = gold[observed]
    complete['observed_logbb'] = observed_logbb(cons)[idx][observed]
    complete['p_full'] = predictions['full'][observed]
    complete['p_logbb_only'] = p_logbb
    assert len(complete) == int(observed.sum()) == 49
    assert complete['p_logbb_only'].notna().all()
    io_base.write_csv_both('ablation', 'logbb_complete_case_predictions.csv', complete)
    io_base.write_csv_both('ablation', 'logbb_complete_case_molecule_list.csv', complete[['consensus_idx', 'parent_ik', 'observed_logbb', 'gold_bbbp']])
    complete_metrics = pd.DataFrame([{'model': 'full', 'n': len(complete), **io_base.metrics(gold[observed], predictions['full'][observed])}, {'model': 'logBB_only', 'n': len(complete), **io_base.metrics(gold[observed], p_logbb)}])
    io_base.write_csv_both('ablation', 'logbb_complete_case_metrics.csv', complete_metrics)
    dist = paired_distribution(gold[observed], p_logbb, predictions['full'][observed], 'logBB_only', 'full')
    io_base.write_csv_both('ablation', 'logbb_complete_case_bootstrap_distribution.csv', dist)
    point_left = metric_simple(gold[observed], p_logbb)
    point_right = metric_simple(gold[observed], predictions['full'][observed])
    summary_rows = []
    for metric, frame in dist.groupby('metric'):
        summary_rows.append({'contrast': 'logBB_only - full', 'metric': metric, 'estimate': point_left[metric] - point_right[metric], 'ci95_lo': float(frame['difference'].quantile(0.025)), 'ci95_hi': float(frame['difference'].quantile(0.975)), 'n_boot_valid': int(frame['bootstrap'].nunique()), 'n': len(complete)})
    complete_diff = pd.DataFrame(summary_rows)
    io_base.write_csv_both('ablation', 'logbb_complete_case_paired_bootstrap.csv', complete_diff)
    io_base.write_csv_both('ablation', 'ablation_failures.csv', pd.DataFrame(failures, columns=['analysis', 'variant', 'error_type', 'error']))
    report = {'status': 'complete_with_recorded_failures' if failures else 'complete', 'description': 'cross-channel consistency diagnostic on the same 291 molecules', 'main_analysis_n': 291, 'main_variants': variants, 'logbb_only_in_main_analysis': False, 'complete_case_sensitivity': {'status': 'exploratory', 'n': len(complete), 'comparison': 'full and logBB-only evaluated on the same observed-logBB molecules', **mapping, 'metrics': complete_metrics.to_dict('records'), 'paired_differences': complete_diff.to_dict('records')}, 'missing_logbb_prevalence_filling': False, 'failures': failures, 'no_silent_fallback': True, 'important_boundary': 'channel contribution and cross-channel consistency, not unseen-molecule structural generalization'}
    io_base.write_json_both('ablation', 'ablation_report.json', report)
    return report

def sync_and_verify():
    target = WORK / '01_代码_各阶段'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), target / Path(__file__).name)
    rows = []
    for name in ['calibration', 'ablation']:
        left, right = io_base.result_dirs(name)
        for p in sorted(left.iterdir()):
            if not p.is_file() or p.name == '02_05_sha256_manifest.csv':
                continue
            q = right / p.name
            rows.append({'group': name, 'filename': p.name, 'sha256_02': io_base.sha256(p), 'sha256_05': io_base.sha256(q) if q.exists() else None, 'identical': q.exists() and io_base.sha256(p) == io_base.sha256(q)})
    manifest = pd.DataFrame(rows)
    io_base.write_csv_both('calibration', '02_05_sha256_manifest.csv', manifest)
    assert manifest['identical'].all()

def main():
    old.calibration_analysis()
    ablation_analysis()
    sync_and_verify()
    print(json.dumps({'status': 'complete'}))
if __name__ == '__main__':
    main()

from __future__ import annotations
from project_paths import project_root, project_path
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
ROOT = project_root()
CODE = ROOT / '01_代码_各阶段'
sys.path.insert(0, str(CODE))
import stage_external_science_v13 as base
VERSION = 'v15'
SEEDS = list(range(42, 62))
N_BOOT = 2000
base.VERSION = VERSION
base.XGB_PARAMS = {**base.XGB_PARAMS, 'random_state': SEEDS[0]}

def repro_result_dirs(name: str):
    left = base.OUT02 / f'results_{name}_{VERSION}'
    right = ROOT / '05_复现工作区' / '02_结果' / f'results_{name}_{VERSION}'
    left.mkdir(parents=True, exist_ok=True)
    right.mkdir(parents=True, exist_ok=True)
    return (left, right)
base.result_dirs = repro_result_dirs

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def audit_conflicting_cids(assay: pd.DataFrame, unique: pd.DataFrame, report: dict):
    grouped = assay.groupby('PUBCHEM_CID', sort=False)
    duplicate_ids = [int(cid) for cid, frame in grouped if len(frame) > 1]
    conflict_ids = []
    details = []
    for cid in duplicate_ids:
        frame = grouped.get_group(cid)
        phenotypes = frame['Phenotype'].astype(str).tolist()
        values = frame['permeability_numeric'].astype(float).tolist()
        conflict = len(set(phenotypes)) > 1 or len(set(values)) > 1
        if conflict:
            conflict_ids.append(cid)
        details.append({'cid': cid, 'n_records': len(frame), 'conflict': conflict, 'phenotypes': phenotypes, 'permeability_values': values})
    out = unique.copy()
    out['ambiguous_duplicate'] = out['PUBCHEM_CID'].isin(conflict_ids)
    out['duplicate_measurements'] = ''
    for item in details:
        text = '|'.join((f'{p}:{v:g}' for p, v in zip(item['phenotypes'], item['permeability_values'])))
        out.loc[out['PUBCHEM_CID'].eq(item['cid']), 'duplicate_measurements'] = text
    before = int((~out['exclude_any_consensus_overlap']).sum())
    out['eligible_external'] = ~out['exclude_any_consensus_overlap'] & ~out['ambiguous_duplicate']
    after = int(out['eligible_external'].sum())
    cid457193 = out.loc[out['PUBCHEM_CID'].eq(457193)]
    assert conflict_ids == [457193], conflict_ids
    assert len(cid457193) == 1
    assert bool(cid457193.iloc[0]['exclude_any_consensus_overlap'])
    assert before == after == 318
    report = dict(report)
    report.update({'duplicate_cid_count': len(duplicate_ids), 'conflicting_duplicate_cid_count': len(conflict_ids), 'duplicate_cids': duplicate_ids, 'conflicting_duplicate_cids': conflict_ids, 'duplicate_details': details, 'conflict_rule': 'mark ambiguous and exclude before endpoint evaluation; never select the first row', 'eligible_before_ambiguous_exclusion': before, 'eligible_after_ambiguous_exclusion': after, 'cid_457193_already_removed_by_parent_overlap': True, 'cid_457193_effect_on_final_n': 0, 'eligible_external_molecules': after})
    return (out, report)

def metric_values(y, p):
    p = np.clip(np.asarray(p, float), 0, 1)
    return {'auc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p)), 'brier': float(brier_score_loss(y, p)), 'ece': base.ece(y, p)}

def block_bootstrap(y, seed_predictions, baseline='hard_B3DB', n_boot=N_BOOT, seed=20260818):
    rng = np.random.default_rng(seed)
    names = list(seed_predictions)
    n_seed, n_mol = next(iter(seed_predictions.values())).shape
    rows = []
    valid = 0
    while valid < n_boot:
        mi = rng.integers(0, n_mol, n_mol)
        si = rng.integers(0, n_seed, n_seed)
        yy = y[mi]
        if len(np.unique(yy)) < 2:
            continue
        vals = {}
        for name in names:
            pp = seed_predictions[name][si][:, mi].mean(axis=0)
            vals[name] = metric_values(yy, pp)
        for name in names:
            if name == baseline:
                continue
            for metric in ['auc', 'auprc', 'brier', 'ece']:
                rows.append({'bootstrap': valid + 1, 'contrast': f'{name} - {baseline}', 'metric': metric, 'difference': vals[name][metric] - vals[baseline][metric]})
        valid += 1
    dist = pd.DataFrame(rows)
    summary = dist.groupby(['contrast', 'metric'])['difference'].agg(estimate='mean', ci95_lo=lambda x: np.quantile(x, 0.025), ci95_hi=lambda x: np.quantile(x, 0.975), n_boot_valid='size').reset_index()
    summary['n_boot_valid'] //= max(1, len(names) - 1) * 0 + 1
    return (dist, summary)

def evaluate_external(unique: pd.DataFrame) -> dict:
    ext = unique.loc[unique['eligible_external']].reset_index(drop=True)
    train, labels, protocol = base.frozen_training_labels()
    Xtr = base.fingerprints(train['canon_smiles'])
    Xte = base.fingerprints(ext['ncats_parent_smiles'])
    target_audit = []
    names = list(labels)
    for left in names:
        for right in names:
            if names.index(right) <= names.index(left):
                continue
            a, b = (labels[left], labels[right])
            target_audit.append({'left': left, 'right': right, 'n': len(a), 'different_n': int(np.sum(~np.isclose(a, b))), 'max_abs_difference': float(np.max(np.abs(a - b))), 'correlation': float(np.corrcoef(a, b)[0, 1])})
    target_df = pd.DataFrame(target_audit)
    base.write_csv_both('external', 'training_target_identity_audit.csv', target_df)
    per_seed_rows, pred_rows, failures = ([], [], [])
    arrays = {name: [] for name in names}
    y = ext['y_pubchem'].to_numpy(int)
    for seed in SEEDS:
        for name, target in labels.items():
            try:
                params = {**base.XGB_PARAMS, 'random_state': seed}
                model = xgb.XGBRegressor(**params)
                model.fit(Xtr, target)
                p = np.clip(model.predict(Xte), 0, 1)
                arrays[name].append(p)
                per_seed_rows.append({'seed': seed, 'model': name, **base.metrics(y, p)})
                for i, value in enumerate(p):
                    pred_rows.append({'seed': seed, 'model': name, 'molecule_index': i, 'PUBCHEM_CID': int(ext.iloc[i]['PUBCHEM_CID']), 'ncats_parent_inchikey': ext.iloc[i]['ncats_parent_inchikey'], 'gold': int(y[i]), 'prediction': float(value)})
            except Exception as exc:
                failures.append({'seed': seed, 'method': name, 'error_type': type(exc).__name__, 'error': str(exc)})
    base.write_csv_both('external', 'external_failures.csv', pd.DataFrame(failures, columns=['seed', 'method', 'error_type', 'error']))
    if failures:
        raise RuntimeError(f'external model failures: {failures}')
    arrays = {name: np.vstack(values) for name, values in arrays.items()}
    mean_predictions = {name: values.mean(axis=0) for name, values in arrays.items()}
    base.write_csv_both('external', 'external_predictions_by_seed.csv', pd.DataFrame(pred_rows))
    mean_df = ext[['PUBCHEM_CID', 'Title', 'Phenotype', 'Permeability', 'permeability_numeric', 'exact_10', 'y_pubchem', 'y_10_low', 'y_10_moderate', 'ncats_parent_smiles', 'ncats_parent_inchikey', 'ncats_connectivity_key']].copy()
    for name, p in mean_predictions.items():
        mean_df[f'p_mean__{name}'] = p
    base.write_csv_both('external', 'external_predictions.csv', mean_df)
    per_seed = pd.DataFrame(per_seed_rows)
    base.write_csv_both('external', 'external_metrics_per_seed.csv', per_seed)
    summary = per_seed.groupby('model')[['auc', 'auprc', 'brier', 'ece', 'calibration_intercept', 'calibration_slope']].agg(['mean', 'std', 'min', 'max'])
    summary.columns = [f'{a}_{b}' for a, b in summary.columns]
    summary = summary.reset_index()
    base.write_csv_both('external', 'external_seed_summary.csv', summary)
    metric_df = pd.DataFrame([{'label_rule': 'pubchem_phenotype', 'model': name, **base.metrics(y, p)} for name, p in mean_predictions.items()])
    base.write_csv_both('external', 'external_metrics.csv', metric_df)
    ci, diffs = base.bootstrap_metrics(y, mean_predictions, 'hard_B3DB', n_boot=N_BOOT)
    ci.insert(0, 'label_rule', 'pubchem_phenotype')
    diffs.insert(0, 'label_rule', 'pubchem_phenotype')
    base.write_csv_both('external', 'external_metric_bootstrap_ci.csv', ci)
    base.write_csv_both('external', 'external_paired_bootstrap.csv', diffs)
    block_dist, block_summary = block_bootstrap(y, arrays)
    base.write_csv_both('external', 'external_molecule_seed_block_bootstrap_distribution.csv', block_dist)
    base.write_csv_both('external', 'external_molecule_seed_block_bootstrap.csv', block_summary)
    maj_weight = target_df[(target_df.left == 'channel_majority') & (target_df.right == 'channel_evidence_count_weighted') | (target_df.right == 'channel_majority') & (target_df.left == 'channel_evidence_count_weighted')].iloc[0]
    pmax = float(np.max(np.abs(mean_predictions['channel_majority'] - mean_predictions['channel_evidence_count_weighted'])))
    reason = 'training targets are exactly identical under the current common training mask' if int(maj_weight['different_n']) == 0 else 'training targets differ; prediction equivalence is an empirical fitted-model result'
    identity = {'training_target_different_n': int(maj_weight['different_n']), 'training_target_max_abs_difference': float(maj_weight['max_abs_difference']), 'training_target_correlation': float(maj_weight['correlation']), 'mean_prediction_max_abs_difference': pmax, 'mean_prediction_mean_abs_difference': float(np.mean(np.abs(mean_predictions['channel_majority'] - mean_predictions['channel_evidence_count_weighted']))), 'predictions_exactly_identical': bool(pmax == 0), 'reason': reason}
    base.write_json_both('external', 'majority_weighted_identity_audit.json', identity)
    protocol.update({'prespecified_seeds': SEEDS, 'seed_count': len(SEEDS), 'external_data_used_for_training_or_tuning': False, 'main_prediction': 'mean across 20 prespecified XGBoost seeds'})
    report = {'status': 'complete', 'endpoint_interpretation': 'independent PAMPA-BBB experimental endpoint after parent-connectivity exclusion; not an overall BBB gold standard', 'protocol': protocol, 'external_n': len(ext), 'main_label_counts': mean_df['y_pubchem'].value_counts().sort_index().to_dict(), 'models': names, 'majority_weighted_identity': identity, 'no_silent_fallback': True, 'metrics_on_mean_predictions': metric_df.to_dict('records'), 'unavailable_real_baselines': [{'method': m, 'status': 'N/A', 'reason': 'real molecule-by-source label matrix is not identifiable'} for m in ['Dawid-Skene', 'Snorkel', 'MAP latent class']]}
    base.write_json_both('external', 'external_report.json', report)
    return report

def baseline_manifest():
    roots = [ROOT / '03_指南与文档' / 'BBB_AI_BIB中文论文外部证据增强_v1.4.docx', ROOT / '03_指南与文档' / 'BBB_AI_BIB中文论文正文结构清理_v1.2.docx', CODE / 'stage_external_science_v13.py', CODE / 'stage_calibration_ablation_v13.py', CODE / 'build_manuscript_v13.py', CODE / 'reorder_references_v13.py']
    for folder in [ROOT / '02_结果' / 'results_external_v13', ROOT / '02_结果' / 'results_calibration_v13', ROOT / '02_结果' / 'results_ablation_v13', ROOT / '03_指南与文档' / '论文素材' / 'figures_v13']:
        roots.extend(sorted((p for p in folder.iterdir() if p.is_file())))
    rows = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in roots]
    obj = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'files': rows, 'historical_results_modified': False}
    base.write_json_both('external', 'baseline_manifest.json', obj)

def verify_pairs_v15():
    rows = []
    for name in ['external', 'source_matrix']:
        left, right = repro_result_dirs(name)
        left_files = {p.name: p for p in left.iterdir() if p.is_file() and p.name != '02_05_sha256_manifest.csv'}
        right_files = {p.name: p for p in right.iterdir() if p.is_file() and p.name != '02_05_sha256_manifest.csv'}
        for filename in sorted(set(left_files) | set(right_files)):
            a, b = (left_files.get(filename), right_files.get(filename))
            rows.append({'result_group': name, 'filename': filename, 'in_02': a is not None, 'in_05': b is not None, 'sha256_02': sha256(a) if a else None, 'sha256_05': sha256(b) if b else None, 'identical': bool(a and b and (sha256(a) == sha256(b)))})
    audit = pd.DataFrame(rows)
    base.write_csv_both('external', '02_05_sha256_manifest.csv', audit)
    assert audit['identical'].all()
    return audit

def main():
    baseline_manifest()
    raw_path = base.RAW / 'NCATS_PAMPA_BBB_AID1845228.csv'
    assay, assay_report = base.clean_assay(raw_path)
    props_path = base.RAW / 'NCATS_PAMPA_BBB_AID1845228_properties.csv'
    props, prop_report = base.fetch_properties(sorted(assay['PUBCHEM_CID'].unique()), props_path)
    unique0, dedup0 = base.deduplicate_ncarts(assay, props)
    unique, dedup = audit_conflicting_cids(assay, unique0, dedup0)
    base.write_bytes_both('external', raw_path.name, raw_path.read_bytes())
    base.write_bytes_both('external', props_path.name, props_path.read_bytes())
    base.write_csv_both('external', 'ncats_unique_molecule_audit.csv', unique)
    base.write_json_both('external', 'ncats_qualification_report.json', {'status': 'qualified_after_parent_deduplication', 'assay_parse': assay_report, 'property_retrieval': prop_report, 'deduplication': dedup, 'source_article': {'citation': 'Kato et al., Frontiers in Pharmacology (2023)', 'doi': '10.3389/fphar.2023.1291246'}, 'database_record': {'name': 'PubChem BioAssay AID 1845228', 'aid': 1845228}, 'claim_boundary': 'independent PAMPA-BBB experimental endpoint; not an overall BBB gold standard'})
    base.source_matrix_audit()
    evaluate_external(unique)
    target = base.OUT05 / '01_代码_各阶段'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), target / Path(__file__).name)
    verify_pairs_v15()
    print(json.dumps({'status': 'complete', 'external_n': int(unique['eligible_external'].sum()), 'seeds': SEEDS}, ensure_ascii=False))
if __name__ == '__main__':
    main()

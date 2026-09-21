from __future__ import annotations
from project_paths import project_root, project_path
import argparse
import hashlib
import io
import json
import math
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
ROOT = project_root()
CODE = ROOT / '01_代码_各阶段'
OUT02 = ROOT / '02_结果'
OUT05 = ROOT / '05_复现工作区'
RAW = ROOT / '00_数据_原始'
VERSION = 'v13'
SEED = 42
N_BOOT = 2000
RADIUS = 2
N_BITS = 2048
XGB_PARAMS = dict(max_depth=5, learning_rate=0.03, n_estimators=300, subsample=0.8, colsample_bytree=0.8, random_state=SEED)
ASSAY_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1845228/CSV'
PROP_TEMPLATE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/CanonicalSMILES,IsomericSMILES,InChIKey,Title/CSV'
RDLogger.DisableLog('rdApp.*')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)

def result_dirs(name: str) -> tuple[Path, Path]:
    a = OUT02 / f'results_{name}_{VERSION}'
    b = OUT05 / f'results_{name}_{VERSION}'
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    return (a, b)

def write_bytes_both(name: str, filename: str, data: bytes) -> None:
    for d in result_dirs(name):
        (d / filename).write_bytes(data)

def write_csv_both(name: str, filename: str, df: pd.DataFrame) -> None:
    payload = df.to_csv(index=False).encode('utf-8-sig')
    write_bytes_both(name, filename, payload)

def write_json_both(name: str, filename: str, obj) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2, default=json_default, allow_nan=False).encode('utf-8')
    write_bytes_both(name, filename, payload)

def download(url: str, path: Path, *, force: bool=False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if force or not path.exists():
        req = urllib.request.Request(url, headers={'User-Agent': 'BBB-HEM-audit/1.3'})
        with urllib.request.urlopen(req, timeout=120) as response:
            path.write_bytes(response.read())
        downloaded = True
    return {'url': url, 'path': str(path), 'downloaded_now': downloaded, 'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'bytes': path.stat().st_size, 'sha256': sha256(path)}

def clean_assay(raw_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_path, dtype=str)
    data = raw[pd.to_numeric(raw['PUBCHEM_CID'], errors='coerce').notna()].copy()
    data['PUBCHEM_CID'] = pd.to_numeric(data['PUBCHEM_CID']).astype(int)
    data['Phenotype'] = data['Phenotype'].str.strip()
    assert len(data) == 438, f'expected 438 assay rows, found {len(data)}'
    assert data['PUBCHEM_CID'].nunique() == 437
    counts = data['Phenotype'].value_counts().to_dict()
    assert counts == {'High': 202, 'Low': 171, 'Moderate': 65}, counts
    descr = ' '.join(raw.astype(str).fillna('').iloc[:2].values.ravel())
    assert 'stirring double-sink PAMPA-BBB' in descr
    assert '10^-6cm/s' in descr.replace(' ', '')
    data['permeability_numeric'] = pd.to_numeric(data['Permeability'].str.replace('>', '', regex=False).str.strip(), errors='coerce')
    data['exact_10'] = data['permeability_numeric'].eq(10)
    data['y_pubchem'] = data['Phenotype'].isin(['Moderate', 'High']).astype(int)
    data['y_10_low'] = np.where(data['exact_10'], 0, data['y_pubchem'])
    data['y_10_moderate'] = np.where(data['exact_10'], 1, data['y_pubchem'])
    report = {'raw_csv_rows_including_two_metadata_rows': len(raw), 'assay_rows': len(data), 'unique_cids': data['PUBCHEM_CID'].nunique(), 'phenotype_counts': counts, 'exact_permeability_10_n': int(data['exact_10'].sum()), 'endpoint': 'stirring double-sink PAMPA-BBB', 'units': '10^-6 cm/s', 'main_binary_rule': 'Low=0; Moderate+High=1', 'positive_direction': 'higher permeability / permeable', 'scope': 'independent PAMPA-BBB experimental endpoint; not an overall BBB gold standard'}
    return (data, report)

def fetch_properties(cids: list[int], cache: Path, force: bool=False) -> tuple[pd.DataFrame, dict]:
    if cache.exists() and (not force):
        return (pd.read_csv(cache), {'cache_used': True, 'sha256': sha256(cache), 'rows': len(pd.read_csv(cache))})
    pieces = []
    calls = []
    for start in range(0, len(cids), 80):
        batch = cids[start:start + 80]
        url = PROP_TEMPLATE.format(cids=','.join(map(str, batch)))
        req = urllib.request.Request(url, headers={'User-Agent': 'BBB-HEM-audit/1.3'})
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = response.read()
        frame = pd.read_csv(io.BytesIO(payload))
        pieces.append(frame)
        calls.append({'n_requested': len(batch), 'n_returned': len(frame), 'url': url})
        time.sleep(0.15)
    props = pd.concat(pieces, ignore_index=True)
    props.to_csv(cache, index=False, encoding='utf-8-sig')
    assert props['CID'].nunique() == len(set(cids)), 'PubChem property retrieval incomplete'
    return (props, {'cache_used': False, 'sha256': sha256(cache), 'rows': len(props), 'calls': calls})

def normalize_smiles(smiles: str) -> dict:
    out = {'canonical_smiles': None, 'isomeric_smiles': None, 'full_inchikey': None, 'parent_smiles': None, 'parent_inchikey': None, 'connectivity_key': None, 'had_multiple_fragments': False, 'normalization_ok': False}
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return out
        out['had_multiple_fragments'] = len(Chem.GetMolFrags(mol)) > 1
        out['canonical_smiles'] = Chem.MolToSmiles(mol, isomericSmiles=False)
        out['isomeric_smiles'] = Chem.MolToSmiles(mol, isomericSmiles=True)
        out['full_inchikey'] = MolToInchiKey(mol)
        parent = rdMolStandardize.FragmentParent(mol)
        try:
            parent = rdMolStandardize.ChargeParent(parent)
        except Exception:
            pass
        out['parent_smiles'] = Chem.MolToSmiles(parent, isomericSmiles=False)
        out['parent_inchikey'] = MolToInchiKey(parent)
        out['connectivity_key'] = out['parent_inchikey'].split('-')[0]
        out['normalization_ok'] = True
    except Exception:
        pass
    return out

def normalize_frame(smiles: pd.Series, prefix: str) -> pd.DataFrame:
    recs = [normalize_smiles(s) for s in smiles]
    return pd.DataFrame(recs).add_prefix(prefix)

def deduplicate_ncarts(assay: pd.DataFrame, props: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    assay = assay.merge(props, left_on='PUBCHEM_CID', right_on='CID', how='left', validate='many_to_one')
    assay = pd.concat([assay.reset_index(drop=True), normalize_frame(assay['SMILES'], 'ncats_')], axis=1)
    assert assay['ncats_normalization_ok'].all(), 'NCATS contains unnormalizable structures'
    cons_path = OUT05 / 'results_stage5' / 'data' / 'consensus_dataset.csv'
    cons = pd.read_csv(cons_path, dtype=str).fillna('')
    cons_norm = normalize_frame(cons['canon_smiles'], 'cons_')
    cons2 = pd.concat([cons.reset_index(drop=True), cons_norm], axis=1)
    bbbp = pd.read_csv(RAW / 'BBBP.csv', dtype=str).fillna('')
    bbbp_norm = normalize_frame(bbbp['smiles'], 'bbbp_')
    bbbp2 = pd.concat([bbbp.reset_index(drop=True), bbbp_norm], axis=1)
    b3db = pd.read_csv(RAW / 'B3DB_数据' / 'B3DB_classification.tsv', sep='\t', dtype=str).fillna('')
    b3db_norm = normalize_frame(b3db['SMILES'], 'b3db_')
    b3db2 = pd.concat([b3db.reset_index(drop=True), b3db_norm], axis=1)
    cid_set = set(pd.to_numeric(b3db['CID'], errors='coerce').dropna().astype(int))
    canonical_set = set(cons2['cons_canonical_smiles'].dropna())
    full_set = set(cons2['cons_full_inchikey'].dropna())
    parent_full_set = set(cons2['cons_parent_inchikey'].dropna())
    connectivity_set = set(cons2['cons_connectivity_key'].dropna())
    bbbp_connectivity = set(bbbp2['bbbp_connectivity_key'].dropna())
    b3db_connectivity = set(b3db2['b3db_connectivity_key'].dropna())
    assay['overlap_cid_b3db'] = assay['PUBCHEM_CID'].isin(cid_set)
    assay['overlap_canonical_consensus'] = assay['ncats_canonical_smiles'].isin(canonical_set)
    assay['overlap_full_inchikey_consensus'] = assay['ncats_full_inchikey'].isin(full_set)
    assay['overlap_parent_full_inchikey_consensus'] = assay['ncats_parent_inchikey'].isin(parent_full_set)
    assay['overlap_parent_connectivity_consensus'] = assay['ncats_connectivity_key'].isin(connectivity_set)
    assay['overlap_parent_connectivity_bbbp'] = assay['ncats_connectivity_key'].isin(bbbp_connectivity)
    assay['overlap_parent_connectivity_b3db'] = assay['ncats_connectivity_key'].isin(b3db_connectivity)
    assay['salt_parent_only_overlap'] = assay['overlap_parent_connectivity_consensus'] & ~assay['overlap_full_inchikey_consensus'] & assay['ncats_had_multiple_fragments']
    assay['stereo_or_protonation_only_overlap'] = assay['overlap_parent_connectivity_consensus'] & ~assay['overlap_full_inchikey_consensus'] & ~assay['ncats_had_multiple_fragments']
    assay['exclude_any_consensus_overlap'] = assay['overlap_parent_connectivity_consensus']
    agg = {'Phenotype': 'first', 'Permeability': lambda x: '|'.join(map(str, x)), 'permeability_numeric': 'mean', 'exact_10': 'max', 'y_pubchem': 'first', 'y_10_low': 'first', 'y_10_moderate': 'first', 'SMILES': 'first', 'Title': 'first', 'ncats_canonical_smiles': 'first', 'ncats_isomeric_smiles': 'first', 'ncats_full_inchikey': 'first', 'ncats_parent_smiles': 'first', 'ncats_parent_inchikey': 'first', 'ncats_connectivity_key': 'first', 'ncats_had_multiple_fragments': 'max', 'overlap_cid_b3db': 'max', 'overlap_canonical_consensus': 'max', 'overlap_full_inchikey_consensus': 'max', 'overlap_parent_full_inchikey_consensus': 'max', 'overlap_parent_connectivity_consensus': 'max', 'overlap_parent_connectivity_bbbp': 'max', 'overlap_parent_connectivity_b3db': 'max', 'salt_parent_only_overlap': 'max', 'stereo_or_protonation_only_overlap': 'max', 'exclude_any_consensus_overlap': 'max'}
    unique = assay.groupby('PUBCHEM_CID', as_index=False).agg(agg)
    eligible = unique.loc[~unique['exclude_any_consensus_overlap']].copy()
    assert eligible['PUBCHEM_CID'].is_unique
    report = {'assay_rows': len(assay), 'unique_cids': len(unique), 'cid_overlap_b3db_unique': int(unique['overlap_cid_b3db'].sum()), 'canonical_smiles_overlap_consensus': int(unique['overlap_canonical_consensus'].sum()), 'full_inchikey_overlap_consensus': int(unique['overlap_full_inchikey_consensus'].sum()), 'parent_full_inchikey_overlap_consensus': int(unique['overlap_parent_full_inchikey_consensus'].sum()), 'parent_connectivity_overlap_consensus': int(unique['overlap_parent_connectivity_consensus'].sum()), 'parent_connectivity_overlap_bbbp': int(unique['overlap_parent_connectivity_bbbp'].sum()), 'parent_connectivity_overlap_b3db': int(unique['overlap_parent_connectivity_b3db'].sum()), 'salt_parent_only_overlap': int(unique['salt_parent_only_overlap'].sum()), 'stereo_or_protonation_only_overlap': int(unique['stereo_or_protonation_only_overlap'].sum()), 'eligible_external_molecules': len(eligible), 'deduplication_rule': 'exclude any NCATS molecule sharing the RDKit-standardized parent connectivity InChIKey with the consensus universe', 'consensus_sha256': sha256(cons_path), 'bbbp_rows': len(bbbp), 'b3db_rows': len(b3db)}
    return (unique, report)

def fingerprints(smiles: pd.Series) -> np.ndarray:
    gen = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS, includeChirality=True)
    out = np.zeros((len(smiles), N_BITS), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            raise ValueError(f'invalid SMILES at row {i}')
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), out[i])
    return out

def label_count(raw: str) -> int:
    return sum((z.strip() in {'0', '1', '0.0', '1.0'} for z in str(raw).split(';')))

def frozen_training_labels() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    cons = pd.read_csv(OUT05 / 'results_stage5' / 'data' / 'consensus_dataset.csv', dtype=str).fillna('')
    train = cons[(cons['split'] == 'train') & pd.to_numeric(cons['in_reference_out']).eq(0)].reset_index(drop=True)
    labs = pd.read_csv(OUT05 / 'results_stage7' / 'data' / 'labels_abc.csv')
    assert len(train) == 1234 and np.array_equal(train['parent_ik'], labs['parent_ik'])
    hard = pd.to_numeric(train['b3db_label'], errors='coerce').to_numpy(float)
    mask = np.isfinite(hard)
    assert mask.sum() == 1157
    bbbp = pd.to_numeric(train['bbbp_label'], errors='coerce').to_numpy(float)
    b3db = hard.copy()
    stack = np.c_[b3db, bbbp]
    majority = np.nanmean(stack, axis=1)
    b3n = train['b3db_raw_labels'].map(label_count).to_numpy(float)
    bbn = train['bbbp_raw_labels'].map(label_count).to_numpy(float)
    b3n = np.where(np.isfinite(b3db), np.maximum(b3n, 1), 0)
    bbn = np.where(np.isfinite(bbbp), np.maximum(bbn, 1), 0)
    weighted = np.divide(np.nan_to_num(b3db) * b3n + np.nan_to_num(bbbp) * bbn, b3n + bbn, out=np.full(len(train), np.nan), where=b3n + bbn > 0)
    labels = {'hard_B3DB': hard, 'channel_majority': majority, 'channel_evidence_count_weighted': weighted, 'HEM_probability': pd.to_numeric(labs['label_C_sigmoid_mu']).to_numpy(float)}
    for y in labels.values():
        assert np.isfinite(y[mask]).all()
    protocol = {'training_rows': len(train), 'common_training_molecules': int(mask.sum()), 'common_mask': 'finite B3DB hard label, frozen stage7 definition', 'majority_definition': 'unweighted mean of available B3DB and BBBP binary channels', 'weighted_definition': 'channel labels weighted by observable within-channel label counts; not an estimated reliability model', 'morgan_radius': RADIUS, 'morgan_bits': N_BITS, 'chirality': True, 'xgboost': XGB_PARAMS, 'random_seed': SEED, 'tuning': 'none; frozen stage7 parameters'}
    return (train.loc[mask].reset_index(drop=True), {k: v[mask] for k, v in labels.items()}, protocol)

def ece(y: np.ndarray, p: np.ndarray, n_bins: int=10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    return float(sum((np.mean(bins == k) * abs(np.mean(p[bins == k]) - np.mean(y[bins == k])) for k in range(n_bins) if np.any(bins == k))))

def cal_parameters(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-06, 1 - 1e-06)
    logit = np.log(p / (1 - p))[:, None]
    model = LogisticRegression(C=1000000.0, solver='lbfgs', max_iter=1000).fit(logit, y)
    return (float(model.intercept_[0]), float(model.coef_[0, 0]))

def metrics(y, p) -> dict:
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 0, 1)
    intercept, slope = cal_parameters(y, p)
    return {'n': len(y), 'positive_n': int(y.sum()), 'positive_rate': float(y.mean()), 'auc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p)), 'brier': float(brier_score_loss(y, p)), 'ece': ece(y, p), 'calibration_intercept': intercept, 'calibration_slope': slope}

def bootstrap_metrics(y, predictions: dict[str, np.ndarray], baseline: str, n_boot=N_BOOT, seed: int=20260818) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.asarray(y, int)
    rng = np.random.default_rng(seed)
    names = list(predictions)
    metric_names = ['auc', 'auprc', 'brier', 'ece']
    store = {name: {m: [] for m in metric_names} for name in names}

    def boot_values(yy, pp):
        pp = np.clip(pp, 0, 1)
        return {'auc': float(roc_auc_score(yy, pp)), 'auprc': float(average_precision_score(yy, pp)), 'brier': float(brier_score_loss(yy, pp)), 'ece': ece(yy, pp)}
    valid = 0
    while valid < n_boot:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        for name in names:
            mm = boot_values(y[idx], predictions[name][idx])
            for metric in metric_names:
                store[name][metric].append(mm[metric])
        valid += 1
    ci_rows, diff_rows = ([], [])
    point = {name: metrics(y, p) for name, p in predictions.items()}
    for name in names:
        for metric in metric_names:
            lo, hi = np.percentile(store[name][metric], [2.5, 97.5])
            ci_rows.append({'model': name, 'metric': metric, 'estimate': point[name][metric], 'ci95_lo': lo, 'ci95_hi': hi, 'n_boot_valid': valid})
        if name == baseline:
            continue
        for metric in metric_names:
            vals = np.asarray(store[name][metric]) - np.asarray(store[baseline][metric])
            lo, hi = np.percentile(vals, [2.5, 97.5])
            diff_rows.append({'contrast': f'{name} - {baseline}', 'metric': metric, 'estimate': point[name][metric] - point[baseline][metric], 'ci95_lo': lo, 'ci95_hi': hi, 'n_boot_valid': valid})
    return (pd.DataFrame(ci_rows), pd.DataFrame(diff_rows))

def evaluate_external(unique: pd.DataFrame) -> dict:
    ext = unique.loc[~unique['exclude_any_consensus_overlap']].reset_index(drop=True)
    train, labels, protocol = frozen_training_labels()
    Xtr = fingerprints(train['canon_smiles'])
    Xte = fingerprints(ext['ncats_parent_smiles'])
    predictions = {}
    failures = []
    for name, y in labels.items():
        try:
            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(Xtr, y)
            predictions[name] = np.clip(model.predict(Xte), 0, 1)
        except Exception as exc:
            failures.append({'method': name, 'error_type': type(exc).__name__, 'error': str(exc)})
    if failures:
        write_csv_both('external', 'external_failures.csv', pd.DataFrame(failures))
        raise RuntimeError(f'external model failures: {failures}')
    pred = ext[['PUBCHEM_CID', 'Title', 'Phenotype', 'Permeability', 'permeability_numeric', 'exact_10', 'y_pubchem', 'y_10_low', 'y_10_moderate', 'ncats_parent_smiles', 'ncats_parent_inchikey', 'ncats_connectivity_key']].copy()
    for name, p in predictions.items():
        pred[f'p__{name}'] = p
    write_csv_both('external', 'external_predictions.csv', pred)
    all_metrics = []
    all_ci = []
    all_diffs = []
    rules = {'pubchem_phenotype': pred['y_pubchem'].to_numpy(int), 'exact_10_as_low': pred['y_10_low'].to_numpy(int), 'exact_10_as_moderate': pred['y_10_moderate'].to_numpy(int)}
    for rule, y in rules.items():
        for name, p in predictions.items():
            all_metrics.append({'label_rule': rule, 'model': name, **metrics(y, p)})
        ci, diffs = bootstrap_metrics(y, predictions, 'hard_B3DB')
        ci.insert(0, 'label_rule', rule)
        diffs.insert(0, 'label_rule', rule)
        all_ci.append(ci)
        all_diffs.append(diffs)
    metric_df = pd.DataFrame(all_metrics)
    ci_df = pd.concat(all_ci, ignore_index=True)
    diff_df = pd.concat(all_diffs, ignore_index=True)
    write_csv_both('external', 'external_metrics.csv', metric_df)
    write_csv_both('external', 'external_metric_bootstrap_ci.csv', ci_df)
    write_csv_both('external', 'external_paired_bootstrap.csv', diff_df)
    report = {'status': 'complete', 'endpoint_interpretation': 'independent PAMPA-BBB experimental endpoint after parent-connectivity exclusion; not an overall BBB gold standard', 'protocol': protocol, 'external_n': len(ext), 'main_label_counts': pred['y_pubchem'].value_counts().sort_index().to_dict(), 'exact_10_n': int(pred['exact_10'].sum()), 'models': list(predictions), 'unavailable_real_baselines': [{'method': m, 'status': 'N/A', 'reason': 'real molecule-by-source label matrix is not identifiable'} for m in ['Dawid-Skene', 'Snorkel', 'MAP latent class', 'reliability-estimated source weighting']], 'logbb_only': {'status': 'N/A', 'reason': 'a common 1,157-molecule training lane with observed continuous logBB is unavailable; restricting all models to logBB coverage would change the frozen training population'}, 'no_silent_fallback': True, 'metrics': metric_df.to_dict('records')}
    write_json_both('external', 'external_report.json', report)
    return report

def source_matrix_audit() -> dict:
    cons_path = OUT05 / 'results_stage5' / 'data' / 'consensus_dataset.csv'
    cons = pd.read_csv(cons_path, dtype=str).fillna('')
    train = cons[(cons['split'] == 'train') & pd.to_numeric(cons['in_reference_out']).eq(0)].reset_index(drop=True)
    records = []
    for i, row in train.iterrows():
        refs = [z for z in row['reference_list'].split('|') if z]
        labels = [z for z in row['b3db_raw_labels'].split(';') if z in {'0', '1'}]
        records.append({'row': i, 'parent_ik': row['parent_ik'], 'b3db_aggregate_label': row['b3db_label'], 'bbbp_aggregate_label': row['bbbp_label'], 'reference_codes': '|'.join(refs), 'n_reference_codes': len(refs), 'raw_binary_labels': ';'.join(labels), 'n_raw_binary_labels': len(labels), 'counts_match': len(refs) == len(labels), 'mappable_multi_source': len(refs) >= 2 and len(refs) == len(labels)})
    audit = pd.DataFrame(records)
    write_csv_both('source_matrix', 'source_matrix_row_audit.csv', audit)
    refs = audit['reference_codes'].str.split('|').explode()
    refs = refs[refs.ne('')].value_counts().rename_axis('source_code').reset_index(name='molecule_mentions')
    write_csv_both('source_matrix', 'source_counts.csv', refs)
    n_mappable = int(audit['mappable_multi_source'].sum())
    qualifies = n_mappable >= 100
    report = {'status': 'complete', 'n_training_molecules': len(train), 'molecules_with_two_or_more_reference_codes': int(audit['n_reference_codes'].ge(2).sum()), 'molecules_with_reliably_mappable_multi_source_labels': n_mappable, 'criteria': {'at_least_three_distinguishable_sources': False, 'at_least_100_molecules_with_two_or_more_source_labels': qualifies, 'adequate_positive_negative_counts_per_source': False, 'auditable_dependence_and_missingness': False}, 'identifiable': False, 'decision': 'Dawid-Skene, Snorkel, and MAP latent class are N/A on real data', 'reason': 'B3DB supplies an aggregate label followed by reference codes; the codes cannot be treated as independent annotators without per-reference labels', 'prohibition': 'reference-code columns were not synthesized and no real-data latent-source model was run'}
    write_json_both('source_matrix', 'source_matrix_qualification.json', report)
    return report

def baseline_manifest() -> dict:
    files = [ROOT / '03_指南与文档' / 'BBB_AI_BIB中文论文正文结构清理_v1.2.docx', CODE / 'stage_f2_f7_final_v10.py', OUT02 / 'results_f4_v10' / 'f4_report.json', OUT02 / 'results_f5_v10' / 'f5_report.json', OUT02 / 'results_f6_v10' / 'f6_posterior_analysis.json', OUT02 / 'results_f7_v10' / 'f7_case_audit.json']
    rows = [{'path': str(p), 'exists': p.exists(), 'bytes': p.stat().st_size if p.exists() else None, 'sha256': sha256(p) if p.exists() else None} for p in files]
    report = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'files': rows, 'old_results_modified': False}
    write_json_both('external', 'baseline_manifest.json', report)
    return report

def sync_code() -> None:
    target = OUT05 / '01_代码_各阶段'
    target.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve()
    shutil.copy2(src, target / src.name)

def verify_pairs() -> pd.DataFrame:
    rows = []
    for name in ['external', 'source_matrix']:
        a, b = result_dirs(name)
        af = {p.name: p for p in a.iterdir() if p.is_file()}
        bf = {p.name: p for p in b.iterdir() if p.is_file()}
        for filename in sorted(set(af) | set(bf)):
            rows.append({'result_group': name, 'filename': filename, 'in_02': filename in af, 'in_05': filename in bf, 'sha256_02': sha256(af[filename]) if filename in af else None, 'sha256_05': sha256(bf[filename]) if filename in bf else None, 'identical': filename in af and filename in bf and (sha256(af[filename]) == sha256(bf[filename]))})
    out = pd.DataFrame(rows)
    write_csv_both('external', '02_05_sha256_manifest.csv', out)
    assert out['identical'].all()
    return out

def main(force_download: bool=False) -> None:
    baseline_manifest()
    raw_path = RAW / 'NCATS_PAMPA_BBB_AID1845228.csv'
    metadata = download(ASSAY_URL, raw_path, force=force_download)
    assay, assay_report = clean_assay(raw_path)
    props_path = RAW / 'NCATS_PAMPA_BBB_AID1845228_properties.csv'
    props, prop_report = fetch_properties(sorted(assay['PUBCHEM_CID'].unique()), props_path, force=force_download)
    unique, dedup_report = deduplicate_ncarts(assay, props)
    write_bytes_both('external', raw_path.name, raw_path.read_bytes())
    write_bytes_both('external', props_path.name, props_path.read_bytes())
    write_csv_both('external', 'ncats_unique_molecule_audit.csv', unique)
    write_json_both('external', 'ncats_data_dictionary.json', {'PUBCHEM_CID': 'PubChem compound identifier', 'Phenotype': 'PubChem Low/Moderate/High category', 'Permeability': 'reported PAMPA-BBB permeability text', 'permeability_numeric': 'numeric part of Permeability', 'exact_10': 'numeric permeability equals 10', 'y_pubchem': 'Low=0; Moderate/High=1', 'ncats_parent_smiles': 'RDKit fragment/charge-parent non-isomeric SMILES', 'ncats_connectivity_key': 'first block of parent InChIKey used for exclusion', 'exclude_any_consensus_overlap': 'parent connectivity occurs anywhere in the consensus universe'})
    write_json_both('external', 'ncats_qualification_report.json', {'status': 'qualified_after_parent_deduplication', 'raw_file': metadata, 'assay_parse': assay_report, 'property_retrieval': prop_report, 'deduplication': dedup_report, 'source_article': {'citation': 'Kato et al., Frontiers in Pharmacology (2023)', 'doi': '10.3389/fphar.2023.1291246', 'pubchem_aid': 1845228}, 'claim_boundary': 'independent PAMPA-BBB experimental endpoint; passive diffusion assay and not an overall BBB gold standard'})
    source_matrix_audit()
    evaluate_external(unique)
    sync_code()
    verify_pairs()
    print(json.dumps({'status': 'complete', 'eligible_external_molecules': dedup_report['eligible_external_molecules']}, ensure_ascii=False))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-download', action='store_true')
    args = parser.parse_args()
    main(force_download=args.force_download)

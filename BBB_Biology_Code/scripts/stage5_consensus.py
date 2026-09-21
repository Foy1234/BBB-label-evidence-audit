from __future__ import annotations
from project_paths import project_root, project_path
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog('rdApp.*')
DATA_DIR = project_path('00_数据_原始')
B3DB_DIR = DATA_DIR / 'B3DB_数据'
OUTPUT_DIR = project_path('05_复现工作区', 'results_stage5')
SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15
REF_OUT_REFS = {'R27'}
SREMOVER = SaltRemover.SaltRemover()

def parent_inchikey(smiles):
    if not isinstance(smiles, str):
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        m = SREMOVER.StripMol(m)
        Chem.RemoveStereochemistry(m)
        ik = MolToInchiKey(m)
        return ik.split('-')[0] if ik else None
    except Exception:
        return None

def canon_smiles(smiles):
    if not isinstance(smiles, str):
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        m = SREMOVER.StripMol(m)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    except Exception:
        return None

def load_b3db(fn):
    df = pd.read_csv(B3DB_DIR / fn, sep='\t', dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def parse_refs(ref_field):
    if not isinstance(ref_field, str) or not ref_field.strip():
        return []
    codes = [x.strip() for x in ref_field.split('|')]
    return sorted({x for x in codes if x})

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'data').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    cls = load_b3db('B3DB_classification.tsv')
    reg = load_b3db('B3DB_regression.tsv')
    bbbp = pd.read_csv(DATA_DIR / 'BBBP.csv', dtype=str)
    bbbp.columns = [c.strip() for c in bbbp.columns]
    cls['parent_ik'] = cls['SMILES'].apply(parent_inchikey)
    cls['canon'] = cls['SMILES'].apply(canon_smiles)
    reg['parent_ik'] = reg['SMILES'].apply(parent_inchikey)
    reg['canon'] = reg['SMILES'].apply(canon_smiles)
    bbbp['parent_ik'] = bbbp['smiles'].apply(parent_inchikey)
    bbbp['canon'] = bbbp['smiles'].apply(canon_smiles)
    reg['logbb_num'] = pd.to_numeric(reg['logBB'], errors='coerce')
    reg_logbb = reg[reg['logbb_num'].notna()].groupby('parent_ik').agg(logbb_list=('logbb_num', lambda s: ';'.join((f'{x:.3f}' for x in sorted(s.tolist())))), logbb_min=('logbb_num', 'min'), logbb_max=('logbb_num', 'max'), logbb_n=('logbb_num', 'count'), logbb_refs=('reference', lambda s: '|'.join(sorted({r for ref in s for r in str(ref).split('|') if r.strip()}))))
    reg_logbb['logbb_min'] = reg_logbb['logbb_min'].apply(lambda x: f'{x:.3f}')
    reg_logbb['logbb_max'] = reg_logbb['logbb_max'].apply(lambda x: f'{x:.3f}')
    long_rows = []
    for _, r in reg[reg['logbb_num'].notna()].iterrows():
        refs = parse_refs(r.get('reference', ''))
        long_rows.append({'parent_ik': r['parent_ik'], 'canon_smiles': r['canon'], 'logbb': f"{r['logbb_num']:.3f}", 'logbb_reference_list': '|'.join(refs), 'n_logbb_references': len(refs)})
    pd.DataFrame(long_rows).to_csv(OUTPUT_DIR / 'data' / 'logbb_measurements_long.csv', index=False, encoding='utf-8-sig')
    cls_fail = int(cls['parent_ik'].isna().sum())
    bbbp_fail = int(bbbp['parent_ik'].isna().sum())
    fail_records = []
    for _, r in cls[cls['parent_ik'].isna()].iterrows():
        fail_records.append({'source': 'B3DB_classification', 'smiles': r['SMILES'], 'compound_name': '', 'reason': 'parent InChIKey failed'})
    for _, r in bbbp[bbbp['parent_ik'].isna()].iterrows():
        fail_records.append({'source': 'BBBP', 'smiles': r['smiles'], 'compound_name': '', 'reason': 'parent InChIKey failed'})
    pd.DataFrame(fail_records).to_csv(OUTPUT_DIR / 'reports' / 'parent_inchikey_failures.csv', index=False, encoding='utf-8-sig')
    cls['label'] = (cls['BBB+/BBB-'].str.strip() == 'BBB+').astype(int)
    cls['refs'] = cls['reference'].apply(parse_refs)
    cls['nrefs'] = cls['refs'].apply(len)
    b3db_rows = []
    for ik, g in cls.groupby('parent_ik'):
        if pd.isna(ik):
            continue
        labels = sorted(set((int(x) for x in g['label'].dropna().astype(int))))
        allrefs = sorted({r for refs in g['refs'] for r in refs})
        consensus = None
        if len(labels) == 1:
            consensus = labels[0]
        elif len(labels) > 1:
            consensus = int(g['label'].dropna().astype(int).iloc[0])
        rl = reg_logbb.loc[ik] if ik in reg_logbb.index else None
        logbb_vals = rl['logbb_list'] if rl is not None else ''
        b3db_rows.append({'parent_ik': ik, 'canon_smiles': g['canon'].dropna().iloc[0], 'b3db_raw_labels': ';'.join((str(x) for x in labels)), 'b3db_label': '' if consensus is None else consensus, 'b3db_label_conflict': 1 if len(labels) > 1 else 0, 'reference_list': '|'.join(allrefs), 'n_references': len(allrefs), 'b3db_logbb_list': logbb_vals, 'b3db_logbb_min': str(rl['logbb_min']) if rl is not None else '', 'b3db_logbb_max': str(rl['logbb_max']) if rl is not None else '', 'b3db_logbb_n': int(rl['logbb_n']) if rl is not None else 0, 'b3db_logbb_refs': rl['logbb_refs'] if rl is not None else ''})
    b3db = pd.DataFrame(b3db_rows)
    bbbp['label'] = bbbp['p_np'].astype(int)
    bbbp_rows = []
    for ik, g in bbbp.groupby('parent_ik'):
        if pd.isna(ik):
            continue
        labels = sorted(set((int(x) for x in g['label'].dropna().astype(int))))
        rep = int(g['label'].dropna().astype(int).iloc[0]) if len(labels) else ''
        bbbp_rows.append({'parent_ik': ik, 'canon_smiles': g['canon'].dropna().iloc[0], 'bbbp_name': g['name'].dropna().iloc[0] if 'name' in g else '', 'bbbp_raw_labels': ';'.join((str(x) for x in labels)), 'bbbp_label': rep, 'bbbp_label_conflict': 1 if len(labels) > 1 else 0})
    bbbp_u = pd.DataFrame(bbbp_rows)
    merged = bbbp_u.merge(b3db, on='parent_ik', how='outer', suffixes=('_bbbp', '_b3db'))
    merged['canon_smiles'] = merged['canon_smiles_bbbp'].fillna(merged['canon_smiles_b3db'])
    for col in ('bbbp_label', 'b3db_label'):
        merged[col] = merged[col].fillna('')
    merged['overlap'] = (merged['bbbp_label'].ne('') & merged['b3db_label'].ne('')).astype(int)

    def div_flag(r):
        b, d = (r['bbbp_label'], r['b3db_label'])
        if b == '' or d == '':
            return 0
        return 1 if int(b) != int(d) else 0
    merged['divergence'] = merged.apply(div_flag, axis=1)
    merged['source'] = np.where(merged['overlap'] == 1, 'overlap', np.where(merged['bbbp_label'].ne(''), 'bbbp_only', 'b3db_only'))
    merged['reference_list'] = merged['reference_list'].fillna('')
    merged['n_references'] = merged['n_references'].fillna(0).astype(int)
    rng = np.random.RandomState(SEED)
    merged['split'] = ''
    merged['in_reference_out'] = 0
    bbbp_mask = merged['bbbp_label'].ne('')
    bbbp_idx = merged.index[bbbp_mask].to_numpy().copy()
    n_test = int(round(len(bbbp_idx) * TEST_FRAC))
    rng.shuffle(bbbp_idx)
    test_idx = bbbp_idx[:n_test]
    merged.loc[test_idx, 'split'] = 'test'
    merged['in_reference_out'] = merged['reference_list'].apply(lambda refs: 1 if refs and set(refs.split('|')) & REF_OUT_REFS else 0)
    train_pool = merged.index[(merged['split'] != 'test') & (merged['in_reference_out'] == 0)].to_numpy().copy()
    rng.shuffle(train_pool)
    n_val = int(round(len(train_pool) * VAL_FRAC / (1 - TEST_FRAC)))
    val_idx = train_pool[:n_val]
    train_idx = train_pool[n_val:]
    merged.loc[val_idx, 'split'] = 'validation'
    merged.loc[train_idx, 'split'] = 'train'
    train_keys = set(merged.loc[merged['split'] == 'train', 'parent_ik'])
    test_keys = set(merged.loc[merged['split'] == 'test', 'parent_ik'])
    ro_keys = set(merged.loc[merged['in_reference_out'] == 1, 'parent_ik'])
    leak_test = len(train_keys & test_keys)
    leak_ro = len(train_keys & ro_keys)
    snap_cols = ['parent_ik', 'canon_smiles', 'split', 'reference_list', 'bbbp_label', 'b3db_label']
    snap = merged[snap_cols].copy()
    snap['row_key'] = snap.astype(str).agg('|'.join, axis=1)
    split_hash = hashlib.sha256('\n'.join(sorted(snap['row_key'])).encode('utf-8')).hexdigest()
    merged.to_csv(OUTPUT_DIR / 'data' / 'consensus_dataset.csv', index=False, encoding='utf-8-sig')
    merged.to_json(OUTPUT_DIR / 'data' / 'consensus_dataset.jsonl', orient='records', lines=True, force_ascii=False)
    ref_map = pd.DataFrame([{'reference': r, 'source': '待人工查证（B3DB 原始论文 R1-R50 映射表）', 'doi': '', 'status': 'TODO_manual_traceability'} for r in range(1, 51)])
    ref_map.to_csv(OUTPUT_DIR / 'data' / 'reference_map.csv', index=False, encoding='utf-8-sig')
    ref_counter = Counter()
    for refs in merged['reference_list']:
        for r in refs.split('|'):
            if r:
                ref_counter[r] += 1
    pd.DataFrame([{'reference': r, 'n_molecules': c} for r, c in ref_counter.most_common()]).to_csv(OUTPUT_DIR / 'reports' / 'reference_grouping.csv', index=False, encoding='utf-8-sig')
    lock = {'seed': SEED, 'n_total': len(merged), 'n_train': int((merged['split'] == 'train').sum()), 'n_val': int((merged['split'] == 'validation').sum()), 'n_test': int((merged['split'] == 'test').sum()), 'reference_out_refs': sorted(REF_OUT_REFS), 'n_in_reference_out': int(merged['in_reference_out'].sum()), 'leak_train_test': int(leak_test), 'leak_train_reference_out': int(leak_ro), 'split_hash': split_hash, 'note': 'BBBP 15% test locked; reference-out (R27) excluded from train.'}
    with open(OUTPUT_DIR / 'reports' / 'split_lock.json', 'w', encoding='utf-8') as f:
        json.dump(lock, f, ensure_ascii=False, indent=2)
    meta = {'n_total': len(merged), 'overlap': int((merged['overlap'] == 1).sum()), 'divergence': int((merged['divergence'] == 1).sum()), 'b3db_internal_conflict': int(merged['b3db_label_conflict'].fillna(0).astype(int).sum()), 'bbbp_internal_conflict': int(merged['bbbp_label_conflict'].fillna(0).astype(int).sum()), 'n_in_reference_out': int(merged['in_reference_out'].sum()), 'zero_intersection_train_test': leak_test == 0, 'zero_intersection_train_reference_out': leak_ro == 0, 'split_hash': split_hash, 'reference_distinct': len(ref_counter), 'reference_map_status': 'placeholder; R1-R50 DOI pending (manual)'}
    with open(OUTPUT_DIR / 'reports' / 'stage5_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('=== CHECKLIST ===')
    print(f'  parent InChIKey: BBBP {len(bbbp_u)} / B3DB {len(b3db)}, 失败 {bbbp_fail}/{cls_fail}')
    print(f"  共识集 {len(merged)}, overlap {meta['overlap']}, 分歧 {meta['divergence']}")
    print(f"  split: {lock['n_train']}/{lock['n_val']}/{lock['n_test']}, refout {meta['n_in_reference_out']}")
    print(f'  train∩test={leak_test}, train∩refout={leak_ro}')
    print(f'  split_hash={split_hash[:12]}...')
if __name__ == '__main__':
    main()

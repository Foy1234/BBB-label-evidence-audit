from __future__ import annotations
from project_paths import project_root, project_path
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
CONSENSUS = project_path('05_复现工作区', 'results_stage5', 'data', 'consensus_dataset.csv')
OUTPUT_DIR = project_path('05_复现工作区', 'results_stage7')
SEED = 42
VAL_FRAC_WITHIN_TRAIN = 0.2

def scaffold_of(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smi)
    except Exception:
        return 'INVALID'

def main():
    (OUTPUT_DIR / 'data').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CONSENSUS, dtype=str).fillna('')
    keep = df[df['in_reference_out'].astype(int) == 0].copy()
    train_pool = keep[keep['split'] == 'train'].reset_index(drop=True)
    test_pool = keep[keep['split'] != 'train'].reset_index(drop=True)
    train_pool['scaffold'] = train_pool['canon_smiles'].apply(scaffold_of)
    test_pool['scaffold'] = test_pool['canon_smiles'].apply(scaffold_of)
    train_scaffolds = set(train_pool['scaffold'])
    test_mask = test_pool['scaffold'].apply(lambda s: s not in train_scaffolds)
    scaffold_test_idx = test_pool.index[test_mask].tolist()
    train_scaf_mols = defaultdict(list)
    for i, row in train_pool.iterrows():
        train_scaf_mols[row['scaffold']].append(i)
    rng = np.random.RandomState(SEED)
    scaffolds = sorted(train_scaf_mols.keys())
    rng.shuffle(scaffolds)
    total = len(train_pool)
    n_val_target = int(round(total * VAL_FRAC_WITHIN_TRAIN))
    val_scafs = set()
    val_count = 0
    for s in scaffolds:
        if val_count >= n_val_target:
            break
        val_scafs.add(s)
        val_count += len(train_scaf_mols[s])
    train_pool['scaffold_split'] = np.where(train_pool['scaffold'].isin(val_scafs), 'validation', 'train')
    test_assignment = test_pool.loc[scaffold_test_idx].copy()
    test_assignment['scaffold_split'] = 'test'
    out = pd.concat([train_pool, test_assignment], ignore_index=True)
    test_keys = set(out.loc[out['scaffold_split'] == 'test', 'parent_ik'])
    train_keys = set(out.loc[out['scaffold_split'] == 'train', 'parent_ik'])
    st5_train = set(df[(df['split'] == 'train') & (df['in_reference_out'].astype(int) == 0)]['parent_ik'])
    overlap_tr = len(train_keys & test_keys)
    overlap_tr5 = len(st5_train & test_keys)
    scaf_tr = set(out.loc[out['scaffold_split'] == 'train', 'scaffold'])
    scaf_test = set(out.loc[out['scaffold_split'] == 'test', 'scaffold'])
    scaf_overlap = len(scaf_tr & scaf_test)
    cols = ['parent_ik', 'canon_smiles', 'scaffold', 'scaffold_split', 'bbbp_label', 'b3db_label', 'n_references']
    out[cols].to_csv(OUTPUT_DIR / 'data' / 'scaffold_split_assignments.csv', index=False, encoding='utf-8-sig')
    summary = {'seed': SEED, 'train_pool': len(train_pool), 'test_pool_candidates': len(test_pool), 'scaffold_test': len(test_assignment), 'split_counts': out['scaffold_split'].value_counts().to_dict(), 'parent_overlap_train_test': overlap_tr, 'parent_overlap_stage5train_test': overlap_tr5, 'scaffold_overlap_train_test': scaf_overlap, 'zero_intersection': overlap_tr == 0 and overlap_tr5 == 0 and (scaf_overlap == 0)}
    with open(OUTPUT_DIR / 'reports' / 'scaffold_split_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('=== scaffold split ===')
    print(f"  split: {summary['split_counts']}")
    print(f'  scaffold test: {len(test_assignment)}')
    print(f'  parent overlap(train/test): {overlap_tr}')
    print(f'  scaffold test vs Stage5 train: {overlap_tr5}')
    print(f"  零交集: {summary['zero_intersection']}")
if __name__ == '__main__':
    main()

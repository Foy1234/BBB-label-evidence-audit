from __future__ import annotations
from project_paths import project_root, project_path
import hashlib
import json
from pathlib import Path
import pandas as pd
CONSENSUS = project_path('05_复现工作区', 'results_stage5', 'data', 'consensus_dataset.csv')
SCAFFOLD = project_path('05_复现工作区', 'results_stage7', 'data', 'scaffold_split_assignments.csv')
OUTPUT_DIR = project_path('05_复现工作区', 'results_stage7')

def hset(keys):
    return hashlib.sha256('\n'.join(sorted(keys)).encode('utf-8')).hexdigest()

def main():
    (OUTPUT_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CONSENSUS, dtype=str).fillna('')
    sc = pd.read_csv(SCAFFOLD, dtype=str).fillna('')
    bbbp_test = set(df[df['split'] == 'test']['parent_ik'])
    refout = set(df[df['in_reference_out'].astype(int) == 1]['parent_ik'])
    scaffold_test = set(sc[sc['scaffold_split'] == 'test']['parent_ik'])
    scaffold_train = set(sc[sc['scaffold_split'] == 'train']['parent_ik'])
    train = set(df[(df['split'] == 'train') & (df['in_reference_out'].astype(int) == 0)]['parent_ik'])
    sets = {'BBBP_test': sorted(bbbp_test), 'reference_out': sorted(refout), 'scaffold_test': sorted(scaffold_test), 'scaffold_train': sorted(scaffold_train)}
    checks = {'n_BBBP_test': len(bbbp_test), 'n_reference_out': len(refout), 'n_scaffold_test': len(scaffold_test), 'n_training_protocol': len(train), 'train_vs_BBBP_test': len(train & bbbp_test), 'train_vs_reference_out': len(train & refout), 'train_vs_scaffold_test': len(train & scaffold_test), 'BBBP_test_vs_refout_overlap': len(bbbp_test & refout), 'scaffold_test_vs_refout_overlap': len(scaffold_test & refout), 'scaffold_test_vs_BBBP_test_overlap': len(scaffold_test & bbbp_test)}
    checks['training_isolated'] = checks['train_vs_BBBP_test'] == 0 and checks['train_vs_reference_out'] == 0 and (checks['train_vs_scaffold_test'] == 0)
    frozen = {'hash': {k: hset(v) for k, v in sets.items()}, 'counts': {k: len(v) for k, v in sets.items()}, 'checks': checks, 'overlaps': {'BBBP_test_vs_reference_out': checks['BBBP_test_vs_refout_overlap'], 'scaffold_test_vs_BBBP_test': checks['scaffold_test_vs_BBBP_test_overlap'], 'scaffold_test_vs_reference_out': checks['scaffold_test_vs_refout_overlap']}, 'note': f"Three evaluation views are NOT mutually exclusive: BBBP test (291) overlaps reference-out by {checks['BBBP_test_vs_refout_overlap']} ({checks['BBBP_test_vs_refout_overlap'] / 291 * 100:.0f}%). Training uses split=='train' AND in_reference_out==0 only."}
    with open(OUTPUT_DIR / 'reports' / 'eval_sets_frozen.json', 'w', encoding='utf-8') as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)
    print('=== 三类评估视图冻结 ===')
    for k in ('BBBP_test', 'reference_out', 'scaffold_test'):
        print(f"  {k}: {frozen['counts'][k]} hash={frozen['hash'][k][:12]}...")
    print(f"  训练协议: {checks['n_training_protocol']}")
    print(f"  训练隔离: {checks['training_isolated']}")
if __name__ == '__main__':
    main()

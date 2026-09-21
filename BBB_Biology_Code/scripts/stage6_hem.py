from __future__ import annotations
from project_paths import project_root, project_path
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hem import LatentHEM, build_tensors, build_ref_matrix, collect_references, abstain_summary, coverage_risk
OUTPUT_DIR = project_path('05_复现工作区', 'results_stage6')
CONSENSUS = project_path('05_复现工作区', 'results_stage5', 'data', 'consensus_dataset.csv')

def check_dependencies():
    import torch
    import rdkit
    import xgboost
    import shap
    import torch_geometric
    return {'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__, 'rdkit': rdkit.__version__, 'xgboost': xgboost.__version__, 'shap': shap.__version__}

def lbl(s):
    s = str(s).strip()
    return float(s) if s in ('0', '1', '0.0', '1.0') else np.nan

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    deps = check_dependencies()
    from hem.data import load_consensus
    df = load_consensus(CONSENSUS)
    train_mask = (df['split'] == 'train').values & (df['in_reference_out'].astype(int) == 0).values
    tr_df = df[train_mask].reset_index(drop=True)
    T = build_tensors(tr_df, use_reference_out=False)
    m = LatentHEM(len(tr_df), n_em_iters=25, m_epochs=50, m_lr=0.01, l2_prior=0.1)
    m.fit(T['b3db_idx'], T['b3db_labels'], bbbp_idx=T['bbbp_idx'], bbbp_labels=T['bbbp_labels'], logbb_idx=T['logbb_idx'], logbb_vals=T['logbb_vals'])
    refs = collect_references(tr_df)
    tr_ref = build_ref_matrix(tr_df, refs)
    tr_labels = tr_df['b3db_label'].replace('', np.nan).astype(float).values
    src = m.source_contributions(tr_ref, tr_labels)
    src_df = {'references': refs, 'reliability': src['reliability'].round(4).tolist(), 'count': src['count'].tolist(), 'semantics': 'posterior agreement score (model posterior), NOT fitted per-source params; B3DB has no per-reference labels. Do not report as identified per-source reliability.'}
    with open(OUTPUT_DIR / 'source_reliability.json', 'w', encoding='utf-8') as f:
        json.dump(src_df, f, ensure_ascii=False, indent=2)
    val_mask = (df['split'] == 'validation').values & (df['in_reference_out'].astype(int) == 0).values
    val_df = df[val_mask].reset_index(drop=True)
    T_val = build_tensors(val_df, use_reference_out=True)
    m_val = LatentHEM(len(val_df), n_em_iters=15, m_epochs=30, m_lr=0.01, l2_prior=0.1)
    m_val.fit(T_val['b3db_idx'], T_val['b3db_labels'], bbbp_idx=T_val['bbbp_idx'], bbbp_labels=T_val['bbbp_labels'], logbb_idx=T_val['logbb_idx'], logbb_vals=T_val['logbb_vals'])
    y_true = val_df['b3db_label'].replace('', np.nan).astype(float).values
    has_lab = val_df['b3db_label'].ne('').values
    y_pred = m_val.predict(np.arange(len(val_df)))
    unc = m_val.predict_with_uncertainty(np.arange(len(val_df)))
    if has_lab.sum() == 0:
        s = {'n_total': 0, 'n_abstain': 0, 'abstain_ratio': 0.0, 'n_covered': 0, 'covered_accuracy': float('nan'), 'covered_risk': float('nan')}
        cr = {'tau': [], 'coverage': [], 'risk': []}
    else:
        s = abstain_summary(unc['std_latent'][has_lab], y_true[has_lab], y_pred[has_lab], tau=0.8)
        cr = coverage_risk(unc['std_latent'][has_lab], y_true[has_lab], y_pred[has_lab])
        s['note'] = 'Stage 6 mechanism demonstration on held-out molecules; latent refit on eval labels so accuracy is NOT independent (Stage 7 reports real numbers).'
    m.save(str(OUTPUT_DIR / 'hem_model.json'))
    results = {'dependencies': deps, 'n_train_molecules': len(tr_df), 'n_b3db_labels': int(len(T['b3db_idx'])), 'delta': float(m.delta.detach()), 'logbb_a': float(m.a.detach()), 'logbb_b': float(m.b.detach()), 'logbb_sigma': float(np.exp(m.log_sigma.detach())), 'source_reliability': src_df, 'abstain_validation': s, 'coverage_risk_points': {'tau': cr['tau'].tolist(), 'coverage': cr['coverage'].tolist(), 'risk': cr['risk'].tolist()}}
    with open(OUTPUT_DIR / 'stage6_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print('=== Stage6 完成 ===')
    print(f"  n_train={len(tr_df)}, n_b3db_labels={len(T['b3db_idx'])}")
    print(f'  delta={float(m.delta.detach()):.4f}, a={float(m.a.detach()):.4f}')
    print(f"  abstain ratio={s['abstain_ratio']:.3f}")
    print(f'  保存: {OUTPUT_DIR}')
if __name__ == '__main__':
    main()

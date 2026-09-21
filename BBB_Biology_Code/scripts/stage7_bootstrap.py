from __future__ import annotations
from project_paths import project_root, project_path
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
OUTPUT_DIR = project_path('05_复现工作区', 'results_stage7')
SEED = 42
N_BOOT = 2000

def roc(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))

def pr(y_true, y_pred):
    return float(average_precision_score(y_true, y_pred))

def main():
    encoder = sys.argv[1] if len(sys.argv) > 1 else 'xgboost'
    (OUTPUT_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    preds_file = 'arm_predictions.npz' if encoder == 'xgboost' else 'gine_predictions.npz'
    d = np.load(OUTPUT_DIR / 'reports' / preds_file)
    evals = {'BBBP_test': 'gold_bbbp', 'scaffold_test': 'gold_scaf', 'reference_out': 'gold_refout'}
    arms = ['A', 'B', 'C']
    rng = np.random.RandomState(SEED)
    results = []
    dist_store = {}
    for eval_name, gold_key in evals.items():
        gold = d[gold_key]
        n = len(gold)
        preds = {a: d[f'{a}_{eval_name}'] for a in arms}
        base = {a: {'roc': roc(gold, preds[a]), 'pr': pr(gold, preds[a])} for a in arms}
        pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        for a1, a2 in pairs:
            diff_roc = np.zeros(N_BOOT)
            diff_pr = np.zeros(N_BOOT)
            for b in range(N_BOOT):
                idx = rng.randint(0, n, n)
                g = gold[idx]
                p1, p2 = (preds[a1][idx], preds[a2][idx])
                diff_roc[b] = roc(g, p1) - roc(g, p2)
                diff_pr[b] = pr(g, p1) - pr(g, p2)
            obs_roc = base[a1]['roc'] - base[a2]['roc']
            obs_pr = base[a1]['pr'] - base[a2]['pr']
            p1_all, p2_all = (preds[a1], preds[a2])
            perm_roc = np.zeros(N_BOOT)
            perm_pr = np.zeros(N_BOOT)
            for b in range(N_BOOT):
                swap = rng.rand(n) < 0.5
                q1 = np.where(swap, p2_all, p1_all)
                q2 = np.where(swap, p1_all, p2_all)
                perm_roc[b] = roc(gold, q1) - roc(gold, q2)
                perm_pr[b] = pr(gold, q1) - pr(gold, q2)
            p_roc = float((1 + np.sum(np.abs(perm_roc) >= np.abs(obs_roc))) / (N_BOOT + 1))
            p_pr = float((1 + np.sum(np.abs(perm_pr) >= np.abs(obs_pr))) / (N_BOOT + 1))
            med_roc = float(np.median(diff_roc))
            med_pr = float(np.median(diff_pr))
            results.append({'eval': eval_name, 'arm1': a1, 'arm2': a2, 'roc_A1': base[a1]['roc'], 'roc_A2': base[a2]['roc'], 'roc_diff': obs_roc, 'roc_diff_med': med_roc, 'roc_diff_ci95': [float(np.percentile(diff_roc, 2.5)), float(np.percentile(diff_roc, 97.5))], 'roc_p_perm': p_roc, 'pr_A1': base[a1]['pr'], 'pr_A2': base[a2]['pr'], 'pr_diff': obs_pr, 'pr_diff_med': med_pr, 'pr_diff_ci95': [float(np.percentile(diff_pr, 2.5)), float(np.percentile(diff_pr, 97.5))], 'pr_p_perm': p_pr, 'n': int(n)})
            dist_store[f'{eval_name}_{a1}vs{a2}_roc'] = diff_roc
            dist_store[f'{eval_name}_{a1}vs{a2}_pr'] = diff_pr
    ps = np.array([r['roc_p_perm'] for r in results] + [r['pr_p_perm'] for r in results])
    n_comp = len(ps)
    sorted_idx = np.argsort(ps, kind='stable')
    rank = np.empty(n_comp, dtype=float)
    for i, idx in enumerate(sorted_idx, start=1):
        rank[idx] = i
    q_raw = ps * n_comp / rank
    q_sorted = np.empty(n_comp)
    running = 1.0
    for i in range(n_comp - 1, -1, -1):
        idx = sorted_idx[i]
        running = min(running, q_raw[idx])
        q_sorted[idx] = running
    for i, r in enumerate(results):
        r['roc_q'] = float(min(1.0, q_sorted[i]))
        r['pr_q'] = float(min(1.0, q_sorted[n_comp // 2 + i]))
    with open(OUTPUT_DIR / 'reports' / f'bootstrap_results_{encoder}.json', 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'n_bootstrap': N_BOOT, 'encoder': encoder, 'n_comparisons': n_comp, 'bh_family': 'per-encoder', 'bh_family_note': 'BH within each encoder as pre-specified 18-comparison family; two encoder families reported separately (NOT pooled 36).', 'results': results}, f, ensure_ascii=False, indent=2, default=str)
    np.savez(OUTPUT_DIR / 'reports' / f'bootstrap_distributions_{encoder}.npz', **dist_store)
    print(f'=== 配对置换检验 {encoder} (n={N_BOOT}, {n_comp} 比较) ===')
    for r in results:
        sig = 'sig' if r['roc_q'] < 0.05 else ''
        print(f"  {r['eval']} {r['arm1']}vs{r['arm2']}: roc Δ={r['roc_diff']:+.4f} [{r['roc_diff_ci95'][0]:.4f},{r['roc_diff_ci95'][1]:.4f}] p_perm={r['roc_p_perm']:.4f} q={r['roc_q']:.4f} {sig}")
if __name__ == '__main__':
    main()

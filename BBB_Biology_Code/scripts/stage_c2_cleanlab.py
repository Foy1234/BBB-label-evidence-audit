from __future__ import annotations
from project_paths import project_root, project_path
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
RDLogger.DisableLog('rdApp.*')
ROOT = project_root()
CONSENSUS = ROOT / '05_复现工作区' / 'results_stage5' / 'data' / 'consensus_dataset.csv'
B3DB_RAW = ROOT / '00_数据_原始' / 'B3DB_数据' / 'B3DB_classification.tsv'
SCAFFOLD = ROOT / '05_复现工作区' / 'results_stage7' / 'data' / 'scaffold_split_assignments.csv'
OUT = ROOT / '05_复现工作区' / 'results_c2'
RADIUS = 2
N_BITS = 2048
SEED = 42
XGB_PARAMS = dict(max_depth=5, learning_rate=0.03, n_estimators=300, subsample=0.8, colsample_bytree=0.8, random_state=SEED)

def lbl(s):
    s = str(s).strip()
    return float(s) if s in ('0', '1', '0.0', '1.0') else np.nan

def make_fingerprints(smiles_list):
    gen = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS, includeChirality=True)
    mat = np.zeros((len(smiles_list), N_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        fp = gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, mat[i])
    return mat

def vote_soft_label_map():
    cls = pd.read_csv(B3DB_RAW, sep='\t', dtype=str)
    cls['label'] = (cls['BBB+/BBB-'].str.strip() == 'BBB+').astype(int)
    sr = SaltRemover.SaltRemover()

    def pik(s):
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            m = sr.StripMol(m)
            Chem.RemoveStereochemistry(m)
            return MolToInchiKey(m).split('-')[0]
        except Exception:
            return None
    cls['pik'] = cls['SMILES'].apply(pik)
    vote = cls.groupby('pik')['label'].mean()
    return vote.to_dict()

def metrics(y_true, y_pred):
    return {'roc_auc': float(roc_auc_score(y_true, y_pred)), 'pr_auc': float(average_precision_score(y_true, y_pred))}

def cleanlab_oos_probabilities(X, y):
    n = len(y)
    oos = np.full(n, np.nan)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in skf.split(X, y):
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X[tr_idx], y[tr_idx])
        oos[va_idx] = model.predict_proba(X[va_idx])[:, 1]
    return oos

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from cleanlab.filter import find_label_issues
    from cleanlab.count import compute_confident_joint
    df = pd.read_csv(CONSENSUS, dtype=str).fillna('')
    sc = pd.read_csv(SCAFFOLD, dtype=str).fillna('')
    tr = df[(df['split'] == 'train') & (df['in_reference_out'].astype(int) == 0)].reset_index(drop=True)
    X_tr_all = make_fingerprints(tr['canon_smiles'].tolist())
    a_label = tr['b3db_label'].apply(lbl).values
    has_a = ~np.isnan(a_label)
    y_a = a_label[has_a].astype(int)
    X_a = X_tr_all[has_a]
    oos_prob = cleanlab_oos_probabilities(X_a, y_a)
    issues = find_label_issues(labels=y_a, pred_probs=np.column_stack([1 - oos_prob, oos_prob]), return_indices_ranked_by='self_confidence')
    issue_idx_local = issues if isinstance(issues, np.ndarray) else issues['index']
    issue_mask_local = np.zeros(len(y_a), dtype=bool)
    issue_mask_local[issue_idx_local] = True
    confident_joint = compute_confident_joint(labels=y_a, pred_probs=np.column_stack([1 - oos_prob, oos_prob]))
    y_cl = y_a.copy()
    y_cl[issue_mask_local] = 1 - y_cl[issue_mask_local]
    n_issues = int(issue_mask_local.sum())
    print(f'cleanlab 检测疑错标签: {n_issues}/{len(y_a)} ({n_issues / len(y_a) * 100:.1f}%)')
    tr_local = tr[has_a].reset_index(drop=True)
    tr_local['issue'] = issue_mask_local
    tr_local['oos_prob'] = oos_prob
    tr_local['cl_label'] = y_cl
    tr_local_bbbp = tr_local[tr_local['bbbp_label'].apply(lbl).notna()]
    print(f'疑错分子中有 BBBP 标签的: {len(tr_local_bbbp)}')
    vote = vote_soft_label_map()
    b_vote = tr['parent_ik'].map(vote).values
    b_label = np.where(pd.notna(b_vote), b_vote, a_label)
    b_mask = has_a
    abc = ROOT / '05_复现工作区' / 'results_stage7' / 'data' / 'labels_abc.csv'
    lab = pd.read_csv(abc, dtype=str).fillna('')
    lab['parent_ik'] = lab['parent_ik'].str.strip()
    tr['label_C'] = tr['parent_ik'].map(lab.set_index('parent_ik')['label_C_sigmoid_mu']).astype(float).values
    c_label = tr['label_C'].values
    c_mask = has_a & pd.notna(c_label)
    bbbp_test = df[df['split'] == 'test'].reset_index(drop=True)
    scaf_test = sc[sc['scaffold_split'] == 'test'].reset_index(drop=True)
    refout_df = df[df['in_reference_out'].astype(int) == 1].reset_index(drop=True)
    gold_meta = {'BBBP_test': ('bbbp_label', 'BBBP'), 'scaffold_test': ('b3db_label', 'B3DB'), 'reference_out': ('b3db_label', 'B3DB')}
    eval_sets = [('BBBP_test', bbbp_test), ('scaffold_test', scaf_test), ('reference_out', refout_df)]
    results = []
    cl_full = a_label.copy()
    cl_full[has_a] = y_cl
    arms = [('A', a_label, has_a), ('B', b_label, b_mask), ('C', c_label, c_mask), ('CL', cl_full, has_a)]
    for arm, y, mask in arms:
        X_arm = X_tr_all[mask]
        y_arm = y[mask]
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_arm, y_arm)
        for eval_name, eval_df in eval_sets:
            gold_col, gold_src = gold_meta[eval_name]
            X_ev = make_fingerprints(eval_df['canon_smiles'].tolist())
            gold = eval_df[gold_col].apply(lbl).values
            valid = ~np.isnan(gold)
            if valid.sum() == 0:
                continue
            p = model.predict(X_ev[valid])
            m = metrics(gold[valid], p)
            m.update({'arm': arm, 'eval': eval_name, 'gold': gold_src, 'n': int(valid.sum())})
            results.append(m)
    main_table = pd.DataFrame(results)
    main_table.to_csv(OUT / 'main_table_cleanlab.csv', index=False, encoding='utf-8-sig')
    report = {'n_train_with_label': int(has_a.sum()), 'n_label_issues_cleanlab': n_issues, 'issue_rate': float(n_issues / len(y_a)), 'confident_joint': confident_joint.tolist(), 'noise_estimate': {'p_of_1_given_true_1': None, 'p_of_0_given_true_0': None, 'note': '由 confident_joint 归一化'}, 'main_table': main_table.to_dict(orient='records'), 'note': 'CL = cleanlab 清洗标签（疑错翻转）。A 硬标签/B 软标签/C HEM 为 stage7 对应。训练集 B3DB 标签 1157 个，BBBP 标签仅 95 个，无双标签样本，cleanlab 仅基于模型 OOS 预测概率清洗 B3DB 硬标签。'}
    cj = np.asarray(confident_joint, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cj.sum(axis=1, keepdims=True)
        pnoise = cj / np.where(row_sums == 0, 1, row_sums)
    report['noise_estimate']['p_of_1_given_true_1'] = float(pnoise[1, 1])
    report['noise_estimate']['p_of_0_given_true_0'] = float(pnoise[0, 0])
    report['noise_estimate']['p_of_0_given_true_1'] = float(pnoise[1, 0])
    report['noise_estimate']['p_of_1_given_true_0'] = float(pnoise[0, 1])
    issue_rows = tr_local[tr_local['issue']][['parent_ik', 'canon_smiles', 'b3db_label', 'bbbp_label', 'oos_prob', 'cl_label']].copy()
    issue_rows['b3db_label'] = issue_rows['b3db_label'].astype(float)
    issue_rows.to_csv(OUT / 'cleanlab_issues.csv', index=False, encoding='utf-8-sig')
    with open(OUT / 'stage_c2_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print('\n=== cleanlab 对标主表 (ROC-AUC / PR-AUC) ===')
    print(main_table.to_string(index=False))
    print(f"\n保存: {OUT / 'stage_c2_report.json'}")
if __name__ == '__main__':
    main()

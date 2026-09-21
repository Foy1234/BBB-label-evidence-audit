from __future__ import annotations
from project_paths import project_root, project_path
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hem import LatentHEM, build_tensors
RDLogger.DisableLog('rdApp.*')
CONSENSUS = project_path('05_复现工作区', 'results_stage5', 'data', 'consensus_dataset.csv')
B3DB_RAW = project_path('00_数据_原始', 'B3DB_数据', 'B3DB_classification.tsv')
SCAFFOLD = project_path('05_复现工作区', 'results_stage7', 'data', 'scaffold_split_assignments.csv')
OUTPUT_DIR = project_path('05_复现工作区', 'results_stage7')
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

def fit_hem_mu(train_df):
    T = build_tensors(train_df, use_reference_out=False)
    m = LatentHEM(len(train_df), n_em_iters=25, m_epochs=50, m_lr=0.01, l2_prior=0.1)
    m.fit(T['b3db_idx'], T['b3db_labels'], bbbp_idx=T['bbbp_idx'], bbbp_labels=T['bbbp_labels'], logbb_idx=T['logbb_idx'], logbb_vals=T['logbb_vals'])
    return (m.predict(np.arange(len(train_df))), m)

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
    return {'roc_auc': float(roc_auc_score(y_true, y_pred)), 'pr_auc': float(average_precision_score(y_true, y_pred)), 'accuracy': float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))}

def main():
    (OUTPUT_DIR / 'data').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CONSENSUS, dtype=str).fillna('')
    sc = pd.read_csv(SCAFFOLD, dtype=str).fillna('')
    tr = df[(df['split'] == 'train') & (df['in_reference_out'].astype(int) == 0)].reset_index(drop=True)
    b3db = tr['b3db_label'].apply(lbl).values
    a_label = b3db.copy()
    vote = vote_soft_label_map()
    b_vote = tr['parent_ik'].map(vote).values
    b_label = np.where(pd.notna(b_vote), b_vote, b3db)
    c_soft, hem = fit_hem_mu(tr)
    c_label = c_soft
    has_label = ~np.isnan(a_label)
    n_continuous_b = int((np.abs(b_label - a_label) > 1e-06).sum())
    print(f'A(B3DB-only) 有标签分子: {has_label.sum()} / {len(tr)}')
    print(f'B 软标签连续值(≠硬标签)分子: {n_continuous_b}')
    X_tr = make_fingerprints(tr['canon_smiles'].tolist())
    bbbp_test = df[df['split'] == 'test'].reset_index(drop=True)
    scaf_test = sc[sc['scaffold_split'] == 'test'].reset_index(drop=True)
    refout_df = df[df['in_reference_out'].astype(int) == 1].reset_index(drop=True)
    val_df = df[df['split'] == 'validation'].reset_index(drop=True)
    gold_meta = {'BBBP_test': ('bbbp_label', 'BBBP'), 'scaffold_test': ('b3db_label', 'B3DB'), 'reference_out': ('b3db_label', 'B3DB'), 'validation': ('b3db_label', 'B3DB')}
    eval_sets = [('BBBP_test', bbbp_test), ('scaffold_test', scaf_test), ('reference_out', refout_df), ('validation', val_df)]
    results = []
    preds_store = {}
    equal_mask = has_label
    for arm, y, mask in [('A', a_label, equal_mask), ('B', b_label, equal_mask), ('C', c_label, equal_mask)]:
        X_arm = X_tr[mask]
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
            preds_store[f'{arm}_{eval_name}'] = {'gold': gold[valid], 'pred': p, 'gold_src': gold_src, 'n': int(valid.sum())}
    main_table = pd.DataFrame(results)
    main_table.to_csv(OUTPUT_DIR / 'reports' / 'main_table.csv', index=False, encoding='utf-8-sig')
    np.savez(OUTPUT_DIR / 'reports' / 'arm_predictions.npz', **{k: v['pred'] for k, v in preds_store.items()}, gold_bbbp=preds_store['A_BBBP_test']['gold'], gold_scaf=preds_store['A_scaffold_test']['gold'], gold_refout=preds_store['A_reference_out']['gold'], gold_val=preds_store['A_validation']['gold'], gold_src_bbbp=preds_store['A_BBBP_test']['gold_src'], gold_src_scaf=preds_store['A_scaffold_test']['gold_src'], gold_src_refout=preds_store['A_reference_out']['gold_src'], gold_src_val=preds_store['A_validation']['gold_src'])
    labels_out = tr[['parent_ik', 'canon_smiles', 'b3db_label', 'bbbp_label']].copy()
    labels_out['label_A'] = a_label
    labels_out['label_B'] = b_label
    labels_out['label_C_sigmoid_mu'] = c_label
    labels_out.to_csv(OUTPUT_DIR / 'data' / 'labels_abc.csv', index=False, encoding='utf-8-sig')
    print('=== 基准主表 ===')
    print(main_table.to_string(index=False))
    print(f"保存: {OUTPUT_DIR / 'reports' / 'main_table.csv'}")
if __name__ == '__main__':
    main()

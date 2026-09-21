from __future__ import annotations
from project_paths import project_root, project_path
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoModel, AutoTokenizer
RDLogger.DisableLog('rdApp.*')
ROOT = project_root()
CONSENSUS = ROOT / '05_复现工作区' / 'results_stage5' / 'data' / 'consensus_dataset.csv'
B3DB_RAW = ROOT / '00_数据_原始' / 'B3DB_数据' / 'B3DB_classification.tsv'
SCAFFOLD = ROOT / '05_复现工作区' / 'results_stage7' / 'data' / 'scaffold_split_assignments.csv'
LABELS_ABC = ROOT / '05_复现工作区' / 'results_stage7' / 'data' / 'labels_abc.csv'
OUT = ROOT / '05_复现工作区' / 'results_c4'
MODEL_ID = 'DeepChem/ChemBERTa-77M-MTR'
SEEDS = [42, 52, 62]
XGB_PARAMS = dict(max_depth=5, learning_rate=0.03, n_estimators=300, subsample=0.8, colsample_bytree=0.8)

def lbl(s):
    s = str(s).strip()
    return float(s) if s in ('0', '1', '0.0', '1.0') else np.nan

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
    return cls.groupby('pik')['label'].mean().to_dict()

def embed(smiles_list, tok, model, batch=32):
    model.eval()
    feats = np.zeros((len(smiles_list), 384), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch):
            smis = smiles_list[i:i + batch]
            enc = tok(smis, return_tensors='pt', padding=True, truncation=True, max_length=256)
            out = model(**enc)
            h = out.last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            feats[i:i + len(smis)] = pooled.numpy()
    return feats

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = __import__('time').time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    df = pd.read_csv(CONSENSUS, dtype=str).fillna('')
    sc = pd.read_csv(SCAFFOLD, dtype=str).fillna('')
    tr = df[(df['split'] == 'train') & (df['in_reference_out'].astype(int) == 0)].reset_index(drop=True)
    bbbp_test = df[df['split'] == 'test'].reset_index(drop=True)
    scaf = sc[sc['scaffold_split'] == 'test'].reset_index(drop=True)
    refout = df[df['in_reference_out'].astype(int) == 1].reset_index(drop=True)
    sets = {'train': tr, 'BBBP_test': bbbp_test, 'scaffold_test': scaf, 'reference_out': refout}
    embs = {k: embed(v['canon_smiles'].tolist(), tok, model) for k, v in sets.items()}
    print('embedding done in %.0fs' % (__import__('time').time() - t0), flush=True)
    a_label = tr['b3db_label'].apply(lbl).values
    has_a = ~np.isnan(a_label)
    y_a = a_label[has_a].astype(int)
    X_tr = embs['train'][has_a]
    vote = vote_soft_label_map()
    b_vote = tr['parent_ik'].map(vote).values
    b_full = np.where(pd.notna(b_vote), b_vote, a_label)
    y_b = b_full[has_a].astype(float)
    lab = pd.read_csv(LABELS_ABC, dtype=str).fillna('')
    lab['parent_ik'] = lab['parent_ik'].str.strip()
    tr['label_C'] = tr['parent_ik'].map(lab.set_index('parent_ik')['label_C_sigmoid_mu']).astype(float).values
    c_full = tr['label_C'].values
    y_c2 = c_full
    c_mask = has_a & pd.notna(c_full)
    X_c = embs['train'][c_mask]
    y_cm = y_c2[c_mask]
    arms = {'A': (X_tr, y_a, np.ones(len(y_a), dtype=bool)), 'B': (X_tr, y_b, np.ones(len(y_b), dtype=bool)), 'C': (X_c, y_cm, np.ones(len(y_cm), dtype=bool))}
    gold_meta = {'BBBP_test': 'bbbp_label', 'scaffold_test': 'b3db_label', 'reference_out': 'b3db_label'}
    rows = []
    preds_mean = {}
    for arm, (X, y, _m) in arms.items():
        for ev in ['BBBP_test', 'scaffold_test', 'reference_out']:
            evdf = sets[ev]
            gold = evdf[gold_meta[ev]].apply(lbl).values
            valid = ~np.isnan(gold)
            Xv = embs[ev][valid]
            gv = gold[valid].astype(int)
            rocs, prs, pmeans = ([], [], [])
            for seed in SEEDS:
                m = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed)
                m.fit(X, y)
                p = m.predict(Xv)
                rocs.append(roc_auc_score(gv, p))
                prs.append(average_precision_score(gv, p))
                pmeans.append(p)
            pm = np.mean(pmeans, axis=0)
            preds_mean[arm, ev] = pm
            rows.append({'arm': arm, 'eval': ev, 'n': int(valid.sum()), 'roc_auc_mean': round(float(np.mean(rocs)), 4), 'roc_auc_sd': round(float(np.std(rocs)), 4), 'pr_auc_mean': round(float(np.mean(prs)), 4), 'pr_auc_sd': round(float(np.std(prs)), 4)})
    rng = np.random.default_rng(20260816)
    sig = []
    for ev in ['BBBP_test', 'scaffold_test', 'reference_out']:
        evdf = sets[ev]
        gold = evdf[gold_meta[ev]].apply(lbl).values
        valid = np.nonzero(~np.isnan(gold))[0]
        for a1, a2 in [('A', 'B'), ('A', 'C'), ('B', 'C')]:
            p1 = preds_mean[a1, ev][:len(valid)]
            p2 = preds_mean[a2, ev][:len(valid)]
            gv = gold[valid].astype(int)
            a1s = roc_auc_score(gv, p1)
            a2s = roc_auc_score(gv, p2)
            d_obs = a1s - a2s
            n_perm = 1000
            cnt = 0
            for _ in range(n_perm):
                flip = rng.random(len(gv)) < 0.5
                if flip.sum() == 0 or flip.sum() == len(gv):
                    continue
                q1 = np.where(flip, p2, p1)
                q2 = np.where(flip, p1, p2)
                d = roc_auc_score(gv, q1) - roc_auc_score(gv, q2)
                if abs(d) >= abs(d_obs):
                    cnt += 1
            pval = (cnt + 1) / (n_perm + 1)
            sig.append({'eval': ev, 'comparison': f'{a1}-vs-{a2}', 'roc_a1': round(float(a1s), 4), 'roc_a2': round(float(a2s), 4), 'delta': round(float(d_obs), 4), 'p_perm': round(float(pval), 4)})
    sig = sorted(sig, key=lambda r: r['p_perm'])
    m9 = len(sig)
    for k, r in enumerate(sig):
        r['q_bh'] = round(min(1.0, r['p_perm'] * m9 / (k + 1)), 4)
    for k in range(m9 - 2, -1, -1):
        sig[k]['q_bh'] = round(min(sig[k]['q_bh'], sig[k + 1]['q_bh']), 4)
    main_table = pd.DataFrame(rows)
    sig_table = pd.DataFrame(sig)
    main_table.to_csv(OUT / 'main_table.csv', index=False, encoding='utf-8-sig')
    sig_table.to_csv(OUT / 'significance.csv', index=False, encoding='utf-8-sig')
    report = {'model': MODEL_ID, 'representation': 'ChemBERTa-77M-MTR mean-pooled last hidden state (384-d), frozen', 'head': 'XGBoost regressor (ranking), seeds 42/52/62', 'protocol': 'same train molecules (1234) and frozen views as stage7 A/B/C', 'n_train_labeled': int(len(y_a)), 'main_table': main_table.to_dict(orient='records'), 'significance': sig_table.to_dict(orient='records'), 'note': '复验目的为检验『标签×表示交互』在预训练表示层次是否成立；嵌入为冻结特征，未微调（CPU 约束，已在协议中固定）。'}
    with open(OUT / 'c4_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(main_table.to_string(index=False))
    print()
    print(sig_table.to_string(index=False))
    print('\nSAVED:', OUT)
if __name__ == '__main__':
    main()

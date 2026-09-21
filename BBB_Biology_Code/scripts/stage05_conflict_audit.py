from __future__ import annotations
from project_paths import project_root, project_path
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog('rdApp.*')
DATA = project_path('00_数据_原始')
OUT = project_path('05_复现工作区', 'results_stage0')
SR = SaltRemover.SaltRemover()

def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        m = SR.StripMol(m)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    except Exception:
        return None

def pik(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        m = SR.StripMol(m)
        Chem.RemoveStereochemistry(m)
        return MolToInchiKey(m).split('-')[0]
    except Exception:
        return None

def load():
    cls = pd.read_csv(DATA / 'B3DB_数据' / 'B3DB_classification.tsv', sep='\t', dtype=str)
    bbbp = pd.read_csv(DATA / 'BBBP.csv', dtype=str)
    bbbp.columns = [c.strip() for c in bbbp.columns]
    cls['lab'] = (cls['BBB+/BBB-'].str.strip() == 'BBB+').astype(int)
    bbbp['lab'] = bbbp['p_np'].astype(int)
    return (cls, bbbp)

def majority_agg(x):
    vc = x.value_counts()
    if len(vc) == 2 and vc.iloc[0] == vc.iloc[1]:
        return None
    return int(vc.idxmax())

def internal_conflict_keys(df, key):
    g = df.dropna(subset=[key]).groupby(key)['lab'].nunique()
    return set(g[g > 1].index)

def run_rule(cls, bbbp, key, b3db_mode, filter_mode):
    g = cls.dropna(subset=[key]).groupby(key)['lab']
    co = g.first() if b3db_mode == 'first' else g.agg(majority_agg)
    bo = bbbp.dropna(subset=[key]).groupby(key)['lab'].first()
    if filter_mode == 'consistent':
        drop = internal_conflict_keys(cls, key) | internal_conflict_keys(bbbp, key)
        co = co[~co.index.isin(drop)]
        bo = bo[~bo.index.isin(drop)]
    total_overlap = co.index.intersection(bo.index)
    tie = [k for k in total_overlap if pd.isna(co.loc[k])]
    resolved = [k for k in total_overlap if pd.notna(co.loc[k]) and pd.notna(bo.loc[k])]
    discord = [k for k in resolved if co.loc[k] != bo.loc[k]]
    np_ = int(sum((1 for k in discord if co.loc[k] == 1 and bo.loc[k] == 0)))
    pn = int(sum((1 for k in discord if co.loc[k] == 0 and bo.loc[k] == 1)))
    assert np_ + pn == len(discord), 'direction sum must equal discordance'
    n = np_ + pn
    if n > 0:
        kk = max(np_, pn)
        p = float(stats.binom.cdf(min(np_, pn), n, 0.5) + (1 - stats.binom.cdf(kk - 1, n, 0.5)))
    else:
        p = 1.0
    per_mol = []
    for k in discord:
        direction = 'np_01' if co.loc[k] == 1 and bo.loc[k] == 0 else 'pn_10'
        per_mol.append({'identity_key': k, 'direction': direction, 'b3db_label': co.loc[k], 'bbbp_label': bo.loc[k]})
    return {'identity': key, 'b3db_mode': b3db_mode, 'filter': filter_mode, 'total_overlap': len(total_overlap), 'resolved_overlap': len(resolved), 'unresolved_tie': len(tie), 'discordance': len(discord), 'np_01': np_, 'pn_10': pn, 'binom_p': p, 'per_molecule': per_mol}

def main():
    cls, bbbp = load()
    cls['c'] = cls['SMILES'].apply(canon)
    cls['p'] = cls['SMILES'].apply(pik)
    bbbp['c'] = bbbp['smiles'].apply(canon)
    bbbp['p'] = bbbp['smiles'].apply(pik)
    results = []
    for key in ('c', 'p'):
        for b3db_mode in ('first', 'majority'):
            for f in ('all', 'consistent'):
                results.append(run_rule(cls, bbbp, key, b3db_mode, f))
    ps = np.array([r['binom_p'] for r in results])
    n_comp = len(ps)
    order = np.argsort(ps)
    qvals = np.zeros(n_comp)
    for rank, idx in enumerate(order, start=1):
        qvals[idx] = ps[idx] * n_comp / rank
    q_sorted = np.zeros(n_comp)
    running = 1.0
    for i in range(n_comp - 1, -1, -1):
        idx = order[i]
        running = min(running, qvals[idx])
        q_sorted[idx] = running
    for r, q in zip(results, q_sorted):
        r['q_bh'] = float(min(q, 1.0))
    trace_rows = []
    for r in results:
        rule = f"{r['identity']}_{r['b3db_mode']}_{r['filter']}"
        for m in r.get('per_molecule', []):
            trace_rows.append({'rule': rule, 'identity_key': m['identity_key'], 'b3db_agg': r['b3db_mode'], 'filter': r['filter'], 'direction': m['direction'], 'b3db_label': m['b3db_label'], 'bbbp_label': m['bbbp_label']})
    OUT.mkdir(parents=True, exist_ok=True)
    if trace_rows:
        pd.DataFrame(trace_rows).to_csv(OUT / 'conflict_sensitivity_trace.csv', index=False, encoding='utf-8-sig')
    for r in results:
        r.pop('per_molecule', None)
    with open(OUT / 'conflict_sensitivity_audit.json', 'w', encoding='utf-8') as f:
        json.dump({'primary_rule': 'parent_consistent', 'rules': results, 'n_rules': len(results)}, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(OUT / 'conflict_sensitivity_audit.csv', index=False, encoding='utf-8-sig')
    print('=== 冲突敏感性审计（8 口径）===')
    for r in results:
        name = f"{r['identity']}_{r['b3db_mode']}_{r['filter']}"
        print(f"  {name:24s} tot={r['total_overlap']:>4} res={r['resolved_overlap']:>4} tie={r['unresolved_tie']:>3} disc={r['discordance']:>3} {r['np_01']}:{r['pn_10']} p={r['binom_p']:.4f} q={r['q_bh']:.4f}")
    print(f"保存: {OUT / 'conflict_sensitivity_audit.json'}")
if __name__ == '__main__':
    main()

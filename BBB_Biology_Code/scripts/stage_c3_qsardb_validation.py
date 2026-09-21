from __future__ import annotations
from project_paths import project_root, project_path
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey, MolFromInchi
RDLogger.DisableLog('rdApp.*')
ROOT = project_root()
QDB_DIR = ROOT / '00_数据_原始' / 'QSARDB_独立logBB'
CONSENSUS = ROOT / '05_复现工作区' / 'results_stage5' / 'data' / 'consensus_dataset.csv'
HEM_JSON = ROOT / '05_复现工作区' / 'results_stage6' / 'hem_model.json'
OUT = ROOT / '05_复现工作区' / 'results_c3'
SR = SaltRemover.SaltRemover()
NS = {'qdb': 'http://www.qsardb.org/QDB'}

def parse_qsardb(dataset: str) -> pd.DataFrame:
    base = QDB_DIR / dataset
    compounds = []
    tree = ET.parse(base / 'compounds' / 'compounds.xml')
    for comp in tree.getroot().iter(f"{{{NS['qdb']}}}Compound"):
        cid = comp.find(f"{{{NS['qdb']}}}Id").text.strip()
        inchi_el = comp.find(f"{{{NS['qdb']}}}InChI")
        inchi = inchi_el.text.strip() if inchi_el is not None and inchi_el.text else None
        compounds.append({'cid': cid, 'inchi': inchi})
    values_file = base / 'properties' / 'logBB' / 'values'
    rows = []
    for line in values_file.read_text(encoding='utf-8').splitlines()[1:]:
        if '\t' in line:
            cid, val = line.split('\t', 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            cid, val = (parts[0], parts[1])
        rows.append({'cid': cid.strip(), 'logbb': float(val)})
    df = pd.merge(pd.DataFrame(compounds), pd.DataFrame(rows), on='cid', how='inner')
    df['dataset'] = dataset

    def canon(inchi):
        if not inchi:
            return None
        try:
            m = MolFromInchi(inchi)
            if m is None:
                return None
            m = SR.StripMol(m)
            Chem.RemoveStereochemistry(m)
            return Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
        except Exception:
            return None

    def pik(inchi):
        if not inchi:
            return None
        try:
            m = MolFromInchi(inchi)
            if m is None:
                return None
            m = SR.StripMol(m)
            Chem.RemoveStereochemistry(m)
            return MolToInchiKey(m).split('-')[0]
        except Exception:
            return None
    df['canon'] = df['inchi'].apply(canon)
    df['pik'] = df['inchi'].apply(pik)
    return df

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qdb = pd.concat([parse_qsardb('zhao2006'), parse_qsardb('clark1999')], ignore_index=True)
    n_qdb = len(qdb)
    n_parsed = int(qdb['canon'].notna().sum())
    print(f"QSARDB 解析: 共 {n_qdb} 条 (zhao2006 {len(qdb[qdb.dataset == 'zhao2006'])}, clark1999 {len(qdb[qdb.dataset == 'clark1999'])}), 可解析 {n_parsed}")
    cons = pd.read_csv(CONSENSUS, dtype=str).fillna('')
    cons['canon_b3db'] = cons['canon_smiles_b3db'].fillna(cons['canon_smiles'])
    b3db_canon_map = cons.dropna(subset=['canon_b3db']).drop_duplicates('canon_b3db')
    b3db_pik_map = cons.dropna(subset=['parent_ik']).drop_duplicates('parent_ik')
    qdb_canon = qdb.dropna(subset=['canon']).drop_duplicates('canon')
    qdb_pik = qdb.dropna(subset=['pik']).drop_duplicates('pik')
    hit_canon = qdb_canon.merge(b3db_canon_map[['canon_b3db', 'parent_ik', 'b3db_label', 'bbbp_label', 'b3db_logbb_min', 'b3db_logbb_max', 'overlap', 'divergence']], left_on='canon', right_on='canon_b3db', how='inner')
    hit_pik = qdb_pik.merge(b3db_pik_map[['parent_ik', 'b3db_label', 'bbbp_label', 'b3db_logbb_min', 'b3db_logbb_max', 'overlap', 'divergence']], left_on='pik', right_on='parent_ik', how='inner')
    hit_canon = hit_canon.assign(match='canon')
    hit_pik = hit_pik.assign(match='pik')
    aligned = pd.concat([hit_canon, hit_pik], ignore_index=True)
    aligned = aligned.drop_duplicates(subset=['cid']).reset_index(drop=True)
    print(f'对齐到 B3DB 共识集: canon 命中 {len(hit_canon)}, pik 命中 {len(hit_pik)}, 合并去重 {len(aligned)}')
    if len(aligned) == 0:
        print('无可对齐样本')
        return
    cols = ['cid', 'dataset', 'inchi', 'canon', 'logbb', 'parent_ik', 'b3db_label', 'bbbp_label', 'b3db_logbb_min', 'b3db_logbb_max', 'overlap', 'divergence', 'match']
    aligned[cols].to_csv(OUT / 'qsardb_aligned.csv', index=False, encoding='utf-8-sig')
    a = aligned.copy()
    a['b3db_logbb_min'] = pd.to_numeric(a['b3db_logbb_min'], errors='coerce')
    a['b3db_logbb_max'] = pd.to_numeric(a['b3db_logbb_max'], errors='coerce')
    a['logbb'] = pd.to_numeric(a['logbb'], errors='coerce')
    a = a.dropna(subset=['b3db_logbb_min', 'b3db_logbb_max', 'logbb'])
    print(f'其中有 B3DB logBB 对照: {len(a)}')
    if len(a) > 0:
        a['mid'] = (a['b3db_logbb_min'] + a['b3db_logbb_max']) / 2
        a['d_qsar'] = (a['logbb'] - a['mid']).abs()
        a['within_b3db_range'] = ((a['logbb'] >= a['b3db_logbb_min']) & (a['logbb'] <= a['b3db_logbb_max'])).astype(int)
        r = {'n_aligned_qsardb_logbb': int(len(a)), 'within_b3db_range': int(a['within_b3db_range'].sum()), 'mean_abs_diff': float(a['d_qsar'].mean()), 'corr_pearson': float(np.corrcoef(a['logbb'], a['mid'])[0, 1]) if len(a) >= 3 else None}
        print(f"  within B3DB range: {r['within_b3db_range']}/{len(a)}")
        print(f"  mean |Δ| vs B3DB mid: {r['mean_abs_diff']:.3f}")
        if r['corr_pearson'] is not None:
            print(f"  Pearson(logBB_QSARDB, B3DB_mid): {r['corr_pearson']:.3f}")
        div = a[a['divergence'].astype(float) > 0]
        print(f'其中跨库分歧分子: {len(div)}')
        if len(div) > 0:
            a['b3db_label'] = pd.to_numeric(a['b3db_label'], errors='coerce')
            div['b3db_label'] = pd.to_numeric(div['b3db_label'], errors='coerce')
            pos = div[div['b3db_label'] == 1]
            neg = div[div['b3db_label'] == 0]
            pos_mean = pos['logbb'].mean() if len(pos) else float('nan')
            neg_mean = neg['logbb'].mean() if len(neg) else float('nan')
            print(f'  分歧分子中 B3DB 阳性的 QSARDB logBB 均值: {pos_mean:.3f} (n={len(pos)}) vs B3DB 阴性: {neg_mean:.3f} (n={len(neg)})')
            r['n_divergence_with_qsardb'] = int(len(div))
            r['b3db_pos_mean_logbb'] = float(pos_mean)
            r['b3db_neg_mean_logbb'] = float(neg_mean)
            r['b3db_pos_n'] = int(len(pos))
            r['b3db_neg_n'] = int(len(neg))
    hem_report = {'note': '全量共识集(含reference-out)重拟合 HEM，验证潜变量排序'}
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / '01_代码_各阶段'))
        from hem import LatentHEM, build_tensors
        cons_full = pd.read_csv(CONSENSUS, dtype=str).fillna('')
        T = build_tensors(cons_full, use_reference_out=True)
        n_hem = T['n']
        m = LatentHEM(n_hem, n_em_iters=20, m_epochs=30, m_lr=0.01, l2_prior=0.1)
        m.fit(T['b3db_idx'], T['b3db_labels'], bbbp_idx=T['bbbp_idx'], bbbp_labels=T['bbbp_labels'], logbb_idx=T['logbb_idx'], logbb_vals=T['logbb_vals'])
        cons_full['hem_latent'] = m.mu
        cons_full['hem_sigmoid'] = m.predict(np.arange(n_hem))
        al = aligned.merge(cons_full[['parent_ik', 'hem_latent', 'hem_sigmoid']], on='parent_ik', how='left')
        al = al.dropna(subset=['hem_latent'])
        al['logbb'] = pd.to_numeric(al['logbb'], errors='coerce')
        al = al.dropna(subset=['logbb'])
        n_hem_aligned = len(al)
        print(f'HEM 潜变量可比对样本: {n_hem_aligned}')
        if n_hem_aligned >= 3:
            corr = float(np.corrcoef(al['hem_sigmoid'], al['logbb'])[0, 1])
            r2 = float(corr ** 2)
            from scipy import stats
            rho = float(stats.spearmanr(al['hem_sigmoid'], al['logbb']).statistic)
            hem_report = {'n_samples': n_hem_aligned, 'corr_hem_sigmoid_vs_qsardb_logbb': corr, 'r2': r2, 'spearman_rho': rho, 'hem_latent_range': [float(al['hem_latent'].min()), float(al['hem_latent'].max())], 'note': 'HEM 在全量共识集(含reference-out, %d 分子)重拟合，后验 sigmoid 概率 vs 独立 QSARDB logBB 的相关性与排序' % n_hem}
            print(f'  Pearson(HEM sigmoid, QSARDB logBB): {corr:.3f} (R²={r2:.3f})')
            print(f'  Spearman rho: {rho:.3f}')
            al[['cid', 'dataset', 'parent_ik', 'logbb', 'hem_latent', 'hem_sigmoid']].to_csv(OUT / 'qsardb_hem_verification.csv', index=False, encoding='utf-8-sig')
    except Exception as e:
        import traceback
        hem_report = {'error': str(e), 'note': '全量 HEM 重拟合失败，回退到 stage7 labels_abc 对比'}
        traceback.print_exc()
        abc = ROOT / '05_复现工作区' / 'results_stage7' / 'data' / 'labels_abc.csv'
        if abc.exists():
            lab = pd.read_csv(abc, dtype=str).fillna('')
            lab['parent_ik'] = lab['parent_ik'].str.strip()
            al = aligned.merge(lab[['parent_ik', 'label_C_sigmoid_mu']], on='parent_ik', how='left')
            al = al.dropna(subset=['label_C_sigmoid_mu'])
            al['label_C_sigmoid_mu'] = al['label_C_sigmoid_mu'].astype(float)
            al['logbb'] = al['logbb'].astype(float)
            print(f'  [回退] HEM 潜变量可比对样本: {len(al)}')
            if len(al) >= 3:
                corr = float(np.corrcoef(al['label_C_sigmoid_mu'], al['logbb'])[0, 1])
                r2 = float(corr ** 2)
                hem_report = {'n_samples': int(len(al)), 'corr_hem_sigmoid_vs_qsardb_logbb': corr, 'r2': r2, 'note': '回退：stage7 训练集 HEM'}
                print(f'  Pearson(HEM sigmoid, QSARDB logBB): {corr:.3f} (R²={r2:.3f})')
                al[['cid', 'parent_ik', 'logbb', 'label_C_sigmoid_mu']].to_csv(OUT / 'qsardb_hem_verification.csv', index=False, encoding='utf-8-sig')
    report = {'qsardb_parsed': {'total': n_qdb, 'parsed': n_parsed}, 'aligned': {'canon': int(len(hit_canon)), 'pik': int(len(hit_pik)), 'total_unique': int(len(aligned))}, 'logbb_comparison': r if len(a) else None, 'hem_verification': hem_report}
    with open(OUT / 'stage_c3_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"报告保存: {OUT / 'stage_c3_report.json'}")
if __name__ == '__main__':
    main()

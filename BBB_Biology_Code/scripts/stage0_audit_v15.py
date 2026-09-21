from __future__ import annotations
from project_paths import project_root, project_path
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog('rdApp.*')
DATA_DIR = project_path('00_数据_原始')
B3DB_DIR = DATA_DIR / 'B3DB_数据'
OUTPUT_DIR = project_path('05_复现工作区', 'results_stage0')
LOGBB_THRESHOLD = -0.5
SREMOVER = SaltRemover.SaltRemover()

def canon_smiles(smi):
    if not isinstance(smi, str):
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = SREMOVER.StripMol(m)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
    except Exception:
        return None

def parent_inchikey(smi):
    if not isinstance(smi, str):
        return None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = SREMOVER.StripMol(m)
        Chem.RemoveStereochemistry(m)
        ik = MolToInchiKey(m)
        return ik.split('-')[0] if ik else None
    except Exception:
        return None

def load_b3db(fn):
    df = pd.read_csv(B3DB_DIR / fn, sep='\t', dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def binomial_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    hi = max(k, n - k)
    pval = stats.binom.cdf(lo, n, p) + (1.0 - stats.binom.cdf(hi - 1, n, p))
    return float(min(pval, 1.0))

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = {'BBBP.csv': DATA_DIR / 'BBBP.csv', 'B3DB_classification.tsv': B3DB_DIR / 'B3DB_classification.tsv', 'B3DB_regression.tsv': B3DB_DIR / 'B3DB_regression.tsv', 'B3DB_classification_external.tsv': B3DB_DIR / 'B3DB_classification_external.tsv'}
    input_check = {}
    for name, p in inputs.items():
        exists = p.exists()
        if exists:
            if p.suffix == '.tsv':
                rows = len(pd.read_csv(p, sep='\t', dtype=str))
            else:
                rows = len(pd.read_csv(p, dtype=str))
        else:
            rows = 0
        input_check[name] = {'exists': exists, 'rows': rows}
    report = {'input_check': input_check}
    cls = load_b3db('B3DB_classification.tsv')
    bbbp = pd.read_csv(DATA_DIR / 'BBBP.csv', dtype=str)
    bbbp.columns = [c.strip() for c in bbbp.columns]
    cls['canon'] = cls['SMILES'].apply(canon_smiles)
    bbbp['canon'] = bbbp['smiles'].apply(canon_smiles)
    cls_parse = int(cls['canon'].notna().sum())
    bbbp_parse = int(bbbp['canon'].notna().sum())
    report['parse_rate'] = {'B3DB_cls': {'parsed': cls_parse, 'total': len(cls), 'rate': cls_parse / len(cls)}, 'BBBP': {'parsed': bbbp_parse, 'total': len(bbbp), 'rate': bbbp_parse / len(bbbp)}}
    cls['label'] = (cls['BBB+/BBB-'].str.strip() == 'BBB+').astype(int)
    bbbp['label'] = bbbp['p_np'].astype(int)
    cls_one = cls.dropna(subset=['canon']).drop_duplicates('canon').set_index('canon')
    bbbp_one = bbbp.dropna(subset=['canon']).drop_duplicates('canon').set_index('canon')
    common = cls_one.index.intersection(bbbp_one.index)
    n_overlap = len(common)
    agree = int((cls_one.loc[common, 'label'] == bbbp_one.loc[common, 'label']).sum())
    n_divergence = n_overlap - agree
    bbbp_neg_b3db_pos = int(((bbbp_one.loc[common, 'label'] == 0) & (cls_one.loc[common, 'label'] == 1)).sum())
    bbbp_pos_b3db_neg = int(((bbbp_one.loc[common, 'label'] == 1) & (cls_one.loc[common, 'label'] == 0)).sum())
    p_binom = binomial_two_sided(max(bbbp_neg_b3db_pos, bbbp_pos_b3db_neg), n_divergence)
    report['divergence'] = {'overlap': int(n_overlap), 'agree': agree, 'divergence': int(n_divergence), 'divergence_rate': float(n_divergence / n_overlap) if n_overlap else 0.0, 'direction_bbbp_neg_b3db_pos': int(bbbp_neg_b3db_pos), 'direction_bbbp_pos_b3db_neg': int(bbbp_pos_b3db_neg), 'binomial_p': float(p_binom)}
    div_mask = cls_one.loc[common, 'label'] != bbbp_one.loc[common, 'label']
    div_canons = common[div_mask]
    div_rows = []
    for c in div_canons:
        div_rows.append({'canon_smiles': c, 'bbbp_name': bbbp_one.loc[c, 'name'] if 'name' in bbbp_one else '', 'bbbp_label': int(bbbp_one.loc[c, 'label']), 'b3db_label': int(cls_one.loc[c, 'label']), 'b3db_logBB': cls_one.loc[c, 'logBB'] if 'logBB' in cls_one else ''})
    div_df = pd.DataFrame(div_rows)
    div_df.to_csv(OUTPUT_DIR / 'divergence_molecules.csv', index=False, encoding='utf-8-sig')
    cls['refs'] = cls['reference'].fillna('').str.strip('|').str.split('|')
    cls['pik'] = cls['SMILES'].apply(parent_inchikey)
    mol_refs = cls.dropna(subset=['pik']).groupby('pik')['refs'].apply(lambda x: sorted({r for refs in x for r in refs if r.strip()}))
    n_unique = len(mol_refs)
    n_multi_mol = int((mol_refs.apply(len) >= 2).sum())
    allrefs = [r for refs in mol_refs for r in refs]
    ref_counts_mol = Counter(allrefs)
    report['source_structure'] = {'distinct_sources': len(ref_counts_mol), 'multi_source_molecules': n_multi_mol, 'total_molecules': n_unique, 'row_level_total': len(cls), 'source_size_range': [min(ref_counts_mol.values()), max(ref_counts_mol.values())], 'source_size_median': float(np.median(list(ref_counts_mol.values()))), 'note': f'Molecule-level counts (unique parent InChIKey). Row-level multi-source 4610 is ROW count; molecule-level is {n_multi_mol}.'}
    divergence_table = pd.DataFrame([{'canon_smiles': c, 'bbbp_label': int(bbbp_one.loc[c, 'label']), 'b3db_label': int(cls_one.loc[c, 'label']), 'note': 'canon-first sensitivity reference; primary = parent-consistent (core_evidence)'} for c in div_canons])
    divergence_table.to_csv(OUTPUT_DIR / 'table1_label_divergence.csv', index=False, encoding='utf-8-sig')
    direction_table = pd.DataFrame([{'direction': 'BBBP- 但 B3DB+', 'count': bbbp_neg_b3db_pos, 'note': 'canon-first sensitivity reference; primary = parent-consistent (core_evidence)'}, {'direction': 'BBBP+ 但 B3DB-', 'count': bbbp_pos_b3db_neg, 'note': 'canon-first sensitivity reference; primary = parent-consistent (core_evidence)'}])
    direction_table.to_csv(OUTPUT_DIR / 'table2_direction_test.csv', index=False, encoding='utf-8-sig')
    source_table = pd.DataFrame([{'reference': r, 'n_molecules': c, 'share': c / n_unique} for r, c in ref_counts_mol.most_common()])
    source_table.to_csv(OUTPUT_DIR / 'table3_source_redundancy.csv', index=False, encoding='utf-8-sig')
    nref_mol = mol_refs.apply(len)
    nref_counts = nref_mol.value_counts().sort_index()
    dist_rows = []
    for n in sorted(nref_counts.index):
        dist_rows.append({'n_sources': int(n), 'n_molecules': int(nref_counts[n]), 'share': float(nref_counts[n] / n_unique)})
    single = int((nref_mol == 1).sum())
    multi = int((nref_mol >= 2).sum())
    dist_table = pd.DataFrame(dist_rows)
    dist_table.to_csv(OUTPUT_DIR / 'table4_reference_distribution.csv', index=False, encoding='utf-8-sig')
    report['reference_distribution'] = {'single_source_molecules': single, 'single_source_share': single / n_unique, 'multi_source_molecules': multi, 'multi_source_share': multi / n_unique, 'note': 'Molecule-level (unique parent InChIKey); row-level differs.'}
    audit_path = OUTPUT_DIR / 'conflict_sensitivity_audit.json'
    core_evidence = None
    if audit_path.exists():
        with open(audit_path, encoding='utf-8') as f:
            audit = json.load(f)
        for r in audit.get('rules', []):
            if r['identity'] == 'p' and r['b3db_mode'] == 'first' and (r['filter'] == 'consistent'):
                core_evidence = {'primary_rule': 'parent_consistent', 'n_discordance': r['discordance'], 'direction_np_pn': f"{r['np_01']}:{r['pn_10']}", 'binom_p': r['binom_p'], 'q_bh': r['q_bh'], 'note': 'Pre-specified primary rule. first-label counts (58, 46:12) are sensitivity reference, not the core evidence.'}
                break
    report['core_evidence'] = core_evidence
    with open(OUTPUT_DIR / 'stage0_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print('=== 审计报告已保存 ===')
    if core_evidence:
        print(f"核心证据(parent-consistent): 分歧 {core_evidence['n_discordance']}, 方向 {core_evidence['direction_np_pn']}, p={core_evidence['binom_p']:.4f}")
if __name__ == '__main__':
    main()

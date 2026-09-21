from __future__ import annotations
from project_paths import project_root, project_path
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ROOT = project_root()
CONSENSUS = ROOT / '05_复现工作区' / 'results_stage5' / 'data' / 'consensus_dataset.csv'
REFMAP = ROOT / '05_复现工作区' / 'results_stage5' / 'data' / 'reference_map.csv'
ALIGNED = ROOT / '05_复现工作区' / 'results_c3' / 'qsardb_aligned.csv'
OUT05 = ROOT / '05_复现工作区' / 'results_e1'
OUT02 = ROOT / '02_结果' / 'results_e1'
N_BOOT = 2000
SEED = 20260816
TOL = 0.01

def refs_of(s):
    return set(re.findall('R\\d+', str(s)))

def concordance(df, label):
    a = df['logbb'].to_numpy(dtype=float)
    b = df['mid'].to_numpy(dtype=float)
    n = len(a)
    r = stats.pearsonr(a, b)[0] if n >= 3 else np.nan
    rho = stats.spearmanr(a, b).statistic if n >= 3 else np.nan
    mae = float(np.mean(np.abs(a - b)))
    diffs = a - b
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    loa_lo = bias - 1.96 * sd
    loa_hi = bias + 1.96 * sd
    rng = np.random.default_rng(SEED)
    clusters = df['parent_ik'].to_numpy()
    uniq = np.unique(clusters)
    r_boot, rho_boot, mae_boot, bias_boot = ([], [], [], [])
    for _ in range(N_BOOT):
        idx = rng.choice(len(uniq), size=len(uniq), replace=True)
        picked = np.concatenate([np.where(clusters == u)[0] for u in uniq[idx]])
        if len(picked) < 3:
            continue
        r_boot.append(stats.pearsonr(a[picked], b[picked])[0])
        rho_boot.append(stats.spearmanr(a[picked], b[picked]).statistic)
        mae_boot.append(float(np.mean(np.abs(a[picked] - b[picked]))))
        bias_boot.append(float(np.mean(a[picked] - b[picked])))

    def ci(v):
        v = np.asarray(v)
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {'subset': label, 'n': int(n), 'pearson_r': round(float(r), 4), 'pearson_ci': ci(r_boot), 'spearman_rho': round(float(rho), 4), 'spearman_ci': ci(rho_boot), 'mae': round(mae, 4), 'mae_ci': ci(mae_boot), 'bland_altman_bias': round(bias, 4), 'bias_ci': ci(bias_boot), 'loa': [round(loa_lo, 4), round(loa_hi, 4)]}

def main():
    for out in (OUT05, OUT02):
        out.mkdir(parents=True, exist_ok=True)
    cons = pd.read_csv(CONSENSUS, dtype=str).fillna('')
    rmap = pd.read_csv(REFMAP, dtype=str).fillna('')
    r19 = rmap[rmap.reference == '19']['source'].iloc[0][:80]
    r22 = rmap[rmap.reference == '22']['source'].iloc[0][:80]
    al = pd.read_csv(ALIGNED, dtype=str, encoding='utf-8-sig').fillna('')
    al['logbb'] = pd.to_numeric(al['logbb'], errors='coerce')
    al['b3db_logbb_min'] = pd.to_numeric(al['b3db_logbb_min'], errors='coerce')
    al['b3db_logbb_max'] = pd.to_numeric(al['b3db_logbb_max'], errors='coerce')
    al['mid'] = (al['b3db_logbb_min'] + al['b3db_logbb_max']) / 2
    m = al.merge(cons[['parent_ik', 'reference_list', 'b3db_label', 'b3db_logbb_list', 'b3db_logbb_refs']], on='parent_ik', how='left')
    m['refs'] = m['reference_list'].apply(refs_of)
    m['logbb_refs'] = m['b3db_logbb_refs'].apply(refs_of)
    comp = m.dropna(subset=['logbb', 'mid']).copy()
    comp['src_overlap'] = comp['refs'].apply(lambda s: 'R19' in s or 'R22' in s)
    comp['logbb_ref_overlap'] = comp['logbb_refs'].apply(lambda s: 'R19' in s or 'R22' in s)
    comp['dup_exact'] = comp['logbb'] == comp['mid']
    comp['dup_near'] = (comp['logbb'] - comp['mid']).abs() <= TOL
    comp['level'] = np.where(comp['src_overlap'], 'L3_source_overlap(R19/R22)', np.where(comp['dup_near'], 'L2_value_duplicate_unflagged(possible_transcription)', 'L1_unflagged_nonduplicate'))
    comp['category'] = np.where(comp['dup_exact'], 'exact_duplicate', np.where(comp['dup_near'], 'near_duplicate(tol=0.01)', np.where(comp['src_overlap'], 'molecule_overlap_only', 'disjoint_source')))
    l2 = comp[comp['level'] == 'L2_value_duplicate_unflagged(possible_transcription)']
    l2_flagged_in_logbb = int(l2['logbb_ref_overlap'].sum())
    prov_cols = ['cid', 'dataset', 'parent_ik', 'logbb', 'mid', 'src_overlap', 'logbb_ref_overlap', 'dup_exact', 'dup_near', 'category', 'level']
    prov = comp[prov_cols].copy()
    prov['paper_source'] = np.where(prov.dataset == 'zhao2006', 'Zhao2007 (R19)', 'Clark1999 (R22)')
    prov_out = prov.rename(columns={'logbb': 'qsardb_logbb', 'mid': 'b3db_logbb_mid', 'src_overlap': 'b3db_source_contains_R19_or_R22'})
    counts = prov.groupby(['dataset', 'category', 'level']).size().reset_index(name='n')
    res = {}
    for lvl, lab in [('L1_unflagged_nonduplicate', 'L1_unflagged_nonduplicate'), ('L2_value_duplicate_unflagged(possible_transcription)', 'L2_value_duplicate_unflagged'), ('L3_source_overlap(R19/R22)', 'L3_source_overlap')]:
        sub = comp[comp['level'] == lvl]
        if len(sub) >= 3:
            res[lab] = concordance(sub, lab)
        else:
            res[lab] = {'subset': lab, 'n': int(len(sub)), 'note': '样本不足，不作推断'}
    res['all_combined'] = concordance(comp, 'all_combined')
    report = {'finding': 'R19 = Zhao YH et al. 2007; R22 = Clark DE 1999：QSARDB zhao2006/clark1999 与 B3DB 的 R19/R22 为同一原始论文，原 C3「独立外部验证」存在同源循环风险。可比样本（n=281）按来源与值重复分为三层：L1 未追溯至 R19/R22 且无值级重复的子集、L2 值级重复但分子来源未标注（疑似转录）、L3 分子来源含 R19/R22。需要强调：L1 表示「未发现来源重叠」，不等于已证明严格独立；75 个 L1 分子仅用于来源限定的一致性分析。C3 主张降级为「跨资源一致性审计」（cross-resource concordance audit）。', 'r19_source': r19, 'r22_source': r22, 'provenance_counts': counts.to_dict(orient='records'), 'l2_flagged_in_logbb_refs': l2_flagged_in_logbb, 'l2_n': int(len(l2)), 'concordance': res, 'n_boot': N_BOOT, 'seed': SEED, 'note': '分子簇 bootstrap：以 parent_ik 为簇（同一分子的重复测量不拆分），2000 次有放回重采样分子簇后重算统计量，取 2.5/97.5 分位数为 95% CI。L1 的 r 仅描述该来源限定子集内 QSARDB 与 B3DB logBB 的一致性，不得解读为外部独立验证成功。'}
    for out in (OUT05, OUT02):
        prov_out.to_csv(out / 'provenance_table.csv', index=False, encoding='utf-8-sig')
        comp[['cid', 'parent_ik', 'logbb', 'mid', 'src_overlap']].to_csv(out / 'blantaltman.csv', index=False, encoding='utf-8-sig')
        with open(out / 'concordance_audit.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:2400])
if __name__ == '__main__':
    main()

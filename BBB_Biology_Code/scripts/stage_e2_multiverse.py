from __future__ import annotations
from project_paths import project_root, project_path
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog('rdApp.*')
DATA = project_path('00_数据_原始')
OUT05 = project_path('05_复现工作区') / 'results_e2'
OUT02 = project_path('02_结果') / 'results_e2'
SR = SaltRemover.SaltRemover()
N_BOOT = 2000
SEED = 20260816
GLOBAL_RULE = ('p', 'first', 'consistent')

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
        ik = MolToInchiKey(m)
        return ik.split('-')[0] if ik else None
    except Exception:
        return None

def majority_agg(x):
    vc = x.value_counts()
    if len(vc) == 2 and vc.iloc[0] == vc.iloc[1]:
        return None
    return int(vc.idxmax())

def build_rule_tables(key):
    cg = cls.dropna(subset=[key]).groupby(key)['lab']
    co_first = cg.first()
    co_maj = cg.agg(majority_agg)
    bo = bbbp.dropna(subset=[key]).groupby(key)['lab'].first()
    conflict_c = set(cg.nunique()[cg.nunique() > 1].index)
    conflict_b = set(bbbp.dropna(subset=[key]).groupby(key)['lab'].nunique()[bbbp.dropna(subset=[key]).groupby(key)['lab'].nunique() > 1].index)
    overlap = co_first.index.intersection(bo.index)
    df = pd.DataFrame({'co_first': co_first.loc[overlap], 'co_maj': co_maj.loc[overlap], 'bo': bo.loc[overlap]}).fillna({'co_maj': -1})
    df['conflict_c'] = [k in conflict_c for k in overlap]
    df['conflict_b'] = [k in conflict_b for k in overlap]
    return (df, list(overlap))

def audit_from_indices(df, idx):
    sub = df.iloc[idx]
    out = {}
    for mode in ('first', 'majority'):
        co = sub['co_first'].to_numpy() if mode == 'first' else sub['co_maj'].to_numpy()
        valid = co >= 0
        for filt in ('all', 'consistent'):
            v = valid.copy()
            if filt == 'consistent':
                v = v & ~sub['conflict_c'].to_numpy() & ~sub['conflict_b'].to_numpy()
            cc = co[v]
            bb = sub['bo'].to_numpy()[v]
            disc = cc != bb
            n01 = int(np.sum(disc & (cc == 1) & (bb == 0)))
            n10 = int(np.sum(disc & (cc == 0) & (bb == 1)))
            out[mode, filt] = (n01, n10)
    return out

def main():
    global cls, bbbp
    cls = pd.read_csv(DATA / 'B3DB_数据' / 'B3DB_classification.tsv', sep='\t', dtype=str)
    bbbp = pd.read_csv(DATA / 'BBBP.csv', dtype=str)
    bbbp.columns = [c.strip() for c in bbbp.columns]
    cls['lab'] = (cls['BBB+/BBB-'].str.strip() == 'BBB+').astype(int)
    bbbp['lab'] = bbbp['p_np'].astype(int)
    cls['c'] = cls['SMILES'].apply(canon)
    cls['p'] = cls['SMILES'].apply(pik)
    bbbp['c'] = bbbp['smiles'].apply(canon)
    bbbp['p'] = bbbp['smiles'].apply(pik)
    tables = {}
    for key in ('p', 'c'):
        tables[key] = build_rule_tables(key)
    df_p, keys_p = tables['p']
    df_c, keys_c = tables['c']
    canon_by_pik_b3 = cls.dropna(subset=['c', 'p']).groupby('p')['c'].unique()
    canon_by_pik_b = bbbp.dropna(subset=['c', 'p']).groupby('p')['c'].unique()
    overlap_piks = set(keys_p)
    pik_to_canons = {}
    for k in overlap_piks:
        s = set(canon_by_pik_b3.get(k, [])) & set(canon_by_pik_b.get(k, []))
        if s:
            pik_to_canons[k] = s
    canon_key_to_idx = {k: i for i, k in enumerate(keys_c)}
    pik_canon_idx = {k: [canon_key_to_idx[c] for c in s if c in canon_key_to_idx] for k, s in pik_to_canons.items()}
    pik_list = list(keys_p)
    pik_pos = {k: i for i, k in enumerate(pik_list)}
    rng = np.random.default_rng(SEED)
    n = len(pik_list)
    rule_names = []
    for key in ('p', 'c'):
        for mode in ('first', 'majority'):
            for filt in ('all', 'consistent'):
                rule_names.append((key, mode, filt))
    boot = {rn: [] for rn in rule_names}
    for b in range(N_BOOT):
        idx = rng.choice(n, size=n, replace=True)
        for (mode, filt), (n01, n10) in audit_from_indices(df_p, idx).items():
            boot['p', mode, filt].append((n01, n10))
        c_idx = np.concatenate([pik_canon_idx[pik_list[i]] for i in idx if pik_list[i] in pik_canon_idx]) if any((pik_list[i] in pik_canon_idx for i in idx)) else np.array([], dtype=int)
        if len(c_idx):
            for (mode, filt), (n01, n10) in audit_from_indices(df_c, c_idx).items():
                boot['c', mode, filt].append((n01, n10))
        else:
            for mode, filt in [('first', 'all'), ('first', 'consistent'), ('majority', 'all'), ('majority', 'consistent')]:
                boot['c', mode, filt].append((0, 0))
        if (b + 1) % 500 == 0:
            print('bootstrap %d/%d' % (b + 1, N_BOOT), flush=True)
    rule_effects = []
    forest_rows = []
    g_obs_a, g_obs_b = audit_from_indices(df_p, np.arange(len(df_p)))[GLOBAL_RULE[1], GLOBAL_RULE[2]]
    g_obs_rho = g_obs_a / (g_obs_a + g_obs_b) if g_obs_a + g_obs_b else np.nan
    for rn in rule_names:
        arr = boot[rn]
        rho = np.array([a / (a + b_) if a + b_ > 0 else np.nan for a, b_ in arr])
        rho = rho[~np.isnan(rho)]
        if rn[0] == 'p':
            a, b_ = audit_from_indices(df_p, np.arange(len(df_p)))[rn[1], rn[2]]
        else:
            a, b_ = audit_from_indices(df_c, np.arange(len(df_c)))[rn[1], rn[2]]
        obs_rho = a / (a + b_) if a + b_ else np.nan
        mean_boot = float(np.mean(rho))
        lo = float(np.percentile(rho, 0.3125))
        hi = float(np.percentile(rho, 99.6875))
        lo95 = float(np.percentile(rho, 2.5))
        hi95 = float(np.percentile(rho, 97.5))
        rule_effects.append({'rule': '%s_%s_%s' % rn, 'identity': rn[0], 'b3db_mode': rn[1], 'filter': rn[2], 'observed_n01': int(a), 'observed_n10': int(b_), 'observed_rho': round(obs_rho, 4), 'bootstrap_mean_rho': round(mean_boot, 4), 'ci95': [round(lo95, 4), round(hi95, 4)], 'simultaneous_ci95_bonf_two_sided': [round(lo, 4), round(hi, 4)], 'n_boot_valid': int(len(rho))})
        forest_rows.append({'rule': '%s_%s_%s' % rn, 'observed_n01': int(a), 'observed_n10': int(b_), 'observed_rho': round(obs_rho, 4), 'mean_boot': round(mean_boot, 4), 'ci95_lo': round(lo95, 4), 'ci95_hi': round(hi95, 4), 'sim_lo': round(lo, 4), 'sim_hi': round(hi, 4)})
    g_arr = boot[GLOBAL_RULE]
    g_rho = np.array([a / (a + b_) if a + b_ > 0 else np.nan for a, b_ in g_arr])
    g_rho = g_rho[~np.isnan(g_rho)]
    g_lo, g_hi = (float(np.percentile(g_rho, 2.5)), float(np.percentile(g_rho, 97.5)))
    global_effect = {'rule': '%s_%s_%s' % GLOBAL_RULE, 'statistic': 'rho=n01/(n01+n10)', 'observed_rho': round(float(g_obs_rho), 4), 'bootstrap_mean': round(float(np.mean(g_rho)), 4), 'ci95': [round(g_lo, 4), round(g_hi, 4)], 'excludes_null_0.5': bool(g_lo > 0.5), 'conclusion': '方向证据在依赖重采样下稳健（CI 不跨 0.5）' if g_lo > 0.5 else '仅可描述为方向一致性（CI 跨 0.5）', 'n_boot_valid': int(len(g_rho))}
    report = {'design': {'resampling_unit': 'parent InChIKey (重叠分子簇)', 'n_clusters': n, 'n_boot': N_BOOT, 'seed': SEED, 'global_effect_prespecified': '%s_%s_%s, rho' % GLOBAL_RULE, 'sign_test_removed': True, 'bonferroni_simultaneous_ci': '0.3125%/99.6875% 分位数（双侧，8 口径，α=0.05）', 'note': '符号检验 P=0.0078（将 8 口径视为独立试验）已删除。重采样以母体分子为簇，簇内保留全部来源与重复测量状态；每次重采样完整重算 8 口径。规则级同时 95% CI 用 Bonferroni 双侧分位数（0.3125%/99.6875%）。'}, 'global_effect': global_effect, 'rule_effects': rule_effects}
    for out in (OUT05, OUT02):
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'multiverse_audit.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        pd.DataFrame(rule_effects).to_csv(out / 'rule_effects.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(forest_rows).to_csv(out / 'forest_source.csv', index=False, encoding='utf-8-sig')
    print('=== 全局主口径效应 ===')
    print(json.dumps(global_effect, ensure_ascii=False, indent=2))
    print()
    print('=== 规则级效应（同时 CI）===')
    for r in rule_effects:
        print('  %-22s n01:n10=%d:%d rho=%.3f 95CI=[%.3f,%.3f] simCI=[%.3f,%.3f]' % (r['rule'], r['observed_n01'], r['observed_n10'], r['observed_rho'], r['ci95'][0], r['ci95'][1], r['simultaneous_ci95_bonf_two_sided'][0], r['simultaneous_ci95_bonf_two_sided'][1]))
if __name__ == '__main__':
    main()

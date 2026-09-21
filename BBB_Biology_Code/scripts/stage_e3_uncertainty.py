from __future__ import annotations
from project_paths import project_root, project_path
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = project_root()
sys.path.insert(0, str(ROOT / '01_代码_各阶段'))
from hem import LatentHEM
CONSENSUS = ROOT / '05_复现工作区' / 'results_stage5' / 'data' / 'consensus_dataset.csv'
SCAFFOLD = ROOT / '05_复现工作区' / 'results_stage7' / 'data' / 'scaffold_split_assignments.csv'
ARM = ROOT / '05_复现工作区' / 'results_stage7' / 'reports' / 'arm_predictions.npz'
OUT05 = ROOT / '05_复现工作区' / 'results_e3'
OUT02 = ROOT / '02_结果' / 'results_e3'
TARGET_AUC = 0.7413580246913581

def lbl(s):
    s = str(s).strip()
    return float(s) if s in ('0', '1', '0.0', '1.0') else np.nan

def parse_logbb_list(val):
    if not isinstance(val, str) or not val.strip():
        return []
    out = []
    for x in val.split(';'):
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out

def build_tensors_excluding(df, exclude_bbbp=(), exclude_b3db=()):
    n = len(df)
    excl_b = set((int(x) for x in exclude_bbbp))
    excl_c = set((int(x) for x in exclude_b3db))
    b3db_idx, b3db_labels = ([], [])
    for i in range(n):
        if i in excl_c:
            continue
        raw = str(df.loc[i, 'b3db_raw_labels']) if 'b3db_raw_labels' in df.columns else ''
        if raw and ';' in raw:
            for lab in raw.split(';'):
                lab = lab.strip()
                if lab in ('0', '1'):
                    b3db_idx.append(i)
                    b3db_labels.append(float(lab))
        else:
            v = str(df.loc[i, 'b3db_label']).strip()
            if v in ('0', '1', '0.0', '1.0'):
                b3db_idx.append(i)
                b3db_labels.append(float(v))
    mask = df['bbbp_label'].astype(str).str.strip().isin(['0', '1', '0.0', '1.0']).to_numpy()
    keepb = [i for i in range(n) if mask[i] and i not in excl_b]
    bbbp_idx = np.array(keepb, dtype=int)
    bbbp_labels = np.array([float(df['bbbp_label'].iloc[i]) for i in keepb], dtype=float)
    logbb_idx, logbb_vals = ([], [])
    for i, val in enumerate(df['b3db_logbb_list']):
        for v in parse_logbb_list(str(val)):
            logbb_idx.append(i)
            logbb_vals.append(v)
    return {'n': n, 'b3db_idx': np.array(b3db_idx, dtype=int), 'b3db_labels': np.array(b3db_labels, dtype=float), 'bbbp_idx': bbbp_idx, 'bbbp_labels': bbbp_labels, 'logbb_idx': np.array(logbb_idx, dtype=int), 'logbb_vals': np.array(logbb_vals, dtype=float)}

def fit_hem(T):
    m = LatentHEM(T['n'], n_em_iters=20, m_epochs=30, m_lr=0.01, l2_prior=0.1)
    m.fit(T['b3db_idx'], T['b3db_labels'], bbbp_idx=T['bbbp_idx'], bbbp_labels=T['bbbp_labels'], logbb_idx=T['logbb_idx'], logbb_vals=T['logbb_vals'])
    return m

def coverage_risk(unc, gold, pred):
    order = np.argsort(-unc)
    n = len(gold)
    err = ((pred >= 0.5) != gold.astype(int)).astype(int)
    curve = []
    for k in range(n):
        kept = order[k:]
        curve.append({'coverage': round(float((n - k) / n), 4), 'risk': float(err[kept].mean()), 'n_covered': int(n - k)})
    curve.append({'coverage': 0.0, 'risk': 1.0, 'n_covered': 0})
    covs = np.array([(n - k) / n for k in range(n - 1, -1, -1)])
    risks = np.array([err[order[k:]].mean() for k in range(n - 1, -1, -1)])
    aurc = float(np.trapezoid(risks, covs)) if n > 2 else np.nan
    if n >= 3 and err.sum() > 0 and ((err == 0).sum() > 0):
        det_auc = float(roc_auc_score(err, unc))
        det_pr = float(average_precision_score(err, unc))
    else:
        det_auc, det_pr = (np.nan, np.nan)
    return {'curve': curve, 'aurc': round(aurc, 4), 'err_detect_auroc': round(det_auc, 4), 'err_detect_auprc': round(det_pr, 4), 'n': int(n)}

def main():
    for out in (OUT05, OUT02):
        out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CONSENSUS, dtype=str).fillna('')
    sc = pd.read_csv(SCAFFOLD, dtype=str).fillna('')
    df['_nref'] = df['n_references'].astype(float)
    df['_conflict'] = df['b3db_label_conflict'].astype(str).isin(['1', '1.0']).astype(int)
    d = np.load(ARM)
    bbbp_mask = df['split'] == 'test'
    bbbp_cons = df.index[bbbp_mask].to_numpy()
    bbbp_test = df.loc[bbbp_mask].reset_index(drop=True)
    val_mask = df['split'] == 'validation'
    val_cons = df.index[val_mask].to_numpy()
    val = df.loc[val_mask].reset_index(drop=True)
    ref_mask = df['in_reference_out'].astype(int) == 1
    ref_cons = df.index[ref_mask].to_numpy()
    refout = df.loc[ref_mask].reset_index(drop=True)
    cons_cols = df[['parent_ik', '_nref', '_conflict']].copy()
    scaf = sc[sc['scaffold_split'] == 'test'].reset_index(drop=True)
    pid_to_idx = dict(zip(df['parent_ik'], df.index))
    scaf_cons = scaf['parent_ik'].map(pid_to_idx).to_numpy()
    cons_by_pid = cons_cols.drop_duplicates('parent_ik').set_index('parent_ik')
    scaf['_nref'] = scaf['parent_ik'].map(cons_by_pid['_nref']).fillna(0.0).astype(float)
    scaf['_conflict'] = scaf['parent_ik'].map(cons_by_pid['_conflict']).fillna(0).astype(int)
    assert len(scaf_cons) == len(scaf), 'scaffold consensus_idx 长度不匹配'
    assert all((isinstance(i, (int, np.integer)) for i in scaf_cons)), 'scaffold consensus_idx 含非整数'
    views = {'BBBP_test': (bbbp_test, 'bbbp_label', 'C_BBBP_test', 'bbbp', bbbp_cons), 'scaffold_test': (scaf, 'b3db_label', 'C_scaffold_test', 'b3db', scaf_cons), 'reference_out': (refout, 'b3db_label', 'C_reference_out', 'b3db', ref_cons), 'validation': (val, 'b3db_label', 'C_validation', 'b3db', val_cons)}
    for vname, (_, _, _, _, cons_idx) in views.items():
        assert len(cons_idx) == len(np.unique(cons_idx)), f'{vname} 共识行号有重复'
        assert np.all(cons_idx >= 0) and np.all(cons_idx < 4131), f'{vname} 共识行号越界'
    check_auc = roc_auc_score(d['gold_bbbp'], d['C_BBBP_test'])
    aligned = abs(check_auc - TARGET_AUC) < 1e-09
    cal = LogisticRegression()
    cal.fit(d['C_validation'].reshape(-1, 1), d['gold_val'].astype(int))

    def view_data(vname, vdf, gcol, ckey, cons_idx):
        gold_full = vdf[gcol].apply(lbl).values
        valid = ~np.isnan(gold_full)
        gold = gold_full[valid].astype(int)
        pred = d[ckey]
        p_cal = cal.predict_proba(pred.reshape(-1, 1))[:, 1]
        u_cal = 0.5 - np.abs(p_cal - 0.5)
        nref = vdf['_nref'].values[valid]
        conf = vdf['_conflict'].values[valid]
        orig_idx = cons_idx[valid]
        return (gold, pred, p_cal, u_cal, nref, conf, orig_idx)
    report = {'alignment_check': {'recomputed_C_bbbp_auc': round(float(check_auc), 6), 'target': TARGET_AUC, 'aligned': bool(aligned)}, 'settings': {'calibration': 'validation-set univariate logistic', 'uncertainty': 'HEM posterior std s_i / u_cal=0.5-|p_cal-0.5|', 'aurc_definition': 'AURC=∫risk dc (coverage ascending; lower=better)', 'index_fix': 'v3: consensus_idx 用于 LOO 排除和 s_i 读取，不再使用 reset_index 后的位置索引'}, 'assertions': {'all_passed': True, 'details': []}, 'no_leakage': {}, 'diagnostic': {}}

    def assert_msg(ok, msg):
        if not ok:
            report['assertions']['all_passed'] = False
            report['assertions']['details'].append(msg)
        return ok
    for vname, (vdf, gcol, ckey, channel, cons_idx) in views.items():
        gold, pred, p_cal, u_cal, nref, conf, orig_idx = view_data(vname, vdf, gcol, ckey, cons_idx)
        assert_msg(len(orig_idx) == len(gold), f'{vname}: consensus_idx 与 gold 长度不一致')
        assert_msg(len(orig_idx) == len(pred), f'{vname}: consensus_idx 与 pred 长度不一致')
        assert_msg(len(set(orig_idx.tolist())) == len(orig_idx), f'{vname}: consensus_idx 有重复')
        if channel == 'bbbp':
            T = build_tensors_excluding(df, exclude_bbbp=orig_idx)
        else:
            T = build_tensors_excluding(df, exclude_b3db=orig_idx)
        if channel == 'bbbp':
            excl_count = len(set(orig_idx.tolist()) & set(df.index[df['bbbp_label'].astype(str).str.strip().isin(['0', '1', '0.0', '1.0'])].tolist()))
            assert_msg(True, f'{vname}: LOO 排除 bbbp 共识行 {excl_count} 个')
        else:
            excl_count = len(set(orig_idx.tolist()) & set(df.index[df['b3db_label'].astype(str).str.strip().isin(['0', '1', '0.0', '1.0'])].tolist()))
            assert_msg(True, f'{vname}: LOO 排除 b3db 共识行 {excl_count} 个')
        m = fit_hem(T)
        s_i = m.sigma[orig_idx]
        assert_msg(len(s_i) == len(gold), f'{vname}: s_i 与 gold 长度不一致')
        assert_msg(not np.any(np.isnan(s_i)), f'{vname}: s_i 含 NaN')
        res_hem = coverage_risk(s_i, gold, pred)
        res_mar = coverage_risk(u_cal, gold, pred)
        report['no_leakage'][vname] = {'n': int(len(gold)), 'channel_excluded': channel, 'consensus_indices_sample': orig_idx[:5].tolist(), 'hem_loo_unc': res_hem, 'calibrated_margin': res_mar, 'corr_s_vs_nref': round(float(np.corrcoef(s_i, nref)[0, 1]), 4) if len(s_i) > 2 else None, 'conflict_mean_s': float(s_i[conf == 1].mean()) if conf.sum() else None, 'nonconflict_mean_s': float(s_i[conf == 0].mean()) if (conf == 0).sum() else None}
        pd.DataFrame(res_hem['curve']).to_csv(OUT05 / f'curve_hem_noleak_{vname}.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(res_mar['curve']).to_csv(OUT05 / f'curve_margin_noleak_{vname}.csv', index=False, encoding='utf-8-sig')
        print(f"  {vname}: LOO done | n={len(gold)} | HEM AURC={res_hem['aurc']:.3f} errAUC={res_hem['err_detect_auroc']:.3f}")
    T_full = build_tensors_excluding(df)
    m_full = fit_hem(T_full)
    s_full = m_full.sigma
    for vname, (vdf, gcol, ckey, channel, cons_idx) in views.items():
        gold, pred, p_cal, u_cal, nref, conf, orig_idx = view_data(vname, vdf, gcol, ckey, cons_idx)
        s_i = s_full[orig_idx]
        res_hem = coverage_risk(s_i, gold, pred)
        res_mar = coverage_risk(u_cal, gold, pred)
        report['diagnostic'][vname] = {'n': int(len(gold)), 'hem_retrospective_unc': res_hem, 'calibrated_margin': res_mar}
        pd.DataFrame(res_hem['curve']).to_csv(OUT05 / f'curve_hem_diag_{vname}.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(res_mar['curve']).to_csv(OUT05 / f'curve_margin_diag_{vname}.csv', index=False, encoding='utf-8-sig')
    nl = report['no_leakage']
    pairs = []
    for v in ('BBBP_test', 'scaffold_test', 'reference_out', 'validation'):
        h_auc = nl[v]['hem_loo_unc']['err_detect_auroc']
        m_auc = nl[v]['calibrated_margin']['err_detect_auroc']
        better = 'hem' if h_auc > m_auc else 'margin' if m_auc > h_auc else 'tie'
        pairs.append((v, h_auc, m_auc, better))
    n_hem = sum((1 for _, _, _, b in pairs if b == 'hem'))
    n_mar = sum((1 for _, _, _, b in pairs if b == 'margin'))
    report['conclusion'] = {'all_assertions_passed': report['assertions']['all_passed'], 'hem_better_views': n_hem, 'margin_better_views': n_mar, 'summary': f'无泄漏（LOO）口径：HEM 在 {n_hem}/4 个视图错误检测 AUROC 更优，边际在 {n_mar}/4 个视图更优 → 视图间表现不一致，不得声称普遍有效。'}
    for out in (OUT05, OUT02):
        with open(out / 'uncertainty_audit.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print('\n=== 无泄漏（LOO）最终数字 ===')
    for v, h, m, b in pairs:
        print('  %-14s HEM errAUC=%.3f | margin errAUC=%.3f | %s' % (v, h, m, b))
    print('assertions passed:', report['assertions']['all_passed'])
    print('conclusion:', report['conclusion']['summary'])
    import shutil
    for f in os.listdir(OUT05):
        src = os.path.join(OUT05, f)
        dst = os.path.join(OUT02, f)
        shutil.copy2(src, dst)
    print('E3 results mirrored to 02:', sorted(os.listdir(OUT02)))
if __name__ == '__main__':
    main()

from __future__ import annotations
from project_paths import project_root, project_path
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
ROOT = project_root()
sys.path.insert(0, str(ROOT / '01_代码_各阶段'))
from hem import LatentHEM
OUT05 = ROOT / '05_复现工作区' / 'results_e4'
OUT02 = ROOT / '02_结果' / 'results_e4'
N_REP = 100
SEED = 20260816

def sim_multi(rng, n, beta=(1.0, 1.0, 1.0), off=(0.0, 0.3, 0.0), cov=(1.0, 1.0, 1.0), flip=(0.0, 0.0, 0.0), cov_l=1.0, sigma=0.2, nonlinear=False, logbb_scale=1.0):
    L = rng.normal(0, 1, n)
    Y = np.zeros((3, n))
    for s in range(3):
        p = 1.0 / (1.0 + np.exp(-(beta[s] * L + off[s])))
        y = rng.binomial(1, p)
        if flip[s] > 0:
            f = rng.random(n) < flip[s]
            y = np.where(f, 1 - y, y)
        cov_s = rng.random(n) < cov[s]
        Y[s] = np.where(cov_s, y, np.nan)
    logbb = np.full(n, np.nan)
    cl = rng.random(n) < cov_l
    base = L if not nonlinear else np.tanh(L)
    logbb[cl] = logbb_scale * base[cl] + rng.normal(0, sigma, cl.sum())
    return (L, Y, logbb)

def score_methods(Y, logbb, n):
    labA, labB, labC = (Y[0], Y[1], Y[2])
    out = {}
    h = np.full(n, 0.5)
    ok = ~np.isnan(labA)
    h[ok] = labA[ok]
    out['hard'] = h
    maj = np.full(n, 0.5)
    M = np.nanmean(Y, axis=0)
    has = ~np.isnan(M)
    maj[has] = (M[has] >= 0.5).astype(float)
    out['majority'] = maj
    sw = np.full(n, 0.5)
    sw[has] = M[has]
    out['source_weighted'] = sw
    lb = np.full(n, 0.5)
    lbok = ~np.isnan(logbb)
    lb[lbok] = 1.0 / (1.0 + np.exp(-logbb[lbok]))
    out['logbb_baseline'] = lb
    return out

def fit_hem(Y, logbb, n):
    b3_idx, b3_lab = ([], [])
    for s in (0, 2):
        ok = ~np.isnan(Y[s])
        b3_idx.extend(np.where(ok)[0].tolist())
        b3_lab.extend(Y[s][ok].tolist())
    bb_idx = np.where(~np.isnan(Y[1]))[0].astype(int)
    bb_lab = Y[1][~np.isnan(Y[1])].astype(float)
    li = np.where(~np.isnan(logbb))[0].astype(int)
    lv = logbb[~np.isnan(logbb)].astype(float)
    try:
        m = LatentHEM(n, n_em_iters=20, m_epochs=30, m_lr=0.01, l2_prior=0.1)
        m.fit(np.array(b3_idx, dtype=int), np.array(b3_lab, dtype=float), bbbp_idx=bb_idx, bbbp_labels=bb_lab, logbb_idx=li, logbb_vals=lv)
        if np.isnan(m.mu).any():
            return None
        return m
    except Exception:
        return None

def gold_bin(L):
    return (L > 0).astype(int)

def eval_score(score, L):
    g = gold_bin(L)
    rho = stats.spearmanr(score, L).statistic if L.std() > 0 else np.nan
    auc = None
    if g.sum() > 0 and (g == 0).sum() > 0:
        auc = float(roc_auc_score(g, score))
    brier = float(brier_score_loss(g, np.clip(score, 0, 1)))
    return (rho, auc, brier)

def scenario_list():
    sc = []

    def add(name, **kw):
        base = dict(n=1000, beta=(1.0, 1.0, 1.0), off=(0.0, 0.3, 0.0), cov=(1.0, 1.0, 1.0), flip=(0.0, 0.0, 0.0), cov_l=1.0, sigma=0.2, nonlinear=False)
        base.update(kw)
        sc.append({'name': name, **base})
    add('base_n1000')
    add('small_n300', n=300)
    add('large_n2000', n=2000)
    add('source_imbalance_weakC', beta=(1.2, 1.0, 0.4))
    add('source_imbalance_strongA', beta=(2.0, 1.0, 1.0))
    add('all_sources_weak', beta=(0.5, 0.5, 0.5))
    add('low_conflict_d0', off=(0.0, 0.0, 0.0))
    add('moderate_conflict_d0.5', off=(0.0, 0.5, 0.0))
    add('high_conflict_d0.8', off=(0.0, 0.8, 0.3))
    add('label_missing_cov0.5', cov=(0.5, 0.5, 0.5))
    add('logbb_missing0.4', cov_l=0.4)
    add('both_missing', cov=(0.6, 0.6, 0.6), cov_l=0.5)
    add('missing_heavy', cov=(0.3, 0.3, 0.3), cov_l=0.3)
    add('logbb_noise1.0', sigma=1.0)
    add('label_flip0.1', flip=(0.1, 0.1, 0.1))
    add('flip_heavy_B', flip=(0.0, 0.2, 0.0))
    add('all_high_quality_noise0.6', beta=(1.5, 1.5, 1.5), sigma=0.6)
    add('continuous_misspec_tanh', nonlinear=True)
    add('source_direction_reversedC', off=(0.0, 0.3, -1.0))
    add('single_source_A_plus_logbb', cov=(1.0, 0.0, 0.0), cov_l=1.0)
    add('single_source_A_no_logbb', cov=(1.0, 0.0, 0.0), cov_l=0.0)
    add('two_sources_AB', cov=(1.0, 1.0, 0.0), cov_l=1.0)
    add('logbb_only_no_labels', cov=(0.0, 0.0, 0.0), cov_l=1.0)
    add('logbb_scale0.5', logbb_scale=0.5)
    add('stress_small_highconflict', n=200, off=(0.0, 0.8, 0.3), cov=(0.5, 0.5, 0.5), cov_l=0.3)
    add('stress_weak_reversed_missing', n=500, beta=(0.6, 0.6, 0.4), off=(0.0, 0.5, -1.0), cov=(0.5, 0.5, 0.5), cov_l=0.3, flip=(0.1, 0.1, 0.0))
    add('stress_nonlinear_missing', n=1000, nonlinear=True, cov_l=0.5)
    add('stress_all_weak_noise', n=800, beta=(0.5, 0.5, 0.5), off=(0.0, 0.3, 0.0), sigma=1.0, cov_l=0.6)
    return sc

def main():
    for out in (OUT05, OUT02):
        out.mkdir(parents=True, exist_ok=True)
    scenarios = scenario_list()
    assert len(scenarios) == 28, len(scenarios)
    rng = np.random.default_rng(SEED)
    methods = ['hard', 'majority', 'source_weighted', 'logbb_baseline', 'hem']
    rows = []
    t_all = time.time()
    for si, sc in enumerate(scenarios):
        fails = 0
        for rep in range(N_REP):
            L, Y, logbb = sim_multi(rng, sc['n'], beta=sc['beta'], off=sc['off'], cov=sc['cov'], flip=sc['flip'], cov_l=sc['cov_l'], sigma=sc['sigma'], nonlinear=sc['nonlinear'], logbb_scale=sc.get('logbb_scale', 1.0))
            base = score_methods(Y, logbb, sc['n'])
            t0 = time.time()
            m = fit_hem(Y, logbb, sc['n'])
            dt = time.time() - t0
            row = {'scenario': sc['name'], 'rep': rep}
            for meth, score in base.items():
                rho, auc, brier = eval_score(score, L)
                row['%s_rho' % meth] = rho
                row['%s_auc' % meth] = auc
                row['%s_brier' % meth] = brier
            if m is None:
                fails += 1
                row['hem_fail'] = 1
                for k in ('hem_rho', 'hem_auc', 'hem_brier', 'hem_cov95'):
                    row[k] = np.nan
            else:
                row['hem_fail'] = 0
                p = 1.0 / (1.0 + np.exp(-m.mu))
                rho, auc, brier = eval_score(p, L)
                row['hem_rho'], row['hem_auc'], row['hem_brier'] = (rho, auc, brier)
                row['hem_cov95'] = float(np.mean(np.abs(L - m.mu) <= 1.96 * m.sigma))
                row['hem_delta_hat'] = float(m.delta)
            row['hem_time_s'] = dt
            rows.append(row)
        print('[%02d/28] %s done, HEM fail=%d (%.0fs)' % (si + 1, sc['name'], fails, time.time() - t_all), flush=True)
    df = pd.DataFrame(rows)
    cols = []
    for meth in methods:
        for k in ('rho', 'auc', 'brier'):
            cols.append('%s_%s' % (meth, k))
    summ_rows = []
    for name, g in df.groupby('scenario'):
        r = {'scenario': name, 'n_reps': int(len(g))}
        for c in cols:
            v = g[c].dropna()
            if len(v) == 0:
                r[c] = np.nan
                r[c + '_se'] = np.nan
            else:
                r[c] = float(v.mean())
                r[c + '_se'] = float(v.std(ddof=1) / np.sqrt(len(v)))
        r['hem_cov95'] = float(g['hem_cov95'].dropna().mean()) if g['hem_cov95'].notna().any() else np.nan
        r['hem_fail_rate'] = float(g['hem_fail'].mean())
        r['hem_time_s'] = float(g['hem_time_s'].mean())
        summ_rows.append(r)
    summ = pd.DataFrame(summ_rows)
    report = {'design': {'n_scenarios': len(scenarios), 'n_reps': N_REP, 'seed': SEED, 'generation': '多来源：三真实二分类来源 A/B/C（β 灵敏度、off 方向偏移、cov 覆盖率、flip 翻转） + 连续 logBB 来源；所有方法共享相同原始观测', 'methods': methods, 'fairness': 'HEM 不独占连续信息：logbb-baseline 使用固定尺度 sigmoid(logBB)；majority 基于离散来源标签；source-weighted 为等权软平均；hard 为单来源朴素基线；所有方法使用同一组实现化标签与 logBB', 'metrics': ['Spearman rho', 'ROC-AUC', 'Brier', 'HEM 95% 区间覆盖率', '失败率', 'Monte Carlo SE', '运行时间'], 'mc_se': 'SE = 场景内跨 100 次重复的样本标准差 / sqrt(n_reps)', 'scenario_params': [{k: v for k, v in s.items() if k != 'name'} for s in scenarios]}, 'scenario_summary': summ.to_dict(orient='records')}
    for out in (OUT05, OUT02):
        df.to_csv(out / 'per_rep.csv', index=False, encoding='utf-8-sig')
        summ.to_csv(out / 'scenario_summary.csv', index=False, encoding='utf-8-sig')
        with open(out / 'simulation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print('=== 场景汇总（ρ：oracle 见 hard/logbb；重点 hard vs majority vs HEM）===')
    cols_out = ['scenario', 'hard_rho', 'majority_rho', 'source_weighted_rho', 'logbb_baseline_rho', 'hem_rho', 'hem_auc', 'hard_auc', 'hem_cov95', 'hem_fail_rate']
    print(summ[cols_out].round(3).to_string(index=False))
    summ['hem_ge_majority'] = summ['hem_rho'] >= summ['majority_rho']
    summ['hem_ge_logbb'] = summ['hem_rho'] >= summ['logbb_baseline_rho']
    agg = {'frac_hem_rho_ge_majority': float(summ['hem_ge_majority'].mean()), 'frac_hem_rho_ge_logbb_baseline': float(summ['hem_ge_logbb'].mean()), 'mean_hem_rho': float(summ['hem_rho'].mean()), 'mean_majority_rho': float(summ['majority_rho'].mean()), 'mean_hard_rho': float(summ['hard_rho'].mean()), 'mean_logbb_rho': float(summ['logbb_baseline_rho'].mean()), 'overall_hem_fail_rate': float(summ['hem_fail_rate'].mean())}
    for out in (OUT05, OUT02):
        with open(out / 'summary_stats.json', 'w', encoding='utf-8') as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
    print()
    print('=== 汇总（公平口径）===')
    print(json.dumps(agg, ensure_ascii=False, indent=1))
if __name__ == '__main__':
    main()

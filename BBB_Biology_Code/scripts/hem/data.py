from __future__ import annotations
from project_paths import project_root, project_path
from pathlib import Path
import numpy as np
import pandas as pd
DEFAULT_CONSENSUS = project_path('05_复现工作区', 'results_stage5', 'data', 'consensus_dataset.csv')
DEFAULT_LOGBB_LONG = project_path('05_复现工作区', 'results_stage5', 'data', 'logbb_measurements_long.csv')

def load_consensus(path=DEFAULT_CONSENSUS) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna('')
    return df

def load_logbb_measurements(path=DEFAULT_LOGBB_LONG) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna('')
    return df

def parse_logbb_list(val: str):
    if not isinstance(val, str) or not val.strip():
        return []
    out = []
    for x in val.split(';'):
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out

def build_tensors(df: pd.DataFrame, use_reference_out: bool=False) -> dict:
    n = len(df)
    if use_reference_out:
        keep = np.ones(n, dtype=bool)
    else:
        keep = df['in_reference_out'].astype(int).values == 0
    idx_map = np.arange(n)
    b3db_idx = []
    b3db_labels = []
    for i in range(n):
        if not keep[i]:
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
    b3db_idx = np.array(b3db_idx, dtype=int)
    b3db_labels = np.array(b3db_labels, dtype=float)
    bbbp_mask = keep & df['bbbp_label'].astype(str).str.strip().isin([str(x) for x in ('0', '1', '0.0', '1.0', 0, 1)])
    bbbp_idx = idx_map[bbbp_mask.values]
    bbbp_labels = df.loc[bbbp_mask, 'bbbp_label'].astype(float).values
    logbb_idx = []
    logbb_vals = []
    for i, val in enumerate(df['b3db_logbb_list']):
        if not keep[i]:
            continue
        vals = parse_logbb_list(val)
        for v in vals:
            logbb_idx.append(i)
            logbb_vals.append(v)
    logbb_idx = np.array(logbb_idx, dtype=int)
    logbb_vals = np.array(logbb_vals, dtype=float)
    return {'n': n, 'keep': keep, 'b3db_idx': b3db_idx, 'b3db_labels': b3db_labels, 'bbbp_idx': bbbp_idx, 'bbbp_labels': bbbp_labels, 'logbb_idx': logbb_idx, 'logbb_vals': logbb_vals, 'idx_map': idx_map}

def build_ref_matrix(df: pd.DataFrame, refs: list[str]) -> np.ndarray:
    ref_to_col = {r: i for i, r in enumerate(refs)}
    n = len(df)
    mat = np.zeros((n, len(refs)), dtype=np.float64)
    for i, refstr in enumerate(df['reference_list']):
        for r in refstr.split('|'):
            if r in ref_to_col:
                mat[i, ref_to_col[r]] = 1.0
    return mat

def collect_references(df: pd.DataFrame) -> list[str]:
    refs = set()
    for refstr in df['reference_list']:
        for r in refstr.split('|'):
            if r:
                refs.add(r)
    return sorted(refs)

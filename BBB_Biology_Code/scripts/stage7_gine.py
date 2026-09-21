from __future__ import annotations
from project_paths import project_root, project_path
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage7_main import vote_soft_label_map, lbl
from hem import LatentHEM, build_tensors
RDLogger.DisableLog('rdApp.*')
CONSENSUS = project_path('05_复现工作区', 'results_stage5', 'data', 'consensus_dataset.csv')
SCAFFOLD = project_path('05_复现工作区', 'results_stage7', 'data', 'scaffold_split_assignments.csv')
OUTPUT_DIR = project_path('05_复现工作区', 'results_stage7')
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.2
LR = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 120
PATIENCE = 12

def atom_values(atom):
    return {'atomic_num': str(atom.GetAtomicNum()), 'degree': str(atom.GetDegree()), 'formal_charge': str(atom.GetFormalCharge()), 'hybridization': str(atom.GetHybridization()), 'chirality': str(atom.GetChiralTag()), 'aromatic': str(int(atom.GetIsAromatic())), 'total_h': str(atom.GetTotalNumHs())}

def bond_values(bond):
    return {'bond_type': str(bond.GetBondType()), 'conjugated': str(int(bond.GetIsConjugated())), 'in_ring': str(int(bond.IsInRing()))}

def make_vocabularies(smiles_list):
    atom_vocab = {key: set() for key in atom_values(Chem.MolFromSmiles('C').GetAtomWithIdx(0)).keys()}
    bond_vocab = {key: set() for key in bond_values(Chem.MolFromSmiles('CC').GetBondWithIdx(0)).keys()}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            for k, v in atom_values(atom).items():
                atom_vocab[k].add(v)
        for bond in mol.GetBonds():
            for k, v in bond_values(bond).items():
                bond_vocab[k].add(v)
    atom_maps = {k: {v: i for i, v in enumerate(sorted(vals))} for k, vals in atom_vocab.items()}
    bond_maps = {k: {v: i for i, v in enumerate(sorted(vals))} for k, vals in bond_vocab.items()}
    return (atom_maps, bond_maps)

def one_hot(value, mapping):
    vec = np.zeros(len(mapping), dtype=np.float32)
    key = str(value)
    if key in mapping:
        vec[mapping[key]] = 1.0
    else:
        vec[0] = 1.0
    return vec

def build_graph(smiles, atom_maps, bond_maps):
    mol = Chem.MolFromSmiles(smiles)
    node_rows = [np.concatenate([one_hot(atom_values(a)[k], atom_maps[k]) for k in atom_maps]) for a in mol.GetAtoms()]
    edge_rows, edge_pairs = ([], [])
    edge_dim = sum((len(v) for v in bond_maps.values()))
    for bond in mol.GetBonds():
        f = np.concatenate([one_hot(bond_values(bond)[k], bond_maps[k]) for k in bond_maps])
        b, e = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        edge_pairs.extend([[b, e], [e, b]])
        edge_rows.extend([f, f])
    x = torch.tensor(np.asarray(node_rows), dtype=torch.float32)
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous() if edge_pairs else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(np.asarray(edge_rows), dtype=torch.float32) if edge_rows else torch.empty((0, edge_dim), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class GINERegressor(nn.Module):

    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(NUM_LAYERS):
            in_dim = node_dim if i == 0 else HIDDEN_DIM
            mlp = nn.Sequential(nn.Linear(in_dim, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(HIDDEN_DIM))
        self.head = nn.Sequential(nn.Linear(HIDDEN_DIM, 32), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(32, 1))

    def forward(self, data):
        x = data.x
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, data.edge_index, data.edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=DROPOUT, training=self.training)
        pooled = global_mean_pool(x, data.batch)
        return torch.sigmoid(self.head(pooled).view(-1))

def train_gine(train_graphs, val_graphs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]
    model = GINERegressor(node_dim, edge_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
    loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    best = (float('inf'), None)
    patience = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            loss = F.mse_loss(pred, batch.y)
            loss.backward()
            opt.step()
        model.eval()
        vpred, vy = ([], [])
        with torch.no_grad():
            for vb in DataLoader(val_graphs, batch_size=BATCH_SIZE):
                vpred.append(model(vb).numpy())
                vy.append(vb.y.numpy())
        vloss = float(np.mean((np.concatenate(vpred) - np.concatenate(vy)) ** 2))
        if vloss < best[0]:
            best = (vloss, {k: v.detach().clone() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best[1])
    return model

def main():
    (OUTPUT_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CONSENSUS, dtype=str).fillna('')
    sc = pd.read_csv(SCAFFOLD, dtype=str).fillna('')
    tr = df[(df['split'] == 'train') & (df['in_reference_out'].astype(int) == 0)].reset_index(drop=True)
    bbbp_test = df[df['split'] == 'test'].reset_index(drop=True)
    scaf_test = sc[sc['scaffold_split'] == 'test'].reset_index(drop=True)
    refout_df = df[df['in_reference_out'].astype(int) == 1].reset_index(drop=True)
    b3db = tr['b3db_label'].apply(lbl).values
    a_label = b3db.copy()
    vote = vote_soft_label_map()
    b_vote = tr['parent_ik'].map(vote).values
    b_label = np.where(pd.notna(b_vote), b_vote, b3db)
    T = build_tensors(tr, use_reference_out=False)
    hem = LatentHEM(len(tr), n_em_iters=25, m_epochs=50, m_lr=0.01, l2_prior=0.1)
    hem.fit(T['b3db_idx'], T['b3db_labels'], bbbp_idx=T['bbbp_idx'], bbbp_labels=T['bbbp_labels'], logbb_idx=T['logbb_idx'], logbb_vals=T['logbb_vals'])
    c_label = hem.predict(np.arange(len(tr)))
    equal_mask = ~np.isnan(a_label)
    atom_maps, bond_maps = make_vocabularies(tr['canon_smiles'].tolist())
    tr_graphs = [build_graph(s, atom_maps, bond_maps) for s in tr['canon_smiles']]
    bbbp_graphs = [build_graph(s, atom_maps, bond_maps) for s in bbbp_test['canon_smiles']]
    scaf_graphs = [build_graph(s, atom_maps, bond_maps) for s in scaf_test['canon_smiles']]
    refout_graphs = [build_graph(s, atom_maps, bond_maps) for s in refout_df['canon_smiles']]
    SEEDS = (42, 52, 62)
    results = []
    preds_store = {}
    for arm, y in [('A', a_label), ('B', b_label), ('C', c_label)]:
        mask = equal_mask
        arm_seed_preds = {en: [] for en in ['BBBP_test', 'scaffold_test', 'reference_out']}
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(tr))
            n_val = int(0.15 * len(tr))
            val_idx, train_idx = (idx[:n_val], idx[n_val:])
            torch.manual_seed(seed)
            np.random.seed(seed)
            for i in train_idx:
                if mask[i]:
                    tr_graphs[i].y = torch.tensor([y[i]], dtype=torch.float32)
            for i in val_idx:
                if mask[i]:
                    tr_graphs[i].y = torch.tensor([y[i]], dtype=torch.float32)
            tgraphs = [tr_graphs[i] for i in train_idx if mask[i]]
            vgraphs = [tr_graphs[i] for i in val_idx if mask[i]]
            model = train_gine(tgraphs, vgraphs, seed)
            model.eval()
            for eval_name, graphs in [('BBBP_test', bbbp_graphs), ('scaffold_test', scaf_graphs), ('reference_out', refout_graphs)]:
                preds = []
                with torch.no_grad():
                    for b in DataLoader(graphs, batch_size=BATCH_SIZE):
                        preds.append(model(b).numpy())
                arm_seed_preds[eval_name].append(np.concatenate(preds))
        for eval_name, eval_df in [('BBBP_test', bbbp_test), ('scaffold_test', scaf_test), ('reference_out', refout_df)]:
            gold_col = 'bbbp_label' if eval_name == 'BBBP_test' else 'b3db_label'
            gold = eval_df[gold_col].apply(lbl).values
            valid = ~np.isnan(gold)
            if valid.sum() == 0:
                continue
            stack = np.stack([p for p in arm_seed_preds[eval_name]])
            p_mean = stack.mean(axis=0)
            rocs = [roc_auc_score(gold[valid], p[valid]) for p in stack]
            prs = [average_precision_score(gold[valid], p[valid]) for p in stack]
            results.append({'arm': arm, 'eval': eval_name, 'gold': gold_col, 'n': int(valid.sum()), 'roc_auc_mean': float(np.mean(rocs)), 'roc_auc_std': float(np.std(rocs)), 'pr_auc_mean': float(np.mean(prs)), 'pr_auc_std': float(np.std(prs)), 'n_seeds': len(SEEDS)})
            preds_store[f'{arm}_{eval_name}'] = {'gold': gold[valid], 'pred': p_mean[valid]}
    gine_table = pd.DataFrame(results)
    gine_table.to_csv(OUTPUT_DIR / 'reports' / 'gine_main_table.csv', index=False, encoding='utf-8-sig')
    np.savez(OUTPUT_DIR / 'reports' / 'gine_predictions.npz', **{k: v['pred'] for k, v in preds_store.items()}, gold_bbbp=preds_store['A_BBBP_test']['gold'], gold_scaf=preds_store['A_scaffold_test']['gold'], gold_refout=preds_store['A_reference_out']['gold'])
    print('=== GINE 主表（3 种子）===')
    print(gine_table.to_string(index=False))
    print(f"保存: {OUTPUT_DIR / 'reports' / 'gine_main_table.csv'}")
if __name__ == '__main__':
    main()

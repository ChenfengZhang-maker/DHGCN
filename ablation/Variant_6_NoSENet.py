import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pathlib
import warnings
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import HypergraphConv

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


class Config:
    def __init__(self):
        self.train_data_path = pathlib.Path("./data/train_14400.csv")
        self.val_data_path = pathlib.Path("./data/val_14400.csv")
        self.test_data_path = pathlib.Path("./data/test_14400.csv")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.result_csv_name = os.path.join(current_dir, "Variant6_Results.csv")

        self.embedding_dim = 128
        self.hidden_dim = 64
        self.num_classes = 10
        self.num_epochs = 100
        self.batch_size = 256
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.num_heads = 4
        self.dropout = 0.2
        self.patience = 3
        self.min_improvement = 0.001
        self.alpha_hgcn = 0.7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = Config()


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HyperedgeDataset(Dataset):
    def __init__(self, triples, labels, node_to_idx):
        self.triples = triples
        self.labels = labels
        self.node_to_idx = node_to_idx

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        d, g, s = self.triples[idx]
        y = self.labels[idx]
        triple_idx = torch.tensor(
            [self.node_to_idx[d], self.node_to_idx[g], self.node_to_idx[s]],
            dtype=torch.long,
        )
        return triple_idx, torch.tensor(y, dtype=torch.long)


def build_mode_hyperedge_indices(all_triples, all_rel, node_to_idx, num_classes, device):
    mode_to_triples = {c: [] for c in range(num_classes)}
    for trip, y in zip(all_triples, all_rel):
        mode_to_triples[int(y)].append(trip)

    hyperedge_indices = []
    for c in range(num_classes):
        triples_c = mode_to_triples.get(c, [])
        if len(triples_c) == 0:
            hyperedge_indices.append(None)
            continue

        node_ids, hedge_ids = [], []
        for e, (d, g, s) in enumerate(triples_c):
            node_ids += [node_to_idx[d], node_to_idx[g], node_to_idx[s]]
            hedge_ids += [e, e, e]

        hyperedge_index_c = torch.tensor([node_ids, hedge_ids], dtype=torch.long, device=device)
        hyperedge_indices.append(hyperedge_index_c)
    return hyperedge_indices


# --- 模块：IMF (Inter-mode feature fusion 交互) ---
class IMF_Layer(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super(IMF_Layer, self).__init__()
        self.in_dim = in_dim
        self.W_q = nn.Linear(in_dim, in_dim)
        self.W_k = nn.Linear(in_dim, in_dim)
        self.W_v = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        x_trans = x.permute(1, 0, 2)
        residual = x_trans

        Q = self.W_q(x_trans)
        K = self.W_k(x_trans)
        V = self.W_v(x_trans)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.in_dim ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))

        x_new = torch.matmul(attn, V)
        x_new = self.layer_norm(x_new + residual)

        return x_new.permute(1, 0, 2)


# --- 模块：Highway Gating (高速公路门控) ---
class HighwayGatingLayer(nn.Module):
    def __init__(self, in_dim):
        super(HighwayGatingLayer, self).__init__()
        self.H_layer = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU()
        )
        self.T_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        nn.init.constant_(self.T_gate[0].bias, -1.0)

    def forward(self, x):
        h_feat = self.H_layer(x)
        transform_gate = self.T_gate(x)
        carry_gate = 1.0 - transform_gate
        output = h_feat * transform_gate + x * carry_gate
        return output


class Variant6_NoSENet(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, num_classes, hyperedge_index_list, num_heads=4, alpha=0.7,
                 dropout=0.2):
        super().__init__()
        self.num_modes = num_classes
        self.num_heads = num_heads
        self.alpha = alpha

        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # 1. 独立卷积 (IHC)
        self.hconvs1 = nn.ModuleList([HypergraphConv(embedding_dim, embedding_dim) for _ in range(self.num_modes)])
        self.hconvs2 = nn.ModuleList([HypergraphConv(embedding_dim, embedding_dim) for _ in range(self.num_modes)])

        self._register_edges(hyperedge_index_list)

        # 2. 模式间交互 (IMF)
        self.imf_layer = IMF_Layer(embedding_dim, dropout)

        # 注意：故意省略 ChannelAttention (SE-Net)

        # 3. 高速公路门控 (Highway Gating)
        self.gating_layer = HighwayGatingLayer(embedding_dim)

        # 4. 多视角查询注意力 (MVQA) 参数
        self.mode_embeddings = nn.Parameter(torch.randn(self.num_heads, self.num_modes, embedding_dim))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.head_hgcn = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.head_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self._init_weights()

    def _register_edges(self, hyperedge_index_list):
        dev = torch.device("cpu")
        for hed in hyperedge_index_list:
            if hed is not None: dev = hed.device; break
        for i, hedge_idx in enumerate(hyperedge_index_list):
            name = f"hyperedge_index_mode_{i}"
            if hedge_idx is None:
                self.register_buffer(name, torch.empty(2, 0, dtype=torch.long, device=dev))
            else:
                self.register_buffer(name, hedge_idx)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.mode_embeddings)
        for m in list(self.head_hgcn) + list(self.head_mlp):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, triple_idx):
        x0 = self.node_embedding.weight
        mode_feats = []

        # --- 步骤 1: 独立卷积 (IHC) ---
        for i in range(self.num_modes):
            hedge = getattr(self, f"hyperedge_index_mode_{i}")
            if hedge.numel() == 0:
                mode_feats.append(x0)
                continue

            h1 = self.dropout(self.act(self.hconvs1[i](x0, hedge)))
            h2 = self.hconvs2[i](h1, hedge)
            mode_feats.append(h2)

        X_modes = torch.stack(mode_feats, dim=0)  # [M, N, D]

        # --- 步骤 2: 跨模态特征融合交互 (IMF) ---
        X_processed = self.imf_layer(X_modes)

        # (跳过了 SE-Net 特征精修)

        # --- 步骤 3: 高速公路门控 (Highway Gating) ---
        X_processed = self.gating_layer(X_processed)

        # --- 步骤 4: 多视角查询注意力 (MVQA) ---
        Xm = X_processed.unsqueeze(0)
        Sm = self.mode_embeddings.unsqueeze(2)

        attn_logits = (Xm * Sm).sum(dim=-1)
        attn = torch.softmax(attn_logits, dim=1)

        Z_per_head = (attn.unsqueeze(-1) * Xm).sum(dim=1)
        z = Z_per_head.mean(dim=0)

        # --- 步骤 5: 预测 ---
        d, g, s = triple_idx[:, 0], triple_idx[:, 1], triple_idx[:, 2]

        logits_hgcn = self.head_hgcn(torch.cat([z[d], z[g], z[s]], dim=1))
        logits_mlp = self.head_mlp(torch.cat([x0[d], x0[g], x0[s]], dim=1))

        return logits_hgcn, logits_mlp

    def inference(self, triple_idx):
        l, _ = self.forward(triple_idx)
        return l


def run_experiment(run_id, train_loader, val_loader, test_loader, num_nodes, num_classes, he_list, cw):
    model = Variant6_NoSENet(num_nodes, args.embedding_dim, args.hidden_dim, num_classes, he_list, args.num_heads,
                             args.alpha_hgcn, args.dropout).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=cw)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    patience_cnt = 0
    best_weights = None

    for epoch in range(args.num_epochs):
        model.train()
        train_loss, train_preds, train_trues = 0, [], []

        for idx, labels in train_loader:
            idx, labels = idx.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            out_m, out_a = model(idx)
            loss = args.alpha_hgcn * criterion(out_m, labels) + (1 - args.alpha_hgcn) * criterion(out_a, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(out_m.argmax(1).cpu().numpy())
            train_trues.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_trues, train_preds)
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, val_preds, val_trues = 0, [], []
        with torch.no_grad():
            for idx, labels in val_loader:
                idx, labels = idx.to(args.device), labels.to(args.device)
                out_m, out_a = model(idx)
                loss = args.alpha_hgcn * criterion(out_m, labels) + (1 - args.alpha_hgcn) * criterion(out_a, labels)
                val_loss += loss.item()
                val_preds.extend(out_m.argmax(1).cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_trues, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if val_acc - best_val_acc > args.min_improvement:
            best_val_acc = val_acc
            best_weights = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                break

    if best_weights: model.load_state_dict(best_weights)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for idx, labels in test_loader:
            idx, labels = idx.to(args.device), labels.to(args.device)
            logits = model.inference(idx)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, 1).cpu().numpy())

    metrics = {
        "ACC": accuracy_score(all_labels, all_preds),
        "PRE": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "REC": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "F1": f1_score(all_labels, all_preds, average='macro', zero_division=0)
    }

    try:
        y_oh = OneHotEncoder(sparse_output=False, categories=[range(num_classes)]).fit_transform(
            np.array(all_labels).reshape(-1, 1))
        metrics["AUC"] = roc_auc_score(y_oh, np.array(all_probs), multi_class="ovr", average="macro")
        metrics["AUPR"] = average_precision_score(y_oh, np.array(all_probs), average="macro")
    except:
        metrics["AUC"], metrics["AUPR"] = 0.0, 0.0

    return metrics


def main():
    df_train = pd.read_csv(args.train_data_path)
    df_val = pd.read_csv(args.val_data_path)
    df_test = pd.read_csv(args.test_data_path)

    node_to_idx, cnt = {}, 0
    for df in [df_train, df_val, df_test]:
        for _, r in df.iterrows():
            for c in ['drug', 'gene', 'disease']:
                if r[c] not in node_to_idx:
                    node_to_idx[r[c]] = cnt
                    cnt += 1

    def get_loader(df, shuffle):
        t = list(zip(df['drug'], df['gene'], df['disease']))
        y = df['label'].values
        return DataLoader(HyperedgeDataset(t, y, node_to_idx), batch_size=args.batch_size, shuffle=shuffle)

    train_loader = get_loader(df_train, True)
    val_loader = get_loader(df_val, False)
    test_loader = get_loader(df_test, False)

    df_all = pd.concat([df_train, df_val, df_test])
    all_triples = list(zip(df_all['drug'], df_all['gene'], df_all['disease']))

    he_list = build_mode_hyperedge_indices(all_triples, df_all['label'].values, node_to_idx, args.num_classes,
                                           args.device)

    cw = torch.tensor(compute_class_weight('balanced', classes=np.arange(args.num_classes), y=df_train['label']),
                      dtype=torch.float32).to(args.device)

    results = []
    for i in range(5):
        set_seed(42 + i)
        results.append(
            run_experiment(i, train_loader, val_loader, test_loader, len(node_to_idx), args.num_classes, he_list, cw))

    df_res = pd.DataFrame(results, index=[f"Run_{i + 1}" for i in range(5)])
    mean, std = df_res.mean(), df_res.std()
    df_res.loc["Mean ± Std"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean, std)]

    df_res = df_res[["ACC", "PRE", "AUPR", "REC", "F1", "AUC"]]
    df_res.to_csv(args.result_csv_name)


if __name__ == "__main__":
    main()
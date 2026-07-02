# -*- coding: utf-8 -*-
"""
HGCN_Base_Visualization.py —— 原始基准模型 + t-SNE 特征空间可视化 (中间层提取版)
优化：仅运行 1 次，快速提取特征并绘图。
"""

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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# =============================
# 1. 环境与配置
# =============================
warnings.filterwarnings("ignore")


class Config:
    def __init__(self):
        self.base_dir = pathlib.Path(__file__).resolve().parent

        # 数据集路径 (144000 样本数据集)
        self.train_data_path = pathlib.Path("./data/train_14400.csv")
        self.val_data_path = pathlib.Path("./data/val_14400.csv")
        self.test_data_path = pathlib.Path("./data/test_14400.csv")

        # 保存 6 个指标结果的表格文件名
        self.result_csv_name = "Metrics_HGCN_Base_144000.csv"

        # 模型参数
        self.embedding_dim = 128
        self.hidden_dim = 64
        self.num_classes = 10
        self.num_epochs = 100
        self.batch_size = 256
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5

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


# =============================
# 2. 数据处理类
# =============================
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


def build_single_hyperedge_index(all_triples, node_to_idx, device):
    node_ids, hedge_ids = [], []
    for e, (d, g, s) in enumerate(all_triples):
        node_ids += [node_to_idx[d], node_to_idx[g], node_to_idx[s]]
        hedge_ids += [e, e, e]
    return torch.tensor([node_ids, hedge_ids], dtype=torch.long, device=device)


# =============================
# 3. 核心组件模块
# =============================
class MIL_Layer(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super(MIL_Layer, self).__init__()
        self.in_dim = in_dim
        self.W_q = nn.Linear(in_dim, in_dim)
        self.W_k = nn.Linear(in_dim, in_dim)
        self.W_v = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        residual = x
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.in_dim ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))
        x_new = torch.matmul(attn, V)
        x_new = self.layer_norm(x_new + residual)
        return x_new


# =============================
# 4. 原始基准模型架构
# =============================
class HGCN_Base(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, num_classes,
                 hyperedge_index, dropout=0.2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        self.conv1 = HypergraphConv(embedding_dim, embedding_dim)
        self.conv2 = HypergraphConv(embedding_dim, embedding_dim)

        self.register_buffer("hedge_index", hyperedge_index)

        self.mil_layer = MIL_Layer(embedding_dim, dropout)
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

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        for m in list(self.head_hgcn) + list(self.head_mlp):
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def forward(self, triple_idx):
        x0 = self.node_embedding.weight
        edge_index = self.hedge_index

        x = self.dropout(self.act(self.conv1(x0, edge_index)))
        x = self.conv2(x, edge_index)

        z = self.mil_layer(x)

        d, g, s = triple_idx[:, 0], triple_idx[:, 1], triple_idx[:, 2]

        h_triple_hgcn = torch.cat([z[d], z[g], z[s]], dim=1)
        h_triple_mlp = torch.cat([x0[d], x0[g], x0[s]], dim=1)

        x_64 = self.head_hgcn[1](self.head_hgcn[0](h_triple_hgcn))
        feat_for_tsne = self.head_hgcn[3](self.head_hgcn[2](x_64))
        logits_hgcn = self.head_hgcn[4](feat_for_tsne)

        logits_mlp = self.head_mlp(h_triple_mlp)

        return logits_hgcn, logits_mlp, feat_for_tsne

    def inference(self, triple_idx):
        l, _, feat_for_tsne = self.forward(triple_idx)
        return l, feat_for_tsne


# =============================
# 5. 可视化函数 (t-SNE)
# =============================
def plot_tsne(embeddings, labels, filename="TSNE_Standard_HGCN_144000.png"):
    print(f"\nGenerating t-SNE plot for feature space... ({len(embeddings)} points, Dim: {embeddings.shape[1]})")

    if len(embeddings) > 5000:
        idx = np.random.choice(len(embeddings), 5000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=emb_2d[:, 0], y=emb_2d[:, 1],
        hue=labels,
        palette=sns.color_palette("tab10", n_colors=len(np.unique(labels))),
        legend="full",
        alpha=0.7,
        s=20,
        edgecolor=None
    )
    plt.title("(a) t-SNE of Mixed Heterogeneous Features", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Semantic Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"t-SNE plot saved successfully as: {filename}")


# =============================
# 6. 训练与评估流程
# =============================
def run_experiment(run_id, train_loader, val_loader, test_loader, num_nodes, num_classes, he_idx, cw):
    print(f"\n{'=' * 20} Run {run_id + 1} Start {'=' * 20}")

    model = HGCN_Base(num_nodes, args.embedding_dim, args.hidden_dim, num_classes,
                      he_idx, args.dropout).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=cw)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    patience_cnt = 0
    best_weights = None

    for epoch in range(args.num_epochs):
        model.train()
        train_loss, train_preds, train_trues = 0, [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [Train]", leave=False)
        for idx, labels in pbar:
            idx, labels = idx.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            out_m, out_a, _ = model(idx)
            loss = args.alpha_hgcn * criterion(out_m, labels) + (1 - args.alpha_hgcn) * criterion(out_a, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(out_m.argmax(1).cpu().numpy())
            train_trues.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_acc = accuracy_score(train_trues, train_preds)
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, val_preds, val_trues = 0, [], []
        with torch.no_grad():
            for idx, labels in val_loader:
                idx, labels = idx.to(args.device), labels.to(args.device)
                out_m, out_a, _ = model(idx)
                loss = args.alpha_hgcn * criterion(out_m, labels) + (1 - args.alpha_hgcn) * criterion(out_a, labels)
                val_loss += loss.item()
                val_preds.extend(out_m.argmax(1).cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_trues, val_preds)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f} ACC={train_acc:.4f} | "
              f"Val Loss={avg_val_loss:.4f} ACC={val_acc:.4f}")

        if val_acc - best_val_acc > args.min_improvement:
            best_val_acc = val_acc
            best_weights = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stopping triggered at Epoch {epoch + 1}")
                break

    if best_weights: model.load_state_dict(best_weights)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_embeddings = []

    print("Testing best model...")
    with torch.no_grad():
        for idx, labels in tqdm(test_loader, desc="Testing"):
            idx, labels = idx.to(args.device), labels.to(args.device)
            logits, feat_for_tsne = model.inference(idx)

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, 1).cpu().numpy())
            all_embeddings.append(feat_for_tsne.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

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

    print(f"Run {run_id + 1} Result: {metrics}")

    return metrics, all_embeddings, np.array(all_labels)


# =============================
# 7. 主程序
# =============================
def main():
    print(f"Loading data from {args.train_data_path}...")
    df_train = pd.read_csv(args.train_data_path)
    df_val = pd.read_csv(args.val_data_path)
    df_test = pd.read_csv(args.test_data_path)

    node_to_idx, cnt = {}, 0
    for df in [df_train, df_val, df_test]:
        for _, r in df.iterrows():
            for c in ['drug', 'gene', 'disease']:
                if r[c] not in node_to_idx: node_to_idx[r[c]] = cnt; cnt += 1

    def get_loader(df, shuffle):
        t = list(zip(df['drug'], df['gene'], df['disease']))
        y = df['label'].values
        return DataLoader(HyperedgeDataset(t, y, node_to_idx), batch_size=args.batch_size, shuffle=shuffle)

    train_loader = get_loader(df_train, True)
    val_loader = get_loader(df_val, False)
    test_loader = get_loader(df_test, False)

    df_all = pd.concat([df_train, df_val, df_test])
    all_triples = list(zip(df_all['drug'], df_all['gene'], df_all['disease']))
    he_idx = build_single_hyperedge_index(all_triples, node_to_idx, args.device)

    from sklearn.utils.class_weight import compute_class_weight
    cw = torch.tensor(compute_class_weight('balanced', classes=np.arange(args.num_classes), y=df_train['label']),
                      dtype=torch.float32).to(args.device)

    results = []
    final_embeddings, final_labels = None, None

    # 只跑 1 次即可
    for i in range(1):
        set_seed(42 + i)
        metrics, emb, labels = run_experiment(i, train_loader, val_loader, test_loader, len(node_to_idx),
                                              args.num_classes, he_idx, cw)
        results.append(metrics)
        if i == 0:
            final_embeddings = emb
            final_labels = labels

    df_res = pd.DataFrame(results, index=["Run_1"])
    df_res = df_res[["ACC", "PRE", "AUPR", "REC", "F1", "AUC"]]

    output_csv_path = args.base_dir / args.result_csv_name
    df_res.to_csv(output_csv_path)

    print(f"\nSaved metric results to {output_csv_path}")
    print(df_res)

    if final_embeddings is not None:
        plot_tsne(final_embeddings, final_labels)


if __name__ == "__main__":
    main()
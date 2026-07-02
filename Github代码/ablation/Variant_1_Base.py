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
        self.result_csv_name = os.path.join(current_dir, "Variant1_Results.csv")

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


def build_unified_hyperedge_index(all_triples, node_to_idx, device):
    node_ids, hedge_ids = [], []
    for e, (d, g, s) in enumerate(all_triples):
        node_ids += [node_to_idx[d], node_to_idx[g], node_to_idx[s]]
        hedge_ids += [e, e, e]
    return torch.tensor([node_ids, hedge_ids], dtype=torch.long, device=device)


class BaseHGCN(nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_dim, num_classes, hyperedge_index, alpha=0.7, dropout=0.2):
        super().__init__()
        self.alpha = alpha

        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.hconv1 = HypergraphConv(embedding_dim, embedding_dim)
        self.hconv2 = HypergraphConv(embedding_dim, embedding_dim)

        if hyperedge_index is None:
            self.register_buffer("hyperedge_index", torch.empty(2, 0, dtype=torch.long))
        else:
            self.register_buffer("hyperedge_index", hyperedge_index)

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
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, triple_idx):
        x0 = self.node_embedding.weight

        h1 = self.dropout(self.act(self.hconv1(x0, self.hyperedge_index)))
        z = self.hconv2(h1, self.hyperedge_index)

        d, g, s = triple_idx[:, 0], triple_idx[:, 1], triple_idx[:, 2]

        logits_hgcn = self.head_hgcn(torch.cat([z[d], z[g], z[s]], dim=1))
        logits_mlp = self.head_mlp(torch.cat([x0[d], x0[g], x0[s]], dim=1))

        return logits_hgcn, logits_mlp

    def inference(self, triple_idx):
        l, _ = self.forward(triple_idx)
        return l


def run_experiment(run_id, train_loader, val_loader, test_loader, num_nodes, num_classes, he_index, cw):
    model = BaseHGCN(num_nodes, args.embedding_dim, args.hidden_dim, num_classes, he_index, args.alpha_hgcn,
                     args.dropout).to(args.device)

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

    he_index = build_unified_hyperedge_index(all_triples, node_to_idx, args.device)

    cw = torch.tensor(compute_class_weight('balanced', classes=np.arange(args.num_classes), y=df_train['label']),
                      dtype=torch.float32).to(args.device)

    results = []
    for i in range(5):
        set_seed(42 + i)
        results.append(
            run_experiment(i, train_loader, val_loader, test_loader, len(node_to_idx), args.num_classes, he_index, cw))

    df_res = pd.DataFrame(results, index=[f"Run_{i + 1}" for i in range(5)])
    mean, std = df_res.mean(), df_res.std()
    df_res.loc["Mean ± Std"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean, std)]

    df_res = df_res[["ACC", "PRE", "AUPR", "REC", "F1", "AUC"]]
    df_res.to_csv(args.result_csv_name)


if __name__ == "__main__":
    main()
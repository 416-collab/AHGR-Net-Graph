import os
import pickle
import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pool_sequence(arr, lengths=None, mode="mean_max_std"):
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n, t, d = arr.shape
    out = []

    for i in range(n):
        if lengths is None:
            seq = arr[i]
        else:
            L = int(lengths[i])
            L = max(1, min(L, t))
            seq = arr[i, :L]

        mean_pool = seq.mean(axis=0)
        max_pool = seq.max(axis=0)
        std_pool = seq.std(axis=0)

        if mode == "mean":
            vec = mean_pool
        elif mode == "mean_max":
            vec = np.concatenate([mean_pool, max_pool], axis=0)
        else:
            vec = np.concatenate([mean_pool, max_pool, std_pool], axis=0)

        out.append(vec)

    return np.stack(out).astype(np.float32)


def build_context_features(text_feat, k=2):
    """
    Similarity-guided context mining within each split.
    For each sample, retrieve top-K previous samples by cosine similarity.
    This gives a direct non-ensemble AHGR-style context vector.
    """
    n, d = text_feat.shape
    norm = text_feat / (np.linalg.norm(text_feat, axis=1, keepdims=True) + 1e-8)
    ctx = np.zeros_like(text_feat, dtype=np.float32)

    for i in range(n):
        if i == 0:
            ctx[i] = 0.0
            continue

        sims = norm[:i] @ norm[i]
        kk = min(k, i)
        idx = np.argsort(sims)[-kk:]
        weights = np.exp(sims[idx])
        weights = weights / (weights.sum() + 1e-8)
        ctx[i] = (text_feat[idx] * weights[:, None]).sum(axis=0)

    return ctx.astype(np.float32)


def load_chsims(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    splits = {}
    for split in ["train", "valid", "test"]:
        d = data[split]

        text = pool_sequence(d["text"], None, "mean_max")
        audio = pool_sequence(d["audio"], d["audio_lengths"], "mean_max_std")
        vision = pool_sequence(d["vision"], d["vision_lengths"], "mean_max_std")
        y = d["regression_labels"].astype(np.float32).reshape(-1)

        splits[split] = {
            "text": text,
            "audio": audio,
            "vision": vision,
            "label": y,
        }

    return splits


def standardize_splits(splits):
    for key in ["text", "audio", "vision"]:
        scaler = StandardScaler()
        splits["train"][key] = scaler.fit_transform(splits["train"][key]).astype(np.float32)
        splits["valid"][key] = scaler.transform(splits["valid"][key]).astype(np.float32)
        splits["test"][key] = scaler.transform(splits["test"][key]).astype(np.float32)

    for split in ["train", "valid", "test"]:
        splits[split]["context"] = build_context_features(splits[split]["text"], k=2)

    return splits


class CHSIMSDataset(Dataset):
    def __init__(self, split_data):
        self.text = torch.tensor(split_data["text"], dtype=torch.float32)
        self.audio = torch.tensor(split_data["audio"], dtype=torch.float32)
        self.vision = torch.tensor(split_data["vision"], dtype=torch.float32)
        self.context = torch.tensor(split_data["context"], dtype=torch.float32)
        self.label = torch.tensor(split_data["label"], dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "vision": self.vision[idx],
            "context": self.context[idx],
            "label": self.label[idx],
        }


class AHGRDirect(nn.Module):
    """
    Direct single-model AHGR-style network for CH-SIMS:
    text/audio/vision encoding + similarity-guided context + aspect-anchor attention.
    This is not an ensemble.
    """
    def __init__(self, text_dim, audio_dim, vision_dim, hidden_dim=256, num_aspects=12, dropout=0.2):
        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.aspect_embeddings = nn.Parameter(torch.randn(num_aspects, hidden_dim) * 0.02)
        self.aspect_gate = nn.Linear(hidden_dim, hidden_dim)

        self.relation_attention = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.regressor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, text, audio, vision, context):
        ht = self.text_proj(text)
        ha = self.audio_proj(audio)
        hv = self.vision_proj(vision)
        hc = self.context_proj(context)

        base = (ht + ha + hv + hc) / 4.0

        # Aspect-anchor attention.
        asp = self.aspect_embeddings
        scores = torch.matmul(self.aspect_gate(base), asp.t())
        weights = torch.softmax(scores, dim=-1)
        hg = torch.matmul(weights, asp)

        stacked = torch.stack([ht, ha, hv, hc, hg], dim=1)
        rel_logits = self.relation_attention(torch.cat([ht, ha, hv, hc, hg], dim=-1))
        rel_weights = torch.softmax(rel_logits, dim=-1).unsqueeze(-1)
        weighted = (stacked * rel_weights).reshape(text.size(0), -1)

        z = self.fusion(weighted)
        pred = torch.tanh(self.regressor(z)).squeeze(-1)
        return pred, weights


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.clip(np.asarray(y_pred).reshape(-1), -1, 1)

    mae = mean_absolute_error(y_true, y_pred)
    corr = pearsonr(y_true, y_pred)[0] if len(np.unique(np.round(y_pred, 4))) > 1 else 0.0

    y_nonneg = (y_true >= 0).astype(int)
    p_nonneg = (y_pred >= 0).astype(int)
    acc_2_nonneg = accuracy_score(y_nonneg, p_nonneg)
    f1_2_nonneg = f1_score(y_nonneg, p_nonneg, average="weighted", zero_division=0)

    mask = y_true != 0
    y_no_neu = (y_true[mask] > 0).astype(int)
    p_no_neu = (y_pred[mask] > 0).astype(int)
    acc_2_no_neutral = accuracy_score(y_no_neu, p_no_neu)
    f1_2_no_neutral = f1_score(y_no_neu, p_no_neu, average="weighted", zero_division=0)

    def to3(x):
        out = np.zeros_like(x, dtype=int)
        out[x < 0] = 0
        out[x == 0] = 1
        out[x > 0] = 2
        return out

    y3 = to3(y_true)
    p3 = to3(y_pred)
    acc_3 = accuracy_score(y3, p3)
    f1_3_macro = f1_score(y3, p3, average="macro", zero_division=0)

    y11 = np.rint((np.clip(y_true, -1, 1) + 1) * 5).astype(int)
    p11 = np.rint((np.clip(y_pred, -1, 1) + 1) * 5).astype(int)
    y11 = np.clip(y11, 0, 10)
    p11 = np.clip(p11, 0, 10)

    acc_11 = accuracy_score(y11, p11)
    f1_11_macro = f1_score(y11, p11, average="macro", zero_division=0)

    return {
        "mae": mae,
        "corr": corr,
        "acc_2_nonneg": acc_2_nonneg,
        "f1_2_nonneg": f1_2_nonneg,
        "acc_2_no_neutral": acc_2_no_neutral,
        "f1_2_no_neutral": f1_2_no_neutral,
        "acc_3": acc_3,
        "f1_3_macro": f1_3_macro,
        "acc_11": acc_11,
        "f1_11_macro": f1_11_macro,
    }


def predict(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for b in loader:
            text = b["text"].to(device)
            audio = b["audio"].to(device)
            vision = b["vision"].to(device)
            context = b["context"].to(device)
            label = b["label"].to(device)

            pred, _ = model(text, audio, vision, context)
            preds.append(pred.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

    return np.concatenate(labels), np.concatenate(preds)


def train_one(seed=42, hidden_dim=256, lr=1e-4, batch_size=64, epochs=80):
    set_seed(seed)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    splits = load_chsims("data/raw/CHSIMS/unaligned.pkl")
    splits = standardize_splits(splits)

    print("Train shapes:",
          splits["train"]["text"].shape,
          splits["train"]["audio"].shape,
          splits["train"]["vision"].shape)

    train_ds = CHSIMSDataset(splits["train"])
    valid_ds = CHSIMSDataset(splits["valid"])
    test_ds = CHSIMSDataset(splits["test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = AHGRDirect(
        text_dim=splits["train"]["text"].shape[1],
        audio_dim=splits["train"]["audio"].shape[1],
        vision_dim=splits["train"]["vision"].shape[1],
        hidden_dim=hidden_dim,
        num_aspects=12,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss()

    best_val_score = -1e9
    best_path = f"checkpoints/AHGR_CHSIMS_Direct_seed{seed}_h{hidden_dim}_lr{lr}.pt"
    patience = 12
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for b in train_loader:
            text = b["text"].to(device)
            audio = b["audio"].to(device)
            vision = b["vision"].to(device)
            context = b["context"].to(device)
            label = b["label"].to(device)

            pred, aspect_weights = model(text, audio, vision, context)
            loss_reg = loss_fn(pred, label)

            # Aspect sparsity regularization for interpretable anchors.
            entropy = -(aspect_weights * torch.log(aspect_weights + 1e-8)).sum(dim=1).mean()
            loss = loss_reg + 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * label.size(0)
            total_n += label.size(0)

        yv, pv = predict(model, valid_loader, device)
        vm = compute_metrics(yv, pv)

        val_score = vm["corr"] - 0.5 * vm["mae"] + vm["acc_2_no_neutral"] + 0.3 * vm["f1_2_no_neutral"]

        print(
            f"Epoch {epoch:02d} | train_loss={total_loss/total_n:.4f} "
            f"| val_mae={vm['mae']:.4f} val_corr={vm['corr']:.4f} "
            f"val_acc2={vm['acc_2_no_neutral']:.4f} val_score={val_score:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    yt, pt = predict(model, test_loader, device)
    tm = compute_metrics(yt, pt)

    row = {
        "model": "AHGR_Net_CHSIMS_Direct",
        "checkpoint": best_path,
        "seed": seed,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "best_val_score": best_val_score,
        **tm,
        "notes": "Direct single AHGR-style neural model; no ensemble",
    }

    print("\n========== CH-SIMS Direct AHGR-Net Result ==========")
    print(pd.DataFrame([row]).to_string(index=False))

    out_path = "outputs/chsims_ahgr_direct_results.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print("\nSaved:", out_path)
    print("Saved checkpoint:", best_path)


if __name__ == "__main__":
    train_one(seed=42, hidden_dim=256, lr=1e-4, batch_size=64, epochs=80)

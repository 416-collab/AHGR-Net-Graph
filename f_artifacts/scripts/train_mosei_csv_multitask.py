import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_vectors(feature_path):
    with open(feature_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    pooled = []
    bin_labels = []
    for x in raw:
        feat = x["feature"].astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)

        mean_pool = feat.mean(axis=1)
        max_pool = feat.max(axis=1)
        pooled.append(np.concatenate([mean_pool, max_pool], axis=0))
        bin_labels.append(int(x["label"]))

    return np.stack(pooled).astype(np.float32), np.array(bin_labels, dtype=np.int64)


class MOSEICSVMultitaskDataset(Dataset):
    def __init__(self, x, sentiment, binary, emotions):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.sentiment = torch.tensor(sentiment, dtype=torch.float32)
        self.binary = torch.tensor(binary, dtype=torch.long)
        self.emotions = torch.tensor(emotions, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "sentiment": self.sentiment[idx],
            "binary": self.binary[idx],
            "emotions": self.emotions[idx],
        }


class MOSEICSVMultitaskModel(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, dropout=0.35):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.sentiment_head = nn.Linear(hidden_dim, 1)
        self.binary_head = nn.Linear(hidden_dim, 2)
        self.emotion_head = nn.Linear(hidden_dim, 6)

    def forward(self, x):
        h = self.encoder(x)
        return {
            "sentiment": self.sentiment_head(h).squeeze(-1),
            "binary_logits": self.binary_head(h),
            "emotion_logits": self.emotion_head(h),
        }


def compute_sentiment_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_pred = np.clip(y_pred, -3, 3)

    mae = mean_absolute_error(y_true, y_pred)

    corr = 0.0
    if len(np.unique(np.round(y_pred, 4))) > 1:
        corr = pearsonr(y_true, y_pred)[0]

    # binary non-negative: >=0 vs <0
    y_bin_nonneg = (y_true >= 0).astype(int)
    p_bin_nonneg = (y_pred >= 0).astype(int)

    acc_2_nonneg = accuracy_score(y_bin_nonneg, p_bin_nonneg)
    f1_2_nonneg = f1_score(y_bin_nonneg, p_bin_nonneg, average="weighted", zero_division=0)

    # binary no-neutral: >0 vs <0, remove y == 0
    mask = y_true != 0
    if mask.sum() > 0:
        y_bin = (y_true[mask] > 0).astype(int)
        p_bin = (y_pred[mask] > 0).astype(int)
        acc_2_no_neutral = accuracy_score(y_bin, p_bin)
        f1_2_no_neutral = f1_score(y_bin, p_bin, average="weighted", zero_division=0)
    else:
        acc_2_no_neutral = 0.0
        f1_2_no_neutral = 0.0

    # 7-class Acc/F1 from rounded sentiment score
    y7 = np.rint(np.clip(y_true, -3, 3) + 3).astype(int)
    p7 = np.rint(np.clip(y_pred, -3, 3) + 3).astype(int)
    y7 = np.clip(y7, 0, 6)
    p7 = np.clip(p7, 0, 6)

    acc_7 = accuracy_score(y7, p7)
    f1_7_macro = f1_score(y7, p7, average="macro", zero_division=0)

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": acc_7,
        "f1_7_macro": f1_7_macro,
        "acc_2_nonneg": acc_2_nonneg,
        "f1_2_nonneg": f1_2_nonneg,
        "acc_2_no_neutral": acc_2_no_neutral,
        "f1_2_no_neutral": f1_2_no_neutral,
    }


def evaluate(model, loader, device):
    model.eval()

    sent_true, sent_pred = [], []
    bin_true, bin_pred = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            out = model(x)

            sent_pred.extend(out["sentiment"].cpu().numpy().tolist())
            sent_true.extend(batch["sentiment"].numpy().tolist())

            pred_bin = out["binary_logits"].argmax(dim=-1).cpu().numpy()
            bin_pred.extend(pred_bin.tolist())
            bin_true.extend(batch["binary"].numpy().tolist())

    sm = compute_sentiment_metrics(sent_true, sent_pred)

    sm["binary_acc"] = accuracy_score(bin_true, bin_pred)
    sm["binary_macro_f1"] = f1_score(bin_true, bin_pred, average="macro", zero_division=0)
    sm["binary_weighted_f1"] = f1_score(bin_true, bin_pred, average="weighted", zero_division=0)

    return sm


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    paths = {
        "train_feat": "data/raw/MOSEI/train.features",
        "val_feat": "data/raw/MOSEI/val.features",
        "test_feat": "data/raw/MOSEI/test.features",
        "train_csv": "data/raw/MOSEI/Data_Train_modified.csv",
        "val_csv": "data/raw/MOSEI/Data_Val_modified.csv",
        "test_csv": "data/raw/MOSEI/Data_Test_modified.csv",
    }

    train_feat, train_bin_from_feature = load_feature_vectors(paths["train_feat"])
    val_feat, val_bin_from_feature = load_feature_vectors(paths["val_feat"])
    test_feat, test_bin_from_feature = load_feature_vectors(paths["test_feat"])

    train_df = pd.read_csv(paths["train_csv"])
    val_df = pd.read_csv(paths["val_csv"])
    test_df = pd.read_csv(paths["test_csv"])

    print("feature shapes:", train_feat.shape, val_feat.shape, test_feat.shape)
    print("csv shapes:", train_df.shape, val_df.shape, test_df.shape)

    # Use gold sentiment from CSV.
    y_train_sent = train_df["sentiment"].astype(np.float32).values
    y_val_sent = val_df["sentiment"].astype(np.float32).values
    y_test_sent = test_df["sentiment"].astype(np.float32).values

    # Binary label from sentiment. This is better than feature file's coarse label.
    y_train_bin = (y_train_sent > 0).astype(np.int64)
    y_val_bin = (y_val_sent > 0).astype(np.int64)
    y_test_bin = (y_test_sent > 0).astype(np.int64)

    emo_cols = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
    y_train_emo = train_df[emo_cols].astype(np.float32).values
    y_val_emo = val_df[emo_cols].astype(np.float32).values
    y_test_emo = test_df[emo_cols].astype(np.float32).values

    # Prefer clean human transcript; fallback to ASR.
    train_texts = train_df["text"].fillna(train_df["ASR"]).fillna("").astype(str).tolist()
    val_texts = val_df["text"].fillna(val_df["ASR"]).fillna("").astype(str).tolist()
    test_texts = test_df["text"].fillna(test_df["ASR"]).fillna("").astype(str).tolist()

    print("Fitting TF-IDF + SVD text representation...")
    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True,
        strip_accents="unicode",
    )

    x_train_tfidf = tfidf.fit_transform(train_texts)
    x_val_tfidf = tfidf.transform(val_texts)
    x_test_tfidf = tfidf.transform(test_texts)

    svd = TruncatedSVD(n_components=128, random_state=42)
    train_text = svd.fit_transform(x_train_tfidf).astype(np.float32)
    val_text = svd.transform(x_val_tfidf).astype(np.float32)
    test_text = svd.transform(x_test_tfidf).astype(np.float32)

    # Normalize feature and text parts using train stats.
    feat_scaler = StandardScaler()
    text_scaler = StandardScaler()

    train_feat = feat_scaler.fit_transform(train_feat).astype(np.float32)
    val_feat = feat_scaler.transform(val_feat).astype(np.float32)
    test_feat = feat_scaler.transform(test_feat).astype(np.float32)

    train_text = text_scaler.fit_transform(train_text).astype(np.float32)
    val_text = text_scaler.transform(val_text).astype(np.float32)
    test_text = text_scaler.transform(test_text).astype(np.float32)

    x_train = np.concatenate([train_feat, train_text], axis=1).astype(np.float32)
    x_val = np.concatenate([val_feat, val_text], axis=1).astype(np.float32)
    x_test = np.concatenate([test_feat, test_text], axis=1).astype(np.float32)

    print("final input shapes:", x_train.shape, x_val.shape, x_test.shape)
    print("sentiment train min/max:", y_train_sent.min(), y_train_sent.max())
    print("binary counts:", np.bincount(y_train_bin))

    train_ds = MOSEICSVMultitaskDataset(x_train, y_train_sent, y_train_bin, y_train_emo)
    val_ds = MOSEICSVMultitaskDataset(x_val, y_val_sent, y_val_bin, y_val_emo)
    test_ds = MOSEICSVMultitaskDataset(x_test, y_test_sent, y_test_bin, y_test_emo)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # Class weights for sentiment-derived binary task.
    counts = np.bincount(y_train_bin, minlength=2).astype(np.float32)
    weights = counts.sum() / (2 * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("binary weights:", weights.detach().cpu().numpy())

    experiments = [
        {"name": "csv_multitask_h256_lr1e4", "hidden_dim": 256, "lr": 1e-4, "dropout": 0.35, "epochs": 25},
        {"name": "csv_multitask_h384_lr8e5", "hidden_dim": 384, "lr": 8e-5, "dropout": 0.40, "epochs": 25},
        {"name": "csv_multitask_h256_lr5e5", "hidden_dim": 256, "lr": 5e-5, "dropout": 0.30, "epochs": 25},
    ]

    rows = []

    for exp in experiments:
        print("\n==========", exp["name"], "==========")

        model = MOSEICSVMultitaskModel(
            input_dim=x_train.shape[1],
            hidden_dim=exp["hidden_dim"],
            dropout=exp["dropout"],
        ).to(device)

        reg_loss = nn.SmoothL1Loss()
        ce_loss = nn.CrossEntropyLoss(weight=weights)
        bce_loss = nn.BCEWithLogitsLoss()

        opt = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-4)

        best_score = -999
        best_state = None

        for epoch in range(1, exp["epochs"] + 1):
            model.train()
            total_loss = 0.0
            total_n = 0

            for batch in train_loader:
                x = batch["x"].to(device)
                sentiment = batch["sentiment"].to(device)
                binary = batch["binary"].to(device)
                emotions = batch["emotions"].to(device)

                out = model(x)

                loss_sent = reg_loss(out["sentiment"], sentiment)
                loss_bin = ce_loss(out["binary_logits"], binary)
                loss_emo = bce_loss(out["emotion_logits"], emotions)

                # Multi-task objective.
                loss = loss_sent + 0.5 * loss_bin + 0.2 * loss_emo

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                total_loss += loss.item() * x.size(0)
                total_n += x.size(0)

            val_m = evaluate(model, val_loader, device)

            # Select checkpoint by regression + binary quality.
            score = (
                val_m["corr"]
                - 0.2 * val_m["mae"]
                + val_m["acc_2_no_neutral"]
                + val_m["f1_2_no_neutral"]
                + 0.5 * val_m["binary_macro_f1"]
            )

            print(
                f"Epoch {epoch:02d} | "
                f"loss={total_loss/total_n:.4f} "
                f"val_mae={val_m['mae']:.4f} "
                f"val_corr={val_m['corr']:.4f} "
                f"val_acc2={val_m['acc_2_no_neutral']:.4f} "
                f"val_f1_2={val_m['f1_2_no_neutral']:.4f} "
                f"val_acc7={val_m['acc_7']:.4f} "
                f"val_f1_7={val_m['f1_7_macro']:.4f}"
            )

            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        test_m = evaluate(model, test_loader, device)

        ckpt = f"checkpoints/mosei_{exp['name']}.pt"
        torch.save(model.state_dict(), ckpt)

        row = {"model": exp["name"], "checkpoint": ckpt, "best_val_score": best_score, **test_m}
        rows.append(row)

        print("TEST:", row)

        pd.DataFrame(rows).to_csv("outputs/mosei_csv_multitask_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/mosei_csv_multitask_results.csv", index=False)

    print("\n========== MOSEI CSV Multi-task Results ==========")
    print(df.to_string(index=False))

    print("\nBest by Acc-2 no neutral")
    print(df.sort_values("acc_2_no_neutral", ascending=False).head(5).to_string(index=False))

    print("\nBest by MAE")
    print(df.sort_values("mae", ascending=True).head(5).to_string(index=False))

    print("\nBest by Corr")
    print(df.sort_values("corr", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()

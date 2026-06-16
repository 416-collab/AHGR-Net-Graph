import os
import sys
import yaml
import torch
import random
import itertools
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.mosi_pkl_dataset import MOSIPKLDataset
from models.graph_sa import GRAPHSA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_datasets(train_ds, val_ds, test_ds):
    text_mean = train_ds.text.mean(axis=0, keepdims=True)
    text_std = train_ds.text.std(axis=0, keepdims=True) + 1e-6

    audio_mean = train_ds.audio.mean(axis=0, keepdims=True)
    audio_std = train_ds.audio.std(axis=0, keepdims=True) + 1e-6

    for ds in [train_ds, val_ds, test_ds]:
        ds.text = ((ds.text - text_mean) / text_std).astype(np.float32)
        ds.audio = ((ds.audio - audio_mean) / audio_std).astype(np.float32)

    return train_ds, val_ds, test_ds


def class_weights_from_dataset(ds, num_classes=7):
    labels = ds.label_class
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts


def compute_metrics(y_raw, pred_cls):
    y_raw = np.asarray(y_raw).reshape(-1)
    pred_cls = np.asarray(pred_cls).reshape(-1)

    y_cls = np.rint(np.clip(y_raw, -3, 3) + 3).astype(int)
    y_cls = np.clip(y_cls, 0, 6)
    pred_score = pred_cls.astype(np.float32) - 3.0

    corr = 0.0
    if len(np.unique(pred_score)) > 1:
        corr = pearsonr(y_raw, pred_score)[0]

    mask = y_raw != 0
    if mask.sum() > 0:
        y_bin = (y_raw[mask] > 0).astype(int)
        p_bin = (pred_score[mask] > 0).astype(int)
        acc_2_no_neutral = accuracy_score(y_bin, p_bin)
        f1_2_no_neutral = f1_score(y_bin, p_bin, average="weighted", zero_division=0)
    else:
        acc_2_no_neutral = 0.0
        f1_2_no_neutral = 0.0

    return {
        "acc_7": accuracy_score(y_cls, pred_cls),
        "f1_7_macro": f1_score(y_cls, pred_cls, average="macro", zero_division=0),
        "mae": mean_absolute_error(y_raw, pred_score),
        "corr": corr,
        "acc_2_nonneg": accuracy_score((y_raw >= 0).astype(int), (pred_score >= 0).astype(int)),
        "f1_2_nonneg": f1_score((y_raw >= 0).astype(int), (pred_score >= 0).astype(int), average="weighted", zero_division=0),
        "acc_2_no_neutral": acc_2_no_neutral,
        "f1_2_no_neutral": f1_2_no_neutral,
    }


def predict(model, loader, device):
    model.eval()
    preds, y_raw = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["text"].to(device),
                batch["audio"].to(device),
                batch["context"].to(device),
            )
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            y_raw.extend(batch["label_raw"].numpy().reshape(-1).tolist())

    return np.array(y_raw), np.array(preds)


def run_one(cfg, params, device):
    set_seed(params["seed"])

    train_ds = MOSIPKLDataset(
        cfg["data_path"],
        split="train",
        max_context=1,
        num_aspects=cfg["num_aspects"],
        use_vision=True,
    )
    val_ds = MOSIPKLDataset(
        cfg["data_path"],
        split="valid",
        max_context=1,
        num_aspects=cfg["num_aspects"],
        use_vision=True,
    )
    test_ds = MOSIPKLDataset(
        cfg["data_path"],
        split="test",
        max_context=1,
        num_aspects=cfg["num_aspects"],
        use_vision=True,
    )

    train_ds, val_ds, test_ds = normalize_datasets(train_ds, val_ds, test_ds)

    weights, counts = class_weights_from_dataset(train_ds, num_classes=cfg["num_classes"])
    print("Train class counts:", counts.tolist())
    print("Class weights:", weights.numpy().round(3).tolist())

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False)

    model = GRAPHSA(
        text_dim=cfg["text_dim"],
        audio_dim=25,
        hidden_dim=params["hidden_dim"],
        num_classes=cfg["num_classes"],
        num_aspects=cfg["num_aspects"],
        use_graph=True,
    ).to(device)

    ce = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    best_score = -999
    best_state = None

    for epoch in range(1, params["epochs"] + 1):
        model.train()

        for batch in train_loader:
            out = model(
                batch["text"].to(device),
                batch["audio"].to(device),
                batch["context"].to(device),
                return_features=True,
            )

            labels = batch["label"].to(device)
            aspects = batch["aspects"].to(device)

            loss = ce(out["logits"], labels)

            if params["aspect_loss_weight"] > 0:
                loss = loss + params["aspect_loss_weight"] * bce(out["aspect_logits"], aspects)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        y_val, p_val = predict(model, val_loader, device)
        vm = compute_metrics(y_val, p_val)

        score = (
            vm["acc_7"]
            + vm["f1_7_macro"]
            + vm["corr"]
            + vm["acc_2_no_neutral"]
            - 0.1 * vm["mae"]
        )

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"val_acc7={vm['acc_7']:.4f} "
                f"val_f1={vm['f1_7_macro']:.4f} "
                f"val_mae={vm['mae']:.4f} "
                f"val_corr={vm['corr']:.4f} "
                f"val_acc2={vm['acc_2_no_neutral']:.4f}"
            )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_test, p_test = predict(model, test_loader, device)
    tm = compute_metrics(y_test, p_test)

    row = dict(params)
    row["best_val_score"] = best_score
    row.update(tm)

    ckpt = (
        f"checkpoints/weighted_trimodal_graphsa_seed{params['seed']}"
        f"_lr{params['lr']}_aw{params['aspect_loss_weight']}.pt"
    )
    torch.save(model.state_dict(), ckpt)
    row["checkpoint"] = ckpt

    return row


def main():
    with open("configs/config_mosi_trimodal.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    grid = {
        "seed": [42, 43, 44],
        "lr": [5e-5, 8e-5, 1e-4],
        "hidden_dim": [128],
        "aspect_loss_weight": [0.0, 0.03, 0.05],
        "batch_size": [16],
        "epochs": [35],
        "weight_decay": [1e-4],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    rows = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"\n===== Weighted trimodal run {i}/{len(combos)} =====")
        print(params)

        row = run_one(cfg, params, device)
        print(row)

        rows.append(row)
        pd.DataFrame(rows).to_csv("outputs/mosi_trimodal_weighted_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv("outputs/mosi_trimodal_weighted.csv", index=False)

    print("\nBest by acc_7")
    print(df.sort_values("acc_7", ascending=False).head(10).to_string(index=False))

    print("\nBest by f1_7_macro")
    print(df.sort_values("f1_7_macro", ascending=False).head(10).to_string(index=False))

    print("\nBest by mae")
    print(df.sort_values("mae", ascending=True).head(10).to_string(index=False))

    print("\nBest by corr")
    print(df.sort_values("corr", ascending=False).head(10).to_string(index=False))

    print("\nBest by acc_2_no_neutral")
    print(df.sort_values("acc_2_no_neutral", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()

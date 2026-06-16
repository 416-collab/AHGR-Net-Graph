import os
import sys
import yaml
import torch
import random
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
from models.dc2m_baseline import DC2MBaseline
from models.graph_sa import GRAPHSA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        y_bin_np = (y_raw[mask] > 0).astype(int)
        p_bin_np = (pred_score[mask] > 0).astype(int)
        acc_2_no_neutral = accuracy_score(y_bin_np, p_bin_np)
        f1_2_no_neutral = f1_score(y_bin_np, p_bin_np, average="weighted", zero_division=0)
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


def train_model(name, model, cfg, train_loader, val_loader, test_loader, device, aspect_loss_weight=0.0):
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)

    best_score = -999
    best_state = None

    print(f"\n========== Ablation: {name} ==========")

    for epoch in range(1, 31):
        model.train()

        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            context = batch["context"].to(device)
            labels = batch["label"].to(device)
            aspects = batch["aspects"].to(device)

            if name == "DC2M_Baseline":
                logits = model(text, audio, context)
                loss = ce(logits, labels)
            else:
                out = model(text, audio, context, return_features=True)
                logits = out["logits"]
                loss = ce(logits, labels)
                if aspect_loss_weight > 0:
                    loss = loss + aspect_loss_weight * bce(out["aspect_logits"], aspects)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        y_val, p_val = predict(model, val_loader, device)
        vm = compute_metrics(y_val, p_val)

        score = vm["acc_7"] + vm["f1_7_macro"] + vm["corr"] - 0.1 * vm["mae"]

        print(
            f"Epoch {epoch:02d} | "
            f"val_acc7={vm['acc_7']:.4f} "
            f"val_f1={vm['f1_7_macro']:.4f} "
            f"val_mae={vm['mae']:.4f} "
            f"val_corr={vm['corr']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y_test, p_test = predict(model, test_loader, device)
    tm = compute_metrics(y_test, p_test)

    ckpt = f"checkpoints/ablation_{name}.pt"
    torch.save(model.state_dict(), ckpt)

    row = {"model": name, "checkpoint": ckpt, "best_val_score": best_score}
    row.update(tm)
    print("TEST:", row)
    return row


def main():
    with open("configs/config_mosi.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(43)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MOSIPKLDataset(cfg["data_path"], "train", max_context=1, num_aspects=cfg["num_aspects"])
    val_ds = MOSIPKLDataset(cfg["data_path"], "valid", max_context=1, num_aspects=cfg["num_aspects"])
    test_ds = MOSIPKLDataset(cfg["data_path"], "test", max_context=1, num_aspects=cfg["num_aspects"])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    rows = []

    baseline = DC2MBaseline(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=128,
        num_classes=cfg["num_classes"],
    ).to(device)
    rows.append(train_model("DC2M_Baseline", baseline, cfg, train_loader, val_loader, test_loader, device))

    full = GRAPHSA(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=128,
        num_classes=cfg["num_classes"],
        num_aspects=cfg["num_aspects"],
        use_graph=True,
    ).to(device)
    rows.append(train_model("GRAPH_SA_Full", full, cfg, train_loader, val_loader, test_loader, device, aspect_loss_weight=0.05))

    no_graph = GRAPHSA(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=128,
        num_classes=cfg["num_classes"],
        num_aspects=cfg["num_aspects"],
        use_graph=False,
    ).to(device)
    rows.append(train_model("GRAPH_SA_NoGraph", no_graph, cfg, train_loader, val_loader, test_loader, device, aspect_loss_weight=0.05))

    no_aspect_loss = GRAPHSA(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=128,
        num_classes=cfg["num_classes"],
        num_aspects=cfg["num_aspects"],
        use_graph=True,
    ).to(device)
    rows.append(train_model("GRAPH_SA_NoAspectLoss", no_aspect_loss, cfg, train_loader, val_loader, test_loader, device, aspect_loss_weight=0.0))

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/mosi_real_ablation_results.csv", index=False)

    print("\n========== MOSI Real Ablation Results ==========")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

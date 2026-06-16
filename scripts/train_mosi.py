import os
import sys
import yaml
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

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


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            context = batch["context"].to(device)
            y = batch["label"].cpu().numpy()

            logits = model(text, audio, context)
            pred = logits.argmax(dim=-1).cpu().numpy()

            preds.extend(pred.tolist())
            labels.extend(y.tolist())

    return {
        "acc_7": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "mae_class": mean_absolute_error(labels, preds),
    }


def train_one_model(model_name, model, cfg, train_loader, val_loader, test_loader, device):
    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    best_val = -1.0
    best_path = f"checkpoints/{model_name}_mosi.pt"
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n========== Training {model_name} ==========")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            context = batch["context"].to(device)
            labels = batch["label"].to(device)
            aspects = batch["aspects"].to(device)

            optimizer.zero_grad()

            if model_name == "graph_sa":
                out = model(text, audio, context, return_features=True)
                logits = out["logits"]
                loss = ce_loss(logits, labels)
                loss = loss + cfg.get("aspect_loss_weight", 0.05) * bce_loss(out["aspect_logits"], aspects)
            else:
                logits = model(text, audio, context)
                loss = ce_loss(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_n += bs

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics["acc_7"]

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc7={val_metrics['acc_7']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"val_mae={val_metrics['mae_class']:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"Best validation Acc-7 for {model_name}: {best_val:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test metrics for {model_name}: {test_metrics}")

    return {
        "model": model_name,
        "best_val_acc7": best_val,
        **test_metrics,
    }


def main():
    with open("configs/config_mosi.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
    )
    print("Using device:", device)

    train_ds = MOSIPKLDataset(cfg["data_path"], split="train", max_context=cfg["max_context"], num_aspects=cfg["num_aspects"])
    val_ds = MOSIPKLDataset(cfg["data_path"], split="valid", max_context=cfg["max_context"], num_aspects=cfg["num_aspects"])
    test_ds = MOSIPKLDataset(cfg["data_path"], split="test", max_context=cfg["max_context"], num_aspects=cfg["num_aspects"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    results = []

    baseline = DC2MBaseline(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
    ).to(device)

    results.append(
        train_one_model("dc2m_baseline", baseline, cfg, train_loader, val_loader, test_loader, device)
    )

    graph_sa = GRAPHSA(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        num_aspects=cfg["num_aspects"],
        use_graph=True,
    ).to(device)

    results.append(
        train_one_model("graph_sa", graph_sa, cfg, train_loader, val_loader, test_loader, device)
    )

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("outputs/mosi_results.csv", index=False)

    print("\n========== Final MOSI Results ==========")
    print(df)


if __name__ == "__main__":
    main()

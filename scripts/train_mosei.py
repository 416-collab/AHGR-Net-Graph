import os
import sys
import yaml
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.mosei_features_dataset import MOSEIFeaturesDataset, compute_normalize_stats
from models.dc2m_baseline import DC2MBaseline
from models.graph_sa import GRAPHSA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels, num_classes=2):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["text"].to(device),
                batch["audio"].to(device),
                batch["context"].to(device),
            )
            pred = logits.argmax(dim=-1).cpu().numpy()
            y = batch["label"].numpy()

            preds.extend(pred.tolist())
            labels.extend(y.tolist())

    labels = np.array(labels)
    preds = np.array(preds)

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def train_one_model(name, model, cfg, train_loader, val_loader, test_loader, device, class_weights, aspect_loss_weight=0.0):
    ce = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    bce = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    best_score = -999.0
    best_state = None

    print(f"\n========== Training {name} ==========")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in train_loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            context = batch["context"].to(device)
            labels = batch["label"].to(device)
            aspects = batch["aspects"].to(device)

            if name == "DC2M_MOSEI":
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

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_n += bs

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        val_metrics = evaluate(model, val_loader, device)

        # Balanced selection: macro-F1 + accuracy, because MOSEI is imbalanced.
        val_score = val_metrics["macro_f1"] + val_metrics["acc"]

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.4f} "
            f"val_recall_macro={val_metrics['recall_macro']:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/{name}.pt"
    torch.save(model.state_dict(), ckpt)

    test_metrics = evaluate(model, test_loader, device)
    row = {
        "model": name,
        "checkpoint": ckpt,
        "best_val_score": best_score,
        **test_metrics,
    }

    print("TEST:", row)
    return row


def main():
    with open("configs/config_mosei.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    print("Using device:", device)

    stats = compute_normalize_stats(cfg["train_path"])

    train_ds = MOSEIFeaturesDataset(
        cfg["train_path"],
        max_context=cfg["max_context"],
        num_aspects=cfg["num_aspects"],
        normalize_stats=stats,
    )
    val_ds = MOSEIFeaturesDataset(
        cfg["val_path"],
        max_context=cfg["max_context"],
        num_aspects=cfg["num_aspects"],
        normalize_stats=stats,
    )
    test_ds = MOSEIFeaturesDataset(
        cfg["test_path"],
        max_context=cfg["max_context"],
        num_aspects=cfg["num_aspects"],
        normalize_stats=stats,
    )

    class_weights, counts = compute_class_weights(train_ds.labels, num_classes=cfg["num_classes"])
    print("Train class counts:", counts.tolist())
    print("Class weights:", class_weights.numpy().round(4).tolist())

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    rows = []

    baseline = DC2MBaseline(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
    ).to(device)

    rows.append(
        train_one_model(
            "DC2M_MOSEI",
            baseline,
            cfg,
            train_loader,
            val_loader,
            test_loader,
            device,
            class_weights,
            aspect_loss_weight=0.0,
        )
    )

    graphsa = GRAPHSA(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        num_aspects=cfg["num_aspects"],
        use_graph=True,
    ).to(device)

    rows.append(
        train_one_model(
            "GRAPH_SA_MOSEI",
            graphsa,
            cfg,
            train_loader,
            val_loader,
            test_loader,
            device,
            class_weights,
            aspect_loss_weight=cfg["aspect_loss_weight"],
        )
    )

    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/mosei_results.csv", index=False)

    print("\n========== MOSEI Results ==========")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

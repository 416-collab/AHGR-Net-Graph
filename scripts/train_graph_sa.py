import os
import sys
import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.synthetic_dataset import SyntheticMSADataset
from models.graph_sa import GRAPHSA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def run_epoch(model, loader, optimizer, device, train=True, aspect_loss_weight=0.2):
    model.train() if train else model.eval()

    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    total_loss, total_acc, total_aspect_loss, total_n = 0.0, 0.0, 0.0, 0

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        context = batch["context"].to(device)
        labels = batch["label"].to(device)
        aspects = batch["aspects"].to(device)

        with torch.set_grad_enabled(train):
            out = model(text, audio, context, return_features=True)

            sentiment_loss = ce_loss(out["logits"], labels)
            aspect_loss = bce_loss(out["aspect_logits"], aspects)

            loss = sentiment_loss + aspect_loss_weight * aspect_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(out["logits"], labels) * bs
        total_aspect_loss += aspect_loss.item() * bs
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "acc": total_acc / total_n,
        "aspect_loss": total_aspect_loss / total_n,
    }


def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
    )

    print("Using device:", device)

    train_ds = SyntheticMSADataset(
        size=cfg["train_size"],
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        max_context=cfg["max_context"],
        num_classes=cfg["num_classes"],
        seed=cfg["seed"],
    )

    val_ds = SyntheticMSADataset(
        size=cfg["val_size"],
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        max_context=cfg["max_context"],
        num_classes=cfg["num_classes"],
        seed=cfg["seed"] + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = GRAPHSA(
        text_dim=cfg["text_dim"],
        audio_dim=cfg["audio_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        num_aspects=cfg["num_aspects"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    best_val = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True, aspect_loss_weight=cfg.get('aspect_loss_weight', 0.2))
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False, aspect_loss_weight=cfg.get('aspect_loss_weight', 0.2))

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"train_aspect_loss={train_metrics['aspect_loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_aspect_loss={val_metrics['aspect_loss']:.4f}"
        )

        if val_metrics["acc"] > best_val:
            best_val = val_metrics["acc"]
            torch.save(model.state_dict(), "checkpoints/graph_sa.pt")

    print("Best val acc:", best_val)


if __name__ == "__main__":
    main()

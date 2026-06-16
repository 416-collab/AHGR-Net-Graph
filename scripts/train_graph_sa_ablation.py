import os
import sys
import yaml
import torch
import random
import argparse
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


def run_epoch(model, loader, optimizer, device, train=True, aspect_loss_weight=0.05):
    model.train() if train else model.eval()

    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()

    total_loss, total_acc, total_n = 0.0, 0.0, 0

    for batch in loader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        context = batch["context"].to(device)
        labels = batch["label"].to(device)
        aspects = batch["aspects"].to(device)

        with torch.set_grad_enabled(train):
            out = model(text, audio, context, return_features=True)
            loss = ce_loss(out["logits"], labels)

            if aspect_loss_weight > 0:
                loss = loss + aspect_loss_weight * bce_loss(out["aspect_logits"], aspects)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(out["logits"], labels) * bs
        total_n += bs

    return total_loss / total_n, total_acc / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--use_graph", type=int, default=1)
    parser.add_argument("--aspect_loss_weight", type=float, default=0.05)
    args = parser.parse_args()

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Experiment:", args.name)
    print("use_graph:", bool(args.use_graph))
    print("aspect_loss_weight:", args.aspect_loss_weight)
    print("device:", device)

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
        use_graph=bool(args.use_graph),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    best_val = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, device,
            train=True,
            aspect_loss_weight=args.aspect_loss_weight,
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, device,
            train=False,
            aspect_loss_weight=args.aspect_loss_weight,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), f"checkpoints/{args.name}.pt")

    print("Best val acc:", best_val)


if __name__ == "__main__":
    main()

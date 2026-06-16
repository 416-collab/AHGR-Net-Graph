import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class MOSEISequenceDataset(Dataset):
    """
    MOSEI sequence loader.

    Each raw sample:
      feature: (64, T)
      label: 0/1

    We convert to:
      sequence: (T, 64)
      label: int
    """

    def __init__(self, path, max_len=512, normalize_stats=None):
        super().__init__()
        self.path = path
        self.max_len = max_len

        with open(path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        self.labels = np.array([int(x["label"]) for x in raw], dtype=np.int64)
        self.seqs = []

        for x in raw:
            feat = x["feature"].astype(np.float32)
            feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)

            # (64, T) -> (T, 64)
            seq = feat.T.astype(np.float32)

            # truncate long sequences
            if seq.shape[0] > max_len:
                seq = seq[:max_len]

            self.seqs.append(seq)

        if normalize_stats is not None:
            mean, std = normalize_stats
            self.seqs = [((s - mean) / std).astype(np.float32) for s in self.seqs]

        print(f"Loaded MOSEI sequence {path}: samples={len(self.seqs)}, max_len={max_len}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return {
            "seq": torch.tensor(seq, dtype=torch.float32),
            "length": torch.tensor(seq.shape[0], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_sequence_normalize_stats(path, max_len=512):
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    all_frames = []
    for x in raw:
        feat = x["feature"].astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
        seq = feat.T.astype(np.float32)
        if seq.shape[0] > max_len:
            seq = seq[:max_len]
        all_frames.append(seq)

    arr = np.concatenate(all_frames, axis=0)
    mean = arr.mean(axis=0, keepdims=True).astype(np.float32)
    std = (arr.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    return mean, std


def collate_mosei_sequence(batch):
    lengths = torch.stack([b["length"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])

    max_len = int(lengths.max().item())
    feat_dim = batch[0]["seq"].shape[1]

    seqs = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)

    for i, b in enumerate(batch):
        L = b["seq"].shape[0]
        seqs[i, :L] = b["seq"]

    return {
        "seq": seqs,
        "length": lengths,
        "label": labels,
    }

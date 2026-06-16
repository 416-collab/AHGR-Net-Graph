import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class MOSEIFeaturesDataset(Dataset):
    """
    Loader for MOSEI .features files.

    Each sample:
      feature: numpy array (64, T)
      label: int 0/1

    We convert variable-length sequence into fixed vector:
      mean pooling (64) + max pooling (64) = 128-dim vector

    To reuse DC2M/GRAPH-SA:
      text    = pooled feature vector (128,)
      audio   = dummy scalar (1,)
      context = previous K pooled vectors (K, 128)
      label   = binary class
      aspects = pseudo aspect vector from pooled features
    """

    def __init__(self, path, max_context=3, num_aspects=12, normalize_stats=None):
        super().__init__()
        self.path = path
        self.max_context = max_context
        self.num_aspects = num_aspects

        with open(path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")

        self.labels = np.array([int(x["label"]) for x in raw], dtype=np.int64)

        pooled = []
        for x in raw:
            feat = x["feature"].astype(np.float32)  # (64, T)
            feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
            mean_pool = feat.mean(axis=1)
            max_pool = feat.max(axis=1)
            vec = np.concatenate([mean_pool, max_pool], axis=0)  # (128,)
            pooled.append(vec)

        self.text = np.stack(pooled, axis=0).astype(np.float32)

        if normalize_stats is not None:
            mean, std = normalize_stats
            self.text = ((self.text - mean) / std).astype(np.float32)

        self.audio = np.zeros((len(self.text), 1), dtype=np.float32)

        aspect_source = self.text[:, :num_aspects]
        self.aspects = (aspect_source > aspect_source.mean(axis=0, keepdims=True)).astype(np.float32)

        print(
            f"Loaded MOSEI from {path}: "
            f"text={self.text.shape}, audio={self.audio.shape}, "
            f"labels={self.labels.shape}, aspects={self.aspects.shape}"
        )

    def __len__(self):
        return len(self.labels)

    def _get_context(self, idx):
        ctx = []
        for offset in range(self.max_context, 0, -1):
            j = idx - offset
            if j >= 0:
                ctx.append(self.text[j])
            else:
                ctx.append(np.zeros_like(self.text[idx]))
        return np.stack(ctx, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        return {
            "text": torch.tensor(self.text[idx], dtype=torch.float32),
            "audio": torch.tensor(self.audio[idx], dtype=torch.float32),
            "context": torch.tensor(self._get_context(idx), dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "label_raw": torch.tensor(float(self.labels[idx]), dtype=torch.float32),
            "aspects": torch.tensor(self.aspects[idx], dtype=torch.float32),
        }


def compute_normalize_stats(path):
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    pooled = []
    for x in raw:
        feat = x["feature"].astype(np.float32)
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)
        mean_pool = feat.mean(axis=1)
        max_pool = feat.max(axis=1)
        pooled.append(np.concatenate([mean_pool, max_pool], axis=0))

    arr = np.stack(pooled, axis=0).astype(np.float32)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-6
    return mean, std

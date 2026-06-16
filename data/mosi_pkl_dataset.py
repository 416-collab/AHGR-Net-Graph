import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class MOSIPKLDataset(Dataset):
    """
    Loader for preprocessed CMU-MOSI mosi_data.pkl.

    Expected structure:
    data['train'/'valid'/'test'] contains:
      text   : (N, 50, 300)
      audio  : (N, 50, 5)
      vision : (N, 50, 20)
      labels : (N, 1, 1)

    For our current DC2M/GRAPH-SA implementation:
      text    -> mean pooled to (300,)
      audio   -> mean pooled to (5,)
      context -> previous K utterance text vectors, shape (K, 300)
      label   -> 7-class label in {0,...,6}
      aspects -> pseudo aspect labels from text dimensions
    """

    def __init__(self, pkl_path, split="train", max_context=3, num_aspects=12, use_vision=False):
        super().__init__()
        self.pkl_path = pkl_path
        self.split = split
        self.max_context = max_context
        self.num_aspects = num_aspects
        self.use_vision = use_vision

        with open(pkl_path, "rb") as f:
            all_data = pickle.load(f, encoding="latin1")

        if split not in all_data:
            raise ValueError(f"Split {split} not found. Available: {list(all_data.keys())}")

        data = all_data[split]

        self.text_seq = data["text"].astype(np.float32)      # (N, 50, 300)
        self.audio_seq = data["audio"].astype(np.float32)    # (N, 50, 5)
        self.vision_seq = data["vision"].astype(np.float32)  # (N, 50, 20)
        self.labels_raw = data["labels"].astype(np.float32)  # (N, 1, 1)
        self.ids = data.get("id", None)

        # Mean pooling over time dimension.
        self.text = self.text_seq.mean(axis=1)    # (N, 300)
        audio_pooled = self.audio_seq.mean(axis=1)  # (N, 5)
        vision_pooled = self.vision_seq.mean(axis=1)  # (N, 20)

        if self.use_vision:
            self.audio = np.concatenate([audio_pooled, vision_pooled], axis=1)  # (N, 25)
        else:
            self.audio = audio_pooled

        # Raw MOSI labels are usually continuous sentiment in [-3, 3].
        labels = self.labels_raw.reshape(-1)
        labels = np.clip(labels, -3, 3)

        # Convert regression labels [-3,3] to 7-class labels [0,6].
        self.label_class = np.rint(labels + 3).astype(np.int64)
        self.label_class = np.clip(self.label_class, 0, 6)

        # Pseudo aspect labels from first num_aspects text dimensions.
        # Later we can replace this with lexicon/aspect extraction.
        aspect_source = self.text[:, :num_aspects]
        self.aspects = (aspect_source > 0).astype(np.float32)

        print(
            f"Loaded MOSI {split}: "
            f"text={self.text.shape}, audio={self.audio.shape}, "
            f"labels={self.label_class.shape}, aspects={self.aspects.shape}"
        )

    def __len__(self):
        return len(self.label_class)

    def _get_context(self, idx):
        """
        Use previous K utterance text vectors as dynamic context.
        For early samples, pad with zeros.
        """
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
            "label": torch.tensor(self.label_class[idx], dtype=torch.long),
            "label_raw": torch.tensor(self.labels_raw.reshape(-1)[idx], dtype=torch.float32),
            "aspects": torch.tensor(self.aspects[idx], dtype=torch.float32),
        }

import torch
from torch.utils.data import Dataset

class SyntheticMSADataset(Dataset):
    """
    Synthetic multimodal sentiment dataset.

    Label is generated from:
    - text signal
    - audio signal
    - aspect signal

    This makes GRAPH-SA meaningful because aspects contain label-relevant information.
    """
    def __init__(self, size=1000, text_dim=768, audio_dim=74, max_context=3, num_classes=7, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        self.size = size
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.max_context = max_context
        self.num_classes = num_classes
        self.num_aspects = 12

        self.text = torch.randn(size, text_dim, generator=g)
        self.audio = torch.randn(size, audio_dim, generator=g)
        self.context = torch.randn(size, max_context, text_dim, generator=g)

        # Aspect presence is generated from first 12 text dimensions
        aspect_logits = self.text[:, :self.num_aspects] + 0.25 * torch.randn(size, self.num_aspects, generator=g)
        self.aspects = (torch.sigmoid(aspect_logits) > 0.5).float()

        # Some aspects are positive, some negative
        aspect_polarity = torch.tensor([1.2, 1.0, 0.8, 0.6, -1.2, -1.0, -0.8, -0.6, 0.4, -0.4, 0.2, -0.2])

        text_score = self.text[:, :8].sum(dim=1)
        audio_score = 0.5 * self.audio[:, :4].sum(dim=1)
        context_score = 0.3 * self.context[:, :, :4].mean(dim=(1, 2))
        aspect_score = (self.aspects * aspect_polarity).sum(dim=1)

        score = text_score + audio_score + context_score + 1.5 * aspect_score

        bins = torch.quantile(score, torch.linspace(0, 1, num_classes + 1))
        self.labels = torch.bucketize(score, bins[1:-1]).long()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "text": self.text[idx],
            "audio": self.audio[idx],
            "context": self.context[idx],
            "label": self.labels[idx],
            "aspects": self.aspects[idx],
        }

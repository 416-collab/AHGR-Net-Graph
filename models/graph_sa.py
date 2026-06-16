import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dc2m_baseline import DC2MBaseline


class AspectGraphReasoner(nn.Module):
    """
    Lightweight aspect graph reasoning.
    Uses learnable aspect embeddings + learned adjacency.
    """
    def __init__(self, num_aspects=12, hidden_dim=256):
        super().__init__()
        self.num_aspects = num_aspects
        self.aspect_emb = nn.Embedding(num_aspects, hidden_dim)

        self.adj_logits = nn.Parameter(torch.randn(num_aspects, num_aspects) * 0.02)

        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self):
        aspect_ids = torch.arange(self.num_aspects, device=self.adj_logits.device)
        x = self.aspect_emb(aspect_ids)

        adj = torch.softmax(self.adj_logits, dim=-1)
        x_msg = adj @ x
        x_out = self.update(x_msg) + x
        return x_out, adj


class GRAPHSA(nn.Module):
    """
    GRAPH-SA:
    DC2M baseline + aspect graph reasoning.
    """
    def __init__(self, text_dim=768, audio_dim=74, hidden_dim=256, num_classes=7, num_aspects=12, use_graph=True):
        super().__init__()

        self.use_graph = use_graph

        self.backbone = DC2MBaseline(
            text_dim=text_dim,
            audio_dim=audio_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

        self.aspect_graph = AspectGraphReasoner(
            num_aspects=num_aspects,
            hidden_dim=hidden_dim,
        )

        self.fused_reduce = nn.Linear(hidden_dim * 4, hidden_dim)

        self.aspect_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_aspects),
        )

        self.aspect_sentiment_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, audio, context, return_features=False):
        base = self.backbone(text, audio, context, return_features=True)

        fused = self.fused_reduce(base["fused"])
        aspect_nodes, adj = self.aspect_graph()

        if not self.use_graph:
            adj = torch.eye(adj.size(0), device=adj.device)
            aspect_nodes = self.aspect_graph.aspect_emb(torch.arange(self.aspect_graph.num_aspects, device=adj.device))

        aspect_logits = self.aspect_detector(fused)
        aspect_weights = torch.sigmoid(aspect_logits)

        aspect_context = aspect_weights @ aspect_nodes

        enhanced = self.aspect_sentiment_fusion(
            torch.cat([fused, aspect_context], dim=-1)
        )

        logits = self.classifier(enhanced)

        if return_features:
            return {
                "logits": logits,
                "aspect_logits": aspect_logits,
                "aspect_weights": aspect_weights,
                "aspect_adj": adj,
                "base": base,
            }

        return logits

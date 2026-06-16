import torch
import torch.nn as nn
import torch.nn.functional as F

class DC2MBaseline(nn.Module):
    """
    Simplified DC2M-style model:
    - Text encoder projection
    - Audio encoder projection
    - Adaptive context attention over previous utterance embeddings
    - Text-audio fusion
    - Context-text fusion
    - Sentiment classification
    """
    def __init__(self, text_dim=768, audio_dim=74, hidden_dim=256, num_classes=7):
        super().__init__()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.context_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.text_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        self.context_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def adaptive_context_mining(self, text_h, context_h):
        # text_h: [B, H]
        # context_h: [B, K, H]
        sim = F.cosine_similarity(text_h.unsqueeze(1), context_h, dim=-1)
        weights = torch.softmax(sim, dim=-1)
        context_vec = torch.sum(weights.unsqueeze(-1) * context_h, dim=1)
        return context_vec, weights

    def forward(self, text, audio, context, return_features=False):
        text_h = self.text_proj(text)
        audio_h = self.audio_proj(audio)
        context_h = self.context_proj(context)

        context_vec, context_weights = self.adaptive_context_mining(text_h, context_h)

        # Expert 1: text-audio fusion
        ta_out, _ = self.text_audio_attn(
            query=text_h.unsqueeze(1),
            key=audio_h.unsqueeze(1),
            value=audio_h.unsqueeze(1)
        )
        ta_out = ta_out.squeeze(1)

        # Expert 2: context-text fusion
        ct_out, _ = self.context_text_attn(
            query=context_vec.unsqueeze(1),
            key=text_h.unsqueeze(1),
            value=text_h.unsqueeze(1)
        )
        ct_out = ct_out.squeeze(1)

        fused = torch.cat([text_h, audio_h, ta_out, ct_out], dim=-1)
        logits = self.classifier(fused)

        if return_features:
            return {
                "logits": logits,
                "fused": fused,
                "text_h": text_h,
                "audio_h": audio_h,
                "context_vec": context_vec,
                "context_weights": context_weights,
            }

        return logits

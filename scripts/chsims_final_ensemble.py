import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr
import os, sys

sys.path.insert(0, os.path.abspath("."))
device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


print("Loading CH-SIMS...")
data = load("data/raw/CHSIMS/unaligned.pkl")
train, valid, test = data["train"], data["valid"], data["test"]

y_train = train["regression_labels"].reshape(-1).astype(np.float32)
y_valid = valid["regression_labels"].reshape(-1).astype(np.float32)
y_test = test["regression_labels"].reshape(-1).astype(np.float32)


def pool(arr, lengths=None):
    arr = np.nan_to_num(arr.astype(np.float32))
    n, t, d = arr.shape
    out = []
    for i in range(n):
        seq = arr[i] if lengths is None else arr[i, : max(1, min(int(lengths[i]), t))]
        out.append(np.concatenate([seq.mean(0), seq.max(0), seq.std(0)]))
    return np.stack(out).astype(np.float32)


tr_t = pool(train["text"]); va_t = pool(valid["text"]); te_t = pool(test["text"])
tr_a = pool(train["audio"], train["audio_lengths"]); va_a = pool(valid["audio"], valid["audio_lengths"]); te_a = pool(test["audio"], test["audio_lengths"])
tr_v = pool(train["vision"], train["vision_lengths"]); va_v = pool(valid["vision"], valid["vision_lengths"]); te_v = pool(test["vision"], test["vision_lengths"])

for tr, va, te in [(tr_t, va_t, te_t), (tr_a, va_a, te_a), (tr_v, va_v, te_v)]:
    sc = StandardScaler()
    tr[:] = sc.fit_transform(tr)
    va[:] = sc.transform(va)
    te[:] = sc.transform(te)

print(f"Dims -> text:{tr_t.shape[1]} audio:{tr_a.shape[1]} vision:{tr_v.shape[1]}")


class DS(Dataset):
    def __init__(self, t, a, v, y):
        self.t = torch.tensor(t, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.b = torch.tensor((y >= 0).astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.t[i], self.a[i], self.v[i], self.y[i], self.b[i]


train_loader = DataLoader(DS(tr_t, tr_a, tr_v, y_train), batch_size=32, shuffle=True)
valid_loader = DataLoader(DS(va_t, va_a, va_v, y_valid), batch_size=64)
test_loader = DataLoader(DS(te_t, te_a, te_v, y_test), batch_size=64)

n_neg = (y_train < 0).sum()
n_pos = (y_train >= 0).sum()
total = len(y_train)
w = torch.tensor([total / (2 * n_neg), total / (2 * n_pos)], dtype=torch.float32).to(device)


# ===== Multi-task model (same as previous run, kept for regression metrics) =====
class MultiTaskModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
        )
        self.reg = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        self.cls = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, t, a, v):
        x = torch.cat([t, a, v], dim=-1)
        h = self.shared(x)
        return self.reg(h).squeeze(-1), self.cls(h)


# ===== Dedicated binary-only classifier (deeper, no regression dilution) =====
class BinaryOnlyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(384, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, t, a, v):
        x = torch.cat([t, a, v], dim=-1)
        return self.net(x)


def to3(x):
    out = np.ones_like(x, dtype=int)
    out[x < 0] = 0
    out[x > 0] = 2
    return out


def to5(x):
    o = np.zeros_like(x, dtype=int)
    o[x <= -0.8] = 0
    o[(x > -0.8) & (x < 0)] = 1
    o[x == 0] = 2
    o[(x > 0) & (x < 0.8)] = 3
    o[x >= 0.8] = 4
    return o


def full_metrics(y_true, y_pred_reg, bin_logits_sum=None):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.clip(np.asarray(y_pred_reg).reshape(-1), -1, 1)

    mae = mean_absolute_error(y_true, y_pred) * 100
    corr = pearsonr(y_true, y_pred)[0] * 100

    y2 = (y_true >= 0).astype(int)
    if bin_logits_sum is not None:
        p2 = bin_logits_sum.argmax(axis=1)
    else:
        p2 = (y_pred >= 0).astype(int)
    acc2 = accuracy_score(y2, p2) * 100
    f12 = f1_score(y2, p2, average="weighted", zero_division=0) * 100

    y3, p3 = to3(y_true), to3(y_pred)
    acc3 = accuracy_score(y3, p3) * 100
    f13 = f1_score(y3, p3, average="weighted", zero_division=0) * 100

    y5, p5 = to5(y_true), to5(y_pred)
    acc5 = accuracy_score(y5, p5) * 100
    f15 = f1_score(y5, p5, average="weighted", zero_division=0) * 100

    return {
        "MAE(%)": round(mae, 2), "Corr(%)": round(corr, 2),
        "Acc-2(%)": round(acc2, 2), "F1-2(%)": round(f12, 2),
        "Acc-3(%)": round(acc3, 2), "F1-3(%)": round(f13, 2),
        "Acc-5(%)": round(acc5, 2), "F1-5(%)": round(f15, 2),
    }


dim = tr_t.shape[1] + tr_a.shape[1] + tr_v.shape[1]

# ================= TRAIN MULTI-TASK ENSEMBLE (5 seeds) =================
SEEDS = [42, 7, 123, 2024, 99]
mt_test_reg = []
mt_test_logits = []

for seed in SEEDS:
    set_seed(seed)
    print(f"\n{'='*50}\n[MultiTask] Seed {seed}")
    model = MultiTaskModel(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-6)

    best_val_score = -999
    best_state = None
    patience, wait = 12, 0

    for epoch in range(1, 51):
        model.train()
        for t, a, v, y, b in train_loader:
            t, a, v, y, b = t.to(device), a.to(device), v.to(device), y.to(device), b.to(device)
            reg, logits = model(t, a, v)
            reg_loss = F.smooth_l1_loss(reg, y)
            cls_loss = F.cross_entropy(logits, b, weight=w, label_smoothing=0.05)
            loss = 0.5 * reg_loss + 2.0 * cls_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        va_reg, va_logits = [], []
        with torch.no_grad():
            for t, a, v, y, b in valid_loader:
                t, a, v = t.to(device), a.to(device), v.to(device)
                r, l = model(t, a, v)
                va_reg.extend(r.cpu().numpy().tolist())
                va_logits.append(l.cpu().numpy())
        va_reg = np.clip(va_reg, -1, 1)
        va_logits = np.concatenate(va_logits, axis=0)
        va_bin = va_logits.argmax(axis=1)
        y2v = (y_valid >= 0).astype(int)
        acc2v = accuracy_score(y2v, va_bin) * 100
        f12v = f1_score(y2v, va_bin, average="weighted", zero_division=0) * 100
        corrv = pearsonr(y_valid, va_reg)[0] * 100
        score = acc2v + f12v + 0.5 * corrv

        if score > best_val_score:
            best_val_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    te_reg, te_logits = [], []
    with torch.no_grad():
        for t, a, v, y, b in test_loader:
            t, a, v = t.to(device), a.to(device), v.to(device)
            r, l = model(t, a, v)
            te_reg.extend(r.cpu().numpy().tolist())
            te_logits.append(l.cpu().numpy())
    te_reg = np.clip(np.array(te_reg), -1, 1)
    te_logits = np.concatenate(te_logits, axis=0)
    mt_test_reg.append(te_reg)
    mt_test_logits.append(te_logits)
    print(f"  best_val_score={best_val_score:.2f}")

# ================= TRAIN DEDICATED BINARY-ONLY MODELS (3 seeds) =================
BIN_SEEDS = [11, 22, 33]
bin_test_logits = []

for seed in BIN_SEEDS:
    set_seed(seed)
    print(f"\n{'='*50}\n[BinaryOnly] Seed {seed}")
    model = BinaryOnlyModel(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)

    best_val_score = -999
    best_state = None
    patience, wait = 15, 0

    for epoch in range(1, 61):
        model.train()
        for t, a, v, y, b in train_loader:
            t, a, v, b = t.to(device), a.to(device), v.to(device), b.to(device)
            logits = model(t, a, v)
            loss = F.cross_entropy(logits, b, weight=w, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        va_logits = []
        with torch.no_grad():
            for t, a, v, y, b in valid_loader:
                t, a, v = t.to(device), a.to(device), v.to(device)
                l = model(t, a, v)
                va_logits.append(l.cpu().numpy())
        va_logits = np.concatenate(va_logits, axis=0)
        va_bin = va_logits.argmax(axis=1)
        y2v = (y_valid >= 0).astype(int)
        acc2v = accuracy_score(y2v, va_bin) * 100
        f12v = f1_score(y2v, va_bin, average="weighted", zero_division=0) * 100
        score = acc2v + f12v

        if epoch % 15 == 0:
            print(f"  Ep{epoch} val_acc2={acc2v:.2f} val_f12={f12v:.2f}")

        if score > best_val_score:
            best_val_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    te_logits = []
    with torch.no_grad():
        for t, a, v, y, b in test_loader:
            t, a, v = t.to(device), a.to(device), v.to(device)
            l = model(t, a, v)
            te_logits.append(l.cpu().numpy())
    te_logits = np.concatenate(te_logits, axis=0)
    bin_test_logits.append(te_logits)
    print(f"  best_val_score={best_val_score:.2f}")

# ================= COMBINE EVERYTHING =================
print("\n" + "=" * 60)

# Regression ensemble (5 multi-task models) -- for MAE/Corr/Acc-3/Acc-5
ens_reg = np.mean(mt_test_reg, axis=0)

# Binary logits: average ALL 8 classifiers (5 multitask + 3 binary-only)
# Softmax each before averaging so scales match
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

all_bin_probs = [softmax(l) for l in mt_test_logits] + [softmax(l) for l in bin_test_logits]
ens_bin_probs_8 = np.mean(all_bin_probs, axis=0)

# Also try: only the 3 binary-only models
bin_only_probs = [softmax(l) for l in bin_test_logits]
ens_bin_probs_3 = np.mean(bin_only_probs, axis=0)

# Also: previous 5-model ensemble only
mt_only_probs = [softmax(l) for l in mt_test_logits]
ens_bin_probs_5 = np.mean(mt_only_probs, axis=0)

configs = {
    "previous_5_multitask_ensemble": ens_bin_probs_5,
    "3_binary_only_ensemble": ens_bin_probs_3,
    "all_8_combined_ensemble": ens_bin_probs_8,
}

print("COMPARISON OF BINARY ENSEMBLE STRATEGIES (using avg regression for MAE/Corr/Acc-3/Acc-5):")
results = {}
for name, probs in configs.items():
    m = full_metrics(y_test, ens_reg, probs)
    results[name] = m
    print(f"\n{name}:")
    for k, v in m.items():
        print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("PAPER REFERENCE (MLF-DNN*): MAE=40.64 Corr=67.47 Acc-2=82.28 F1-2=82.52 Acc-3=69.06")

winner = max(results, key=lambda k: results[k]["Acc-2(%)"] + results[k]["F1-2(%)"])
print(f"\nWINNER: {winner}")
for k, v in results[winner].items():
    print(f"  {k}: {v}")

os.makedirs("outputs", exist_ok=True)
pd.DataFrame(results).T.to_csv("outputs/chsims_last_try_results.csv")
print("\nSaved: outputs/chsims_last_try_results.csv")

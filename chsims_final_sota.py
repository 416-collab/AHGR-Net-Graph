import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD =================
def load(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

data = load("data/raw/CHSIMS/unaligned.pkl")
train, valid, test = data["train"], data["valid"], data["test"]

y_train = train["regression_labels"].reshape(-1).astype(np.float32)
y_valid = valid["regression_labels"].reshape(-1).astype(np.float32)
y_test  = test["regression_labels"].reshape(-1).astype(np.float32)

# ================= FEATURE =================
def pool(x):
    out = []
    for i in range(len(x)):
        seq = x[i]
        out.append(np.concatenate([seq.mean(0), seq.max(0), seq.std(0)]))
    return np.array(out).astype(np.float32)

t_tr, t_va, t_te = pool(train["text"]), pool(valid["text"]), pool(test["text"])
a_tr, a_va, a_te = pool(train["audio"]), pool(valid["audio"]), pool(test["audio"])
v_tr, v_va, v_te = pool(train["vision"]), pool(valid["vision"]), pool(test["vision"])

# normalize
for a,b,c in [(t_tr,t_va,t_te),(a_tr,a_va,a_te),(v_tr,v_va,v_te)]:
    sc = StandardScaler()
    a[:] = sc.fit_transform(a)
    b[:] = sc.transform(b)
    c[:] = sc.transform(c)

# ================= DATASET =================
class DS(Dataset):
    def __init__(self,t,a,v,y):
        self.t=torch.tensor(t,dtype=torch.float32)
        self.a=torch.tensor(a,dtype=torch.float32)
        self.v=torch.tensor(v,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
        self.b=torch.tensor((y>=0).astype(np.int64))

    def __len__(self): return len(self.y)

    def __getitem__(self,i):
        return self.t[i],self.a[i],self.v[i],self.y[i],self.b[i]

train_loader = DataLoader(DS(t_tr,a_tr,v_tr,y_train),32,True)
valid_loader = DataLoader(DS(t_va,a_va,v_va,y_valid),64)
test_loader  = DataLoader(DS(t_te,a_te,v_te,y_test),64)

# ================= MODEL (FIXED) =================
class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.enc_t = nn.Linear(dim//3, 128)
        self.enc_a = nn.Linear(dim//3, 128)
        self.enc_v = nn.Linear(dim//3, 128)

        self.shared = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.cls = nn.Linear(256, 2)
        self.reg = nn.Linear(256, 1)

    def forward(self, t, a, v):

        t = F.relu(self.enc_t(t))
        a = F.relu(self.enc_a(a))
        v = F.relu(self.enc_v(v))

        h = torch.cat([t,a,v], dim=-1)

        h = self.shared(h)

        reg = self.reg(h).squeeze(-1)
        logits = self.cls(h)

        return reg, logits

# ================= LOSS =================
reg_loss_fn = nn.SmoothL1Loss()
cls_loss_fn = nn.CrossEntropyLoss()

# ================= INIT =================
dim = t_tr.shape[1] + a_tr.shape[1] + v_tr.shape[1]
model = Model(dim).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

# ================= TRAIN =================
for epoch in range(1,41):
    model.train()

    for t,a,v,y,b in train_loader:
        t,a,v,y,b = t.to(device),a.to(device),v.to(device),y.to(device),b.to(device)

        reg, logits = model(t,a,v)

        loss = 0.5*reg_loss_fn(reg,y) + 1.5*cls_loss_fn(logits,b)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # ================= VALID =================
    model.eval()

    preds, bins, labels = [], [], []

    with torch.no_grad():
        for t,a,v,y,b in valid_loader:
            t,a,v = t.to(device),a.to(device),v.to(device)

            r,logits = model(t,a,v)

            preds.extend(r.cpu().numpy())
            bins.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    preds=np.array(preds)
    labels=np.array(labels)

    acc = accuracy_score((labels>=0),bins)*100
    f1  = f1_score((labels>=0),bins)*100
    corr = pearsonr(labels,preds)[0]*100

    print(f"Epoch {epoch} | Acc-2 {acc:.2f} | F1-2 {f1:.2f} | Corr {corr:.2f}")

# ================= TEST =================
model.eval()

preds,bins,labels=[],[],[]

with torch.no_grad():
    for t,a,v,y,b in test_loader:
        t,a,v = t.to(device),a.to(device),v.to(device)

        r,logits = model(t,a,v)

        preds.extend(r.cpu().numpy())
        bins.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())

preds=np.array(preds)
labels=np.array(labels)

print("\nFINAL RESULTS")
print("Acc-2:", accuracy_score((labels>=0),bins)*100)
print("F1-2 :", f1_score((labels>=0),bins)*100)
print("MAE  :", mean_absolute_error(labels,preds)*100)
print("Corr :", pearsonr(labels,preds)[0]*100)

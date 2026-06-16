import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD DATA =================
def load(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

data = load("data/raw/CHSIMS/unaligned.pkl")
train, valid, test = data["train"], data["valid"], data["test"]

y_train = train["regression_labels"].reshape(-1).astype(np.float32)
y_valid = valid["regression_labels"].reshape(-1).astype(np.float32)
y_test  = test["regression_labels"].reshape(-1).astype(np.float32)

# ================= FEATURES =================
def pool(x):
    out = []
    for i in range(len(x)):
        seq = x[i]
        out.append(np.concatenate([seq.mean(0), seq.max(0), seq.std(0)]))
    return np.array(out).astype(np.float32)

tr_t, va_t, te_t = pool(train["text"]), pool(valid["text"]), pool(test["text"])
tr_a, va_a, te_a = pool(train["audio"]), pool(valid["audio"]), pool(test["audio"])
tr_v, va_v, te_v = pool(train["vision"]), pool(valid["vision"]), pool(test["vision"])

# normalize
sc = StandardScaler()
tr_t = sc.fit_transform(tr_t); va_t = sc.transform(va_t); te_t = sc.transform(te_t)

sc = StandardScaler()
tr_a = sc.fit_transform(tr_a); va_a = sc.transform(va_a); te_a = sc.transform(te_a)

sc = StandardScaler()
tr_v = sc.fit_transform(tr_v); va_v = sc.transform(va_v); te_v = sc.transform(te_v)

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

train_loader = DataLoader(DS(tr_t,tr_a,tr_v,y_train),32,True)
valid_loader = DataLoader(DS(va_t,va_a,va_v,y_valid),64)
test_loader  = DataLoader(DS(te_t,te_a,te_v,y_test),64)

# ================= MODEL =================
class Model(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim,256),
            nn.GELU(),
            nn.Linear(256,256),
            nn.GELU()
        )
        self.cls = nn.Sequential(
            nn.Linear(256,128),
            nn.GELU(),
            nn.Linear(128,2)
        )
        self.reg = nn.Sequential(
            nn.Linear(256,128),
            nn.GELU(),
            nn.Linear(128,1)
        )

    def forward(self,t,a,v):
        x = torch.cat([t,a,v],dim=-1)
        h = self.shared(x)
        return self.reg(h).squeeze(-1), self.cls(h)

# ================= FOCAL LOSS =================
class FocalLoss(nn.Module):
    def __init__(self,gamma=2):
        super().__init__()
        self.gamma=gamma

    def forward(self,logits,target):
        ce=F.cross_entropy(logits,target,reduction='none')
        pt=torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

focal = FocalLoss()

# ================= LOSS =================
def boundary_loss(pred,y):
    pos = pred[y>=0]
    neg = pred[y<0]
    return torch.relu(0.15-(pos.mean()-neg.mean()))

# ================= TRAIN =================
model = Model(tr_t.shape[1]+tr_a.shape[1]+tr_v.shape[1]).to(device)
opt = torch.optim.AdamW(model.parameters(),5e-5)

for epoch in range(1,41):
    model.train()

    for t,a,v,y,b in train_loader:
        t,a,v,y,b=t.to(device),a.to(device),v.to(device),y.to(device),b.to(device)

        reg,logits = model(t,a,v)

        reg_loss = F.smooth_l1_loss(reg,y)
        cls_loss = focal(logits,b)
        cont_loss = boundary_loss(reg,y)

        loss = 0.3*reg_loss + 3.0*cls_loss + 1.0*cont_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    preds=[]; bins=[]; labels=[]

    with torch.no_grad():
        for t,a,v,y,b in valid_loader:
            t,a,v=t.to(device),a.to(device),v.to(device)
            r,logits=model(t,a,v)

            preds.extend(r.cpu().numpy())
            bins.extend(logits.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    preds=np.array(preds); labels=np.array(labels)

    acc=accuracy_score((labels>=0),bins)*100
    f1=f1_score((labels>=0),bins)*100
    corr=pearsonr(labels,preds)[0]*100

    print(f"Epoch {epoch} | Acc-2 {acc:.2f} | F1-2 {f1:.2f} | Corr {corr:.2f}")

# ================= TEST =================
model.eval()
preds=[]; bins=[]; labels=[]

with torch.no_grad():
    for t,a,v,y,b in test_loader:
        t,a,v=t.to(device),a.to(device),v.to(device)
        r,logits=model(t,a,v)

        preds.extend(r.cpu().numpy())
        bins.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())

preds=np.array(preds); labels=np.array(labels)

print("\nFINAL RESULTS")
print("Acc-2:", accuracy_score((labels>=0),bins)*100)
print("F1-2 :", f1_score((labels>=0),bins)*100)

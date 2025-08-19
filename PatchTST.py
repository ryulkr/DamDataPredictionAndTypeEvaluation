#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# PatchTST – 김천부항 6개 결측 Type 일괄 처리
#   · 학습 구간: 2023-01-05 10:00 ~ 2023-02-06 15:00  (33.1 d)
#   · 평가 구간: 2023-02-06 16:00 ~ 2023-02-07 15:00  (24  h)
#   · 추가 피처: dX, 24h mean, 24h std, 6h std
#   · PatchLen 24 / Stride 12 ‧ d_model 128 ‧ depth 2 ‧ head 4
#   · Loss = 0.5·MSE + 0.5·MAE ‧ Early-Stopping(patience 30)
# ────────────────────────────────────────────────────────────────
import os, math, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# ─────────── 공통 설정 ───────────
DATA_DIR   = "./예측할 데이터"
OUT_DIR    = "./타입별예측결과"
DAM_NAME   = "김천부항"                       # 파일명의 접두어
TYPES      = [1, 2, 3, 4, 5, 6]

START_TRAIN = datetime(2023, 1, 5, 10)
END_TRAIN   = datetime(2023, 2, 6, 15)
START_TEST  = datetime(2023, 2, 1, 0) 
END_TEST    = datetime(2023, 2, 7, 15)

INPUT_LEN = 96      # 과거 96시점(=4일) 입력
PATCH_LEN = 24
STRIDE    = 12
PRED_LEN  = 1
BATCH     = 32
EPOCHS    = 300
PATIENCE  = 30
LR        = 1e-3
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SEED      = 42; torch.manual_seed(SEED)

# ─────────── PatchTST 정의 (compact) ───────────
class PatchEmbed(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.proj = nn.Linear(in_ch * PATCH_LEN, 128)
        self.norm = nn.LayerNorm(128)
    def forward(self, x):
        B, T, C = x.shape
        patches = [x[:, i:i+PATCH_LEN].reshape(B, -1)
                   for i in range(0, T-PATCH_LEN+1, STRIDE)]
        return self.norm(self.proj(torch.stack(patches, 1)))

class PosEnc(nn.Module):
    def __init__(self, d, M=1000):
        super().__init__()
        pe = torch.zeros(M, d)
        pos = torch.arange(0, M).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class PatchTST(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.embed = PatchEmbed(in_ch)
        self.pos   = PosEnc(128)
        enc_layer  = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head    = nn.Linear(128, PRED_LEN)
    def forward(self, x):
        z = self.encoder(self.pos(self.embed(x))).mean(1)
        return self.head(z)

# ─────────── 데이터셋 클래스 ───────────
class SeqDS(Dataset):
    def __init__(self, arr, ts, mode, tgt_idx):
        X, Y, T = [], [], []
        for i in range(INPUT_LEN, len(arr)):
            cur = ts[i]
            if mode == "train" and cur <= END_TRAIN:
                X.append(arr[i-INPUT_LEN:i]); Y.append(arr[i, tgt_idx])
            elif mode == "test" and START_TEST <= cur <= END_TEST:
                X.append(arr[i-INPUT_LEN:i]); Y.append(arr[i, tgt_idx]); T.append(str(cur))
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.T = T;  self.mode = mode
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx]) if self.mode == "train" \
               else (self.X[idx], self.Y[idx], self.T[idx])

# ─────────── 메인 루프 ───────────
os.makedirs(OUT_DIR, exist_ok=True)

for tp in TYPES:
    f_in  = f"{DATA_DIR}/{DAM_NAME}_GPS변위계_type{tp}.csv"
    f_out = f"{OUT_DIR}/{DAM_NAME}_type{tp}_PatchTST.csv"
    print(f"\n[Type {tp}]  loading → {f_in}")

    # ---------- 1) 파일 로드 & 피처 ----------
    df = pd.read_csv(f_in, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()

    base = ["AA0001_X", "AA0001_Y", "AA0001_Z", "AA0001_V"]
    df["dX"]       = df["AA0001_X"].diff().fillna(0)
    df["X_roll24"] = df["AA0001_X"].rolling(24).mean().fillna(method="bfill")
    df["X_std24"]  = df["AA0001_X"].rolling(24).std().fillna(0)
    df["X_std6"]   = df["AA0001_X"].rolling(6 ).std().fillna(0)

    cols  = base + ["dX", "X_roll24", "X_std24", "X_std6"]
    data  = df[cols].values
    times = df.index.to_list()
    tgt_i = cols.index("AA0001_X")

    scaler = MinMaxScaler(); data_s = scaler.fit_transform(data)

    # ---------- 2) 데이터셋 만들기 ----------
    train_ds = SeqDS(data_s, times, "train", tgt_i)
    test_ds  = SeqDS(data_s, times, "test",  tgt_i)
    tr_loader = DataLoader(train_ds, BATCH, shuffle=True)
    te_loader = DataLoader(test_ds,  BATCH)

    # ---------- 3) 모델 ----------
    model = PatchTST(len(cols)).to(DEVICE)
    mse, mae = nn.MSELoss(), nn.L1Loss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best, bad = float("inf"), 0

    # ---------- 4) 학습 ----------
    for ep in range(1, EPOCHS + 1):
        model.train(); losses = []
        for X, y in tr_loader:
            X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            opt.zero_grad()
            out = model(X)
            loss = 0.5 * mse(out, y) + 0.5 * mae(out, y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        cur = np.mean(losses)
        if (ep % 10 == 0) or (ep == 1):
            print(f"  Epoch {ep:3d}  Loss {cur:.6f}")
        if cur < best - 1e-4: best, bad = cur, 0
        else: bad += 1
        if bad >= PATIENCE:
            print("  → Early‐Stopping"); break

    # ---------- 5) 예측 ----------
    model.eval(); pred, true, ts_list = [], [], []
    with torch.no_grad():
        for X, y, t in te_loader:
            out = model(X.to(DEVICE)).cpu().numpy().flatten()
            pred.extend(out); true.extend(y.numpy()); ts_list.extend(t)

    dummy = np.zeros((len(pred), scaler.n_features_in_))
    dummy[:, tgt_i] = pred
    pred_inv = scaler.inverse_transform(dummy)[:, tgt_i]
    dummy[:, tgt_i] = true
    true_inv = scaler.inverse_transform(dummy)[:, tgt_i]

    pd.DataFrame({"Time": ts_list,
                  "Actual": true_inv,
                  "Prediction": pred_inv})\
      .to_csv(f_out, index=False)
    print(f"  ✅  saved  {f_out}")

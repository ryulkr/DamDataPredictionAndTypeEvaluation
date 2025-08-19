import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error
import os

# -------------------------------
# 설정
# -------------------------------
target_dam = '충주'
data_path = f'./예측할 데이터/{target_dam}_GPS변위계(예측).csv'
save_path = f'./예측결과/{target_dam}_LSTM_Result.csv'

# -------------------------------
# 데이터 불러오기 및 분할
# -------------------------------
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
X = df[['AA0001_Y', 'AA0001_Z', 'AA0001_V']]
y = df['AA0001_X']

n = len(X)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
test_index = df.index[val_end:]

# -------------------------------
# Dataset 정의
# -------------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y, window_size):
        self.X, self.y = [], []
        for i in range(len(X) - window_size):
            self.X.append(X[i:i+window_size])
            self.y.append(y[i+window_size])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------
# LSTM 모델 정의
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze()

# -------------------------------
# 평가 함수 (BayesOpt용)
# -------------------------------
def train_lstm_model(hidden_size, num_layers, learning_rate, window_size, epochs, batch_size):
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    window_size = int(window_size)
    epochs = int(epochs)
    batch_size = int(batch_size)

    train_dataset = SequenceDataset(X_train.values, y_train.values, window_size)
    val_dataset = SequenceDataset(X_val.values, y_val.values, window_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=X.shape[1], hidden_size=hidden_size, num_layers=num_layers)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()

    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            preds.extend(pred.numpy())
            targets.extend(yb.numpy())

    return -mean_absolute_error(targets, preds)

# -------------------------------
# Bayesian Optimization
# -------------------------------
pbounds = {
    'hidden_size': (16, 200),
    'num_layers': (1, 5),
    'learning_rate': (0.00001, 0.01),
    'window_size': (1, 50),
    'epochs': (10, 500),
    'batch_size': (16, 512)
}

optimizer = BayesianOptimization(f=train_lstm_model, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=15)

# -------------------------------
# 최적 파라미터로 테스트 예측
# -------------------------------
params = optimizer.max['params']
h = int(params['hidden_size'])
l = int(params['num_layers'])
lr = params['learning_rate']
w = int(params['window_size'])
ep = int(params['epochs'])
bs = int(params['batch_size'])

train_dataset = SequenceDataset(X_train.values, y_train.values, w)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)

model = LSTMModel(input_size=X.shape[1], hidden_size=h, num_layers=l)
optimizer_torch = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.L1Loss()

model.train()
for _ in range(ep):
    for xb, yb in train_loader:
        optimizer_torch.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer_torch.step()

# 테스트 예측
test_dataset = SequenceDataset(X_test.values, y_test.values, w)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        pred = model(xb)
        preds.append(pred.item())

# 결과 저장
aligned_index = test_index[w:]
result_df = pd.DataFrame({
    'Time': aligned_index,
    'Actual': y_test.values[w:],
    'Prediction': preds
})
os.makedirs(os.path.dirname(save_path), exist_ok=True)
result_df.to_csv(save_path, index=False)

# 파라미터 출력
print("\n✅ 최적 하이퍼파라미터:")
for k, v in params.items():
    print(f"{k:<14}: {round(v) if k != 'learning_rate' else round(v, 6)}")
print(f"\n📁 예측 결과 저장 완료: {save_path}")

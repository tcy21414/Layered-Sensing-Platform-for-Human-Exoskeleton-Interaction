import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import argparse
from tcn import TemporalConvNet  # Assuming tcn.py is in the same directory
from utils import setup_lr_scheduler  # Assuming utils.py is in the same directory

parser = argparse.ArgumentParser(description='troque regression from IMU & EMG')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--wandb', action='store_true',default=True, help='disable wandb')
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default='50')

args = parser.parse_args()

class MomentDataset(Dataset):
    def __init__(self, df, imu_mean=None, imu_std=None, emg_mean=None, emg_std=None):
        self.df = df
        self.ids = df['id'].unique()
        imu_cols = [f'IMU_{i+1}' for i in range(18)]
        imu_data = df[imu_cols].values  # shape: [N, 18]
        if imu_mean is None or imu_std is None:
            self.imu_mean = imu_data.mean(axis=0, keepdims=True).T   # shape [18]
            self.imu_std = imu_data.std(axis=0, keepdims=True).T    # shape [18]
        else:
            self.imu_mean = imu_mean
            self.imu_std = imu_std

        emg_cols = [f'EMG_{i+1}' for i in range(3)]
        emg_data = df[emg_cols].values  # shape: [N, 3]
        if emg_mean is None or emg_std is None:
            self.emg_mean = emg_data.mean(axis=0, keepdims=True).T
            self.emg_std = emg_data.std(axis=0, keepdims=True).T # Avoid division by zero
        else:
            self.emg_mean = emg_mean
            self.emg_std = emg_std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        sample_df = self.df[self.df['id'] == sample_id]

        imu = sample_df[[f'IMU_{i+1}' for i in range(18)]].values.T.astype(np.float32)
        imu = (imu - self.imu_mean) / (self.imu_std)
        emg = sample_df[[f'EMG_{i+1}' for i in range(3)]].values.T.astype(np.float32)
        emg = (emg - self.emg_mean) / (self.emg_std)  # Avoid division by zero
        label = sample_df['ankle_moment'].values[-1].astype(np.float32)
        return (
            torch.tensor(imu, dtype=torch.float32),       # shape: [18, 200]
            torch.tensor(emg, dtype=torch.float32),       # shape: [3, 200]
            torch.tensor(label)      # shape: scalar
        )

class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(ch, ch // r, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(ch // r, ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class EMGResNetBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            SEBlock(ch)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class EMGEncoder(nn.Module):
    def __init__(self, in_ch=3, hidden_ch=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(),
            EMGResNetBlock(hidden_ch),
            EMGResNetBlock(hidden_ch),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):  # [B, C, L]
        out = self.encoder(x)
        return out.squeeze(-1)


class MomentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.imu_encoder = TemporalConvNet(18, [64,64,64,64,64], kernel_size=4, dropout=0.2)
        self.emg_encoder = EMGEncoder()
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.init_weights()
        self.imu_avg = nn.AdaptiveAvgPool1d(1)

    def init_weights(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, imu, emg):
        imu_feat = self.imu_encoder(imu)
        last_output = imu_feat[:, :, -1] 
        emg_feat = self.emg_encoder(emg)
        fused = torch.cat([last_output, emg_feat], dim=1)
        return self.head(fused).squeeze(1)

df = pd.read_csv('Estimation_dataset_new.csv')
df_loso = pd.read_csv('Estimation_LOSO_new.csv')

dataset = MomentDataset(df)
loso_dataset = MomentDataset(df_loso,dataset.imu_mean,dataset.imu_std,dataset.emg_mean,dataset.emg_std)

print("Total train samples:", len(dataset))
print("Total test samples:", len(loso_dataset))  
train_loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
val_loader = DataLoader(loso_dataset, batch_size=args.bs, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MomentModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler_cfg = {
    "type": "CosineAnnealingLRWithWarmup",
    "params": {
        "warmup_lr": 2e-5,
        "warmup_epoch": 5,
        "T_max": args.n_epochs #epoches
    }
}
scheduler = setup_lr_scheduler(optimizer, scheduler_cfg)
criterion = nn.MSELoss()

with open("torque_result.txt", "w") as f:
        f.write("troque regression from IMU & EMG \n")

for epoch in range(args.n_epochs):
    model.train()
    total_loss = 0
    for imu, emg, label in train_loader:
        imu, emg, label = imu.to(device), emg.to(device), label.to(device)
        pred = model(imu, emg)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    model.eval()
    val_loss = 0
    preds = []
    targets = []
    with torch.no_grad():
        for imu, emg, label in val_loader:
            imu, emg, label = imu.to(device), emg.to(device), label.to(device)
            pred = model(imu, emg)
            val_loss += criterion(pred, label).item()
            preds.append(pred.cpu().numpy())
            targets.append(label.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    print(f"Epoch {epoch+1} | lr: {optimizer.param_groups[0]['lr']} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | RMSE: {rmse:.4f}")
    with open("torque_result.txt", "a") as f:
        f.write(f"Epoch {epoch+1} | lr: {optimizer.param_groups[0]['lr']} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | RMSE: {rmse:.4f}\n")

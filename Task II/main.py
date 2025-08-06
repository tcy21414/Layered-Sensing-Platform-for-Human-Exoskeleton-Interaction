import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# ====================== Data Loader ======================

class MetabDataset(Dataset):
    """
    Dataset class for loading metabolic samples from CSV files.
    Assumes each row is a flattened [IMU, EMG, label] sample.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.samples = []

        n_features = df.shape[1] - 1  # Last column is the label
        total_points = n_features // 24  # 18 IMU + 6 EMG channels
        assert total_points == 9000, f"Expected 9000 timepoints, got {total_points}"

        imu_len = 18 * total_points
        emg_len = 6 * total_points

        for i in range(len(df)):
            row = df.iloc[i]
            all_data = row[:-1].values.astype(np.float32)
            label = int(row[-1])

            imu = all_data[:imu_len].reshape(18, total_points)
            emg = all_data[imu_len:].reshape(6, total_points)
            self.samples.append((imu, emg, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imu, emg, label = self.samples[idx]
        return torch.tensor(imu), torch.tensor(emg), torch.tensor(label)

# ====================== TCN Encoder for IMU ======================

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.downsample(x))

class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder for IMU signal.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(in_channels, 64, dilation=1),
            TCNBlock(64, 64, dilation=2),
            TCNBlock(64, 64, dilation=4),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ====================== SE-ResNet Encoder for EMG ======================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + residual)

class SEResNet1D(nn.Module):
    """
    Squeeze-and-Excitation ResNet encoder for EMG signal.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.initial = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.relu(self.bn(self.initial(x)))
        x = self.blocks(x)
        return x.squeeze(-1)

# ====================== Classifier ======================

class MetabClassifier(nn.Module):
    """
    Metabolic classifier combining IMU and EMG encoders.
    """
    def __init__(self):
        super().__init__()
        self.imu_encoder = TCNEncoder(in_channels=18)
        self.emg_encoder = SEResNet1D(in_channels=6)
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3-class output
        )

    def forward(self, imu, emg):
        imu_feat = self.imu_encoder(imu)
        emg_feat = self.emg_encoder(emg)
        feat = torch.cat([imu_feat, emg_feat], dim=1)
        return self.classifier(feat)

# ====================== Training and Evaluation ======================

def train_and_test(model, train_loader, val_loader, test_loader):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)

    # Training loop
    for epoch in range(60):
        model.train()
        total_loss = 0
        for imu, emg, label in train_loader:
            imu, emg, label = imu.to(device), emg.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(imu, emg)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation loop
    def evaluate(loader, name=""):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imu, emg, label in loader:
                imu, emg, label = imu.to(device), emg.to(device), label.to(device)
                pred = model(imu, emg).argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        acc = correct / total * 100
        print(f"{name} Accuracy: {acc:.2f}%")

    evaluate(val_loader, "Validation")
    evaluate(test_loader, "LOSO Test")

# ====================== Entry Point ======================

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using MPS acceleration")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available, falling back to CPU")

# Load training dataset
dataset = MetabDataset('.../Metab_Dataset.csv')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# Load LOSO test dataset
test_set = MetabDataset('.../Metab_LOSO.csv')
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Instantiate and train model
model = MetabClassifier()
train_and_test(model, train_loader, val_loader, test_loader)

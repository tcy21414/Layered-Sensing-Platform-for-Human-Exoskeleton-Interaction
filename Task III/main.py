import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ======== Device Selection ========
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ======== Custom Dataset ========
class DangerDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.ids = df['id'].unique()
        self.samples = []
        for sid in self.ids:
            df_s = df[df['id'] == sid]
            x = df_s[['strain1', 'strain2']].values.T.astype(np.float32)
            y = df_s['label'].values[0]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y).long()

# ======== SE-ResNet Definition ========
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + x)

class SEResNet1D(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.initial = nn.Conv1d(in_channels, 64, 7, padding=3)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.relu(self.bn(self.initial(x)))
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

# ======== Focal Loss Definition ========
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ======== Training & Evaluation (with Recall & Focal Loss) ========
def train_and_evaluate(model, train_loader, val_loader, test_loader):
    model.to(device)
    criterion = FocalLoss(alpha=0.7, gamma=2)  # Emphasize class 1 (danger)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    for epoch in range(60):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    def evaluate(loader, name):
        model.eval()
        correct = 0
        total = 0
        true_positives = 0
        actual_positives = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                true_positives += ((pred == 1) & (y == 1)).sum().item()
                actual_positives += (y == 1).sum().item()

        acc = correct / total * 100
        recall = true_positives / actual_positives * 100 if actual_positives > 0 else 0.0
        print(f"{name} Accuracy: {acc:.2f}% | Recall (Danger): {recall:.2f}%")

    evaluate(val_loader, "Validation")
    evaluate(test_loader, "LOSO Zero-shot")

# ======== Data Loading & Training Start ========
dataset = DangerDataset('.../Danger_Dataset.csv')
loso = DangerDataset('.../Danger_LOSO.csv')

train_len = int(0.8 * len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)
test_loader = DataLoader(loso, batch_size=64)

model = SEResNet1D()
train_and_evaluate(model, train_loader, val_loader, test_loader)

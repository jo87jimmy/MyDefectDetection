# train_depreciation_mlp.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

def train_mlp_from_csv(csv_path="depreciation_records.csv", output_path="depreciation_mlp.pth"):
    df = pd.read_csv(csv_path)

    # 特徵與標籤
    X = torch.tensor(df[["defect_index", "avg_depth", "max_depth", "total_area"]].values, dtype=torch.float32)
    y = torch.tensor(df["grade"].map({"A - 正常": 0, "B - 觀察中": 1, "C - 建議維修": 2}).values, dtype=torch.long)

    # DataLoader
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 模型與訓練元件
    model = DepreciationMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    torch.save(model, output_path)
    print(f"✅ 模型已訓練並儲存至 {output_path}")


# 模擬資料（你可以改成讀取 CSV）
def generate_sample_data(num_samples=300):
    np.random.seed(42)
    defect_index = np.random.rand(num_samples)
    avg_depth = np.random.rand(num_samples)
    max_depth = avg_depth + np.random.rand(num_samples) * 0.5
    total_area = np.random.rand(num_samples) * 100

    # 簡單規則生成 label（可替換為真實標註）
    label = []
    for i in range(num_samples):
        if defect_index[i] < 0.3 and max_depth[i] < 0.5:
            label.append(0)  # A - 正常
        elif defect_index[i] < 0.6:
            label.append(1)  # B - 觀察中
        else:
            label.append(2)  # C - 建議維修

    df = pd.DataFrame({
        "defect_index": defect_index,
        "avg_depth": avg_depth,
        "max_depth": max_depth,
        "total_area": total_area,
        "label": label
    })
    return df

# 模型定義
class DepreciationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.net(x)

# 主流程
def main():
    # 產生資料
    data = generate_sample_data()

    # 資料轉 tensor
    X = torch.tensor(data[["defect_index", "avg_depth", "max_depth", "total_area"]].values, dtype=torch.float32)
    y = torch.tensor(data["label"].values, dtype=torch.long)

    # 建立 DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化模型與訓練元件
    model = DepreciationMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 訓練迴圈
    for epoch in range(50):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f}")

    # 儲存模型
    torch.save(model, "depreciation_mlp.pth")
    print("✅ 模型已儲存為 depreciation_mlp.pth")

if __name__ == "__main__":
    main()

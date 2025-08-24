import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import os

def train_mlp_from_csv(csv_path="depreciation_records.csv", output_path="depreciation_mlp.pth"):
    """
    從 CSV 資料訓練或微調折舊分析用的 MLP 模型，並儲存為 .pth 檔案。
    若模型已存在，則載入並繼續訓練；否則新建模型。
    """

    # 📥 讀取資料
    df = pd.read_csv(csv_path)
    X = torch.tensor(df[["defect_index", "avg_depth", "max_depth", "total_area"]].values, dtype=torch.float32)
    y = torch.tensor(df["grade"].map({"A - normal": 0, "B - Under_observation": 1, "C - Recommended_repair": 2}).values, dtype=torch.long)

    # 📦 建立資料集與 DataLoader
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 🧠 模型建立或載入
    model = DepreciationMLP()
    if os.path.exists(output_path):
        print(f"📂 偵測到已存在模型 {output_path}，將載入並繼續訓練")
        model.load_state_dict(torch.load(output_path,weights_only=True))  # 載入權重

    # ⚙️ 訓練元件設定
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 🔁 模型訓練迴圈
    for epoch in range(50):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    # 💾 儲存模型
    # torch.save(model, output_path)
    torch.save(model.state_dict(), output_path)  # 只儲存權重
    print(f"✅ 模型已訓練並儲存至 {output_path}")

def compute_depreciation_metrics(defects):
    """
    📌 用途：根據缺陷清單計算折舊分析的核心指標(defect_index)。
    每個缺陷的影響力以「面積 × 深度」表示，並統計整體缺陷數量、平均深度、最大深度與總面積。
    🔢 回傳內容：
    - defect_index：折舊指數(面積 × 深度 的加總)
    - defect_count：缺陷數量
    - avg_depth：平均深度
    - max_depth：最大深度
    - total_area：所有缺陷的總面積
    """
    if not defects:
        return {
            "defect_index": 0,
            "defect_count": 0,
            "avg_depth": 0,
            "max_depth": 0,
            "total_area": 0
        }

    defect_index = sum([d['area'] * d['depth'] for d in defects])
    avg_depth = np.mean([d['depth'] for d in defects])
    max_depth = np.max([d['depth'] for d in defects])
    total_area = sum([d['area'] for d in defects])
    return {
        "defect_index": defect_index,
        "defect_count": len(defects),
        "avg_depth": avg_depth,
        "max_depth": max_depth,
        "total_area": total_area
    }

def classify_depreciation(defect_index):
    """
    📌 用途：根據折舊指數(defect_index)進行分級判斷，協助使用者快速了解物件目前的健康狀態。
    📊 分級邏輯：
    - A：normal(defect_index < 50)
    - B：Under_observation(50 ≤ defect_index < 150)
    - C：Recommended_repair(defect_index ≥ 150)
    🔁 回傳：
    對應的折舊等級字串(含建議)
    """
    """基於你數據的25%和75%百分位數，能確保：
        約25%的數據為A級(normal)
        約50%的數據為B級(Under_observation)
        約25%的數據為C級(Recommended_repair)
        其他考慮的方案：
        標準差法：A<2971, B<6225, C≥6225
        保守法：A<4631, B<6741, C≥6741 """
    if defect_index < 3876:
        return "A - normal"
    elif defect_index < 5554:
        return "B - Under_observation"
    else:
        return "C - Recommended_repair"

def generate_depreciation_record(defects,mlp_model=None):
    """
    📌 用途：整合折舊分析流程，生成一筆完整的紀錄。
    包含分析時間、折舊等級、各項指標與原始缺陷清單，方便儲存、追蹤與可視化。
    🧩 組成：
    - timestamp：分析時間(格式：YYYY-MM-DD HH:MM)
    - grade：折舊等級(由 classify_depreciation 判斷)
    - defect_index / defect_count / avg_depth / max_depth / total_area：由 compute_depreciation_metrics 計算
    - defects：原始缺陷清單(含面積、深度、位置等)
    🔁 回傳：
    一個 dict 結構的折舊分析紀錄
    """
    metrics = compute_depreciation_metrics(defects)
    if mlp_model: #使用 MLP 模型根據缺陷指標預測折舊等級。
        grade, confidence = classify_depreciation_mlp(metrics, mlp_model)
    else:       #使用簡單的閾值分類
        grade = classify_depreciation(metrics["defect_index"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return {
        "timestamp": timestamp,
        "grade": grade,
        "confidence": confidence if mlp_model else "N/A",
        **metrics,
        "defects": defects
    }

import torch.nn as nn
class DepreciationMLP(nn.Module):
    """ 折舊分析用的多層感知器(MLP)模型
    Linear 1 -
    接收折舊指標: defect_index、avg_depth、max_depth、total_area 四個輸入特徵，
    預測折舊等級(A、B、C 三類)。
    輸入維度:4(defect_index、avg_depth、max_depth、total_area)
    隱藏層維度:16
    激活函數:ReLU 非線性轉換
    Linear 2 -
    輸出維度:3(對應三個折舊等級類別 A/B/C)
    🔁 前向傳播流程:
    輸入 → 線性層1 → ReLU → 線性層2 → 輸出 logits
    """
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()# 初始化 nn.Module 父類別
        # 🧠 定義 MLP 架構：輸入層 → 隱藏層 → 輸出層
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),# 線性層：輸入維度 → 隱藏層維度
            nn.ReLU(),# 啟用函數：ReLU 非線性轉換
            nn.Linear(hidden_dim, output_dim)# 線性層：隱藏層 → 輸出類別數
        )
    def forward(self, x):
        return self.model(x)# 前向傳播：將輸入 x 傳入模型並回傳輸出

def classify_depreciation_mlp(metrics, mlp_model):
    """
    使用 MLP 模型根據缺陷指標預測折舊等級，並回傳信心分數。
    🔢 輸入：metrics dict(包含 defect_index、avg_depth、max_depth、total_area)
    🔁 回傳：tuple → (折舊等級字串, 信心分數)
    """
    # 🧮 將指標轉為張量並加上 batch 維度
    input_tensor = torch.tensor([
        metrics["defect_index"],
        metrics["avg_depth"],
        metrics["max_depth"],
        metrics["total_area"]
    ], dtype=torch.float32).unsqueeze(0)

    # 🚫 推論模式(停用梯度)
    with torch.no_grad():
        logits = mlp_model(input_tensor)  # 前向傳播,logits 是 MLP 模型的原始輸出(未經標準化)，通常是每個類別的分數
        probs = torch.softmax(logits, dim=1)  # 計算 softmax 機率分布，softmax 將 logits 轉換為機率分布，使所有類別的機率加總為 1。
        #EX:logits = [1.2, 3.5, 0.8] (所有類別的機率加總為1時)-> probs = [0.12, 0.82, 0.06]
        pred = torch.argmax(probs, dim=1).item()  # 取得預測類別索引，pred 是最大機率的類別索引(即模型預測的分類)。
        confidence = probs[0, pred].item()  # 取得該類別的信心分數，是該類別的機率值，代表模型對這個預測的信心。

    # 📤 回傳等級與信心分數
    label = ["A - normal", "B - Under_observation", "C - Recommended_repair"]
    # 信心分數(confidence)的用途：
    # 報表呈現：讓使用者知道預測是否可靠
    # 警示機制：若信心低於某門檻(例如 0.6)，可標示為「不確定」
    # 模型評估：可用於 ROC 曲線、Precision-Recall 分析
    # 決策支援：高信心可自動通過，低信心可轉人工複核
    return label[pred], confidence

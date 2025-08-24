import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
def train_mlp_from_csv(csv_path="depreciation_records.csv", output_path="depreciation_mlp.pth"):
    """
    負責從 CSV 資料訓練折舊分析用的 MLP 模型，並儲存為 .pth 檔案
    訓練一個簡單的 MLP 模型來預測折舊等級。整體流程包含：
    資料讀取與轉換，模型建立與訓練，模型儲存與提示
    用於持續更新模型的場景，例如每新增 N 筆紀錄就 retrain。
    todo:
    模型評估(accuracy、confusion matrix)
    early stopping 或 learning rate scheduler，多層架構或 dropout   
    """
    df = pd.read_csv(csv_path)  # 從 CSV 檔案讀取折舊紀錄資料
    # 🔢 特徵與標籤準備
    X = torch.tensor(df[["defect_index", "avg_depth", "max_depth", "total_area"]].values, dtype=torch.float32)  # 取出四個指標作為輸入特徵
    y = torch.tensor(df["grade"].map({"A - 正常": 0, "B - 觀察中": 1, "C - 建議維修": 2}).values, dtype=torch.long)  # 將等級轉為數值標籤
    # 📦 建立資料集與 DataLoader
    dataset = torch.utils.data.TensorDataset(X, y)  # 將特徵與標籤打包成 PyTorch 資料集
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)  # 建立資料載入器，批次大小為 32，並打亂順序
    # 🧠 建立模型與訓練元件
    model = DepreciationMLP()  # 建立 MLP 模型（使用預設架構）
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵作為分類損失函式
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用 Adam 優化器，學習率為 0.01
    # 🔁 模型訓練迴圈
    for epoch in range(50):  # 訓練 50 個 epoch
        for batch_x, batch_y in loader:  # 遍歷每個批次
            optimizer.zero_grad()  # 清除前一次的梯度
            logits = model(batch_x)  # 前向傳播，取得預測結果
            loss = criterion(logits, batch_y)  # 計算損失
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新模型參數
    torch.save(model, output_path)  # 儲存訓練完成的模型至指定路徑
    print(f"✅ 模型已訓練並儲存至 {output_path}")  # 顯示訓練完成訊息

def compute_depreciation_metrics(defects):
    """
    📌 用途：根據缺陷清單計算折舊分析的核心指標(defect_index)。
    每個缺陷的影響力以「面積 × 深度」表示，並統計整體缺陷數量、平均深度、最大深度與總面積。
    🔢 回傳內容：
    - defect_index：折舊指數（面積 × 深度 的加總）
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
    📌 用途：根據折舊指數（defect_index）進行分級判斷，協助使用者快速了解物件目前的健康狀態。
    📊 分級邏輯：
    - A：正常（defect_index < 50）
    - B：觀察中（50 ≤ defect_index < 150）
    - C：建議維修（defect_index ≥ 150）
    🔁 回傳：
    對應的折舊等級字串（含建議）
    """
    if defect_index < 50:
        return "A - 正常"
    elif defect_index < 150:
        return "B - 觀察中"
    else:
        return "C - 建議維修"

def generate_depreciation_record(defects,mlp_model=None):
    """
    📌 用途：整合折舊分析流程，生成一筆完整的紀錄。
    包含分析時間、折舊等級、各項指標與原始缺陷清單，方便儲存、追蹤與可視化。
    🧩 組成：
    - timestamp：分析時間（格式：YYYY-MM-DD HH:MM）
    - grade：折舊等級（由 classify_depreciation 判斷）
    - defect_index / defect_count / avg_depth / max_depth / total_area：由 compute_depreciation_metrics 計算
    - defects：原始缺陷清單（含面積、深度、位置等）
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
    預測折舊等級(A、B、C 三類）。
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
    🔢 輸入：metrics dict（包含 defect_index、avg_depth、max_depth、total_area）
    🔁 回傳：tuple → (折舊等級字串, 信心分數)
    """
    # 🧮 將指標轉為張量並加上 batch 維度
    input_tensor = torch.tensor([
        metrics["defect_index"],
        metrics["avg_depth"],
        metrics["max_depth"],
        metrics["total_area"]
    ], dtype=torch.float32).unsqueeze(0)

    # 🚫 推論模式（停用梯度）
    with torch.no_grad():
        logits = mlp_model(input_tensor)  # 前向傳播,logits 是 MLP 模型的原始輸出（未經標準化），通常是每個類別的分數
        probs = torch.softmax(logits, dim=1)  # 計算 softmax 機率分布，softmax 將 logits 轉換為機率分布，使所有類別的機率加總為 1。
        #EX:logits = [1.2, 3.5, 0.8] (所有類別的機率加總為1時)-> probs = [0.12, 0.82, 0.06]
        pred = torch.argmax(probs, dim=1).item()  # 取得預測類別索引，pred 是最大機率的類別索引（即模型預測的分類）。
        confidence = probs[0, pred].item()  # 取得該類別的信心分數，是該類別的機率值，代表模型對這個預測的信心。

    # 📤 回傳等級與信心分數
    label = ["A - 正常", "B - 觀察中", "C - 建議維修"]
    # 信心分數(confidence)的用途：
    # 報表呈現：讓使用者知道預測是否可靠
    # 警示機制：若信心低於某門檻（例如 0.6），可標示為「不確定」
    # 模型評估：可用於 ROC 曲線、Precision-Recall 分析
    # 決策支援：高信心可自動通過，低信心可轉人工複核
    return label[pred], confidence

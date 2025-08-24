import numpy as np
from datetime import datetime
import torch
# 📌 模組用途：
# 本模組用於分析影像中的缺陷資訊，並根據缺陷面積與深度計算折舊指數（defect_index），
# 進一步判斷物件的折舊等級（正常／觀察中／建議維修），並生成一筆完整的折舊分析紀錄。
# 適用於品質監控、設備維護、製程追蹤等場景，可整合至 dashboard 或報表系統。
def compute_depreciation_metrics(defects):
    """
    📌 用途：
    根據缺陷清單計算折舊分析的核心指標。
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
    📌 用途：
    根據折舊指數（defect_index）進行分級判斷，協助使用者快速了解物件目前的健康狀態。
    
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
    📌 用途：
    整合折舊分析流程，生成一筆完整的紀錄。
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
    if mlp_model:
        grade = classify_depreciation_mlp(metrics, mlp_model)
    else:
        grade = classify_depreciation(metrics["defect_index"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return {
        "timestamp": timestamp,
        "grade": grade,
        **metrics,
        "defects": defects
    }
import torch.nn as nn

class DepreciationMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def classify_depreciation_mlp(metrics, mlp_model):
    """
    使用 MLP 模型根據缺陷指標預測折舊等級。
    🔢 輸入：metrics dict（包含 defect_index、avg_depth、max_depth、total_area）
    🔁 回傳：折舊等級字串
    """
    input_tensor = torch.tensor([
        metrics["defect_index"],
        metrics["avg_depth"],
        metrics["max_depth"],
        metrics["total_area"]
    ], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = mlp_model(input_tensor)
        pred = torch.argmax(logits, dim=1).item()

    return ["A - 正常", "B - 觀察中", "C - 建議維修"][pred]
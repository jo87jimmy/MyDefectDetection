import numpy as np
from datetime import datetime
# 📌 模組用途：
# 本模組用於分析影像中的缺陷資訊，並根據缺陷面積與深度計算折舊指數（defect_index），
# 進一步判斷物件的折舊等級（正常／觀察中／建議維修），並生成一筆完整的折舊分析紀錄。
# 適用於品質監控、設備維護、製程追蹤等場景，可整合至 dashboard 或報表系統。
def compute_depreciation_metrics(defects):
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
    if defect_index < 50:
        return "A - 正常"
    elif defect_index < 150:
        return "B - 觀察中"
    else:
        return "C - 建議維修"

def generate_depreciation_record(defects):
    metrics = compute_depreciation_metrics(defects)
    grade = classify_depreciation(metrics["defect_index"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return {
        "timestamp": timestamp,
        "grade": grade,
        **metrics,
        "defects": defects
    }

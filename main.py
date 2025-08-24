import torch
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from depreciation_analysis import generate_depreciation_record
from train_depreciation_mlp import train_mlp_from_csv

from train_depreciation_mlp import DepreciationMLP  # 先把 class 匯入
class FullModel(torch.nn.Module):
    def __init__(self, encoder, bn, decoder):
        super().__init__()
        self.encoder = encoder
        self.bn = bn
        self.decoder = decoder

    def forward(self, x):
        feats = self.encoder(x)
        recons = self.decoder(self.bn(feats))
        return feats, recons
# ===== 輔助函式 =====
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min + 1e-8)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='a'):
    anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for fs, ft in zip(fs_list, ft_list):
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].cpu().numpy()
        a_map_list.append(a_map)
        anomaly_map += a_map
    return anomaly_map, a_map_list

def extract_defect_regions(anomaly_map, threshold=0.6):
    norm_map = min_max_norm(anomaly_map)
    binary_mask = (norm_map > threshold).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # 建立遮罩並找最深點
        mask = np.zeros_like(anomaly_map, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, -1)
        masked_anomaly = np.where(mask == 1, anomaly_map, -np.inf)
        max_idx = np.unravel_index(np.argmax(masked_anomaly), anomaly_map.shape)
        cx, cy = max_idx[1], max_idx[0]  # 注意：OpenCV 是 (x, y)

        depth = anomaly_map[max_idx]

        defects.append({
            "area": area,
            "center": (cx, cy),
            "size": (w, h),
            "depth": depth
        })

    return defects

def extract_and_annotate_defects(img, anomaly_map, threshold=0.6):
    smoothed_map = cv2.GaussianBlur(anomaly_map, (5, 5), sigmaX=2)
    norm_map = min_max_norm(smoothed_map)
    binary_mask = (norm_map > threshold).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img.copy()
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        # 建立遮罩並找最深點
        mask = np.zeros_like(anomaly_map, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, -1)
        masked_anomaly = np.where(mask == 1, anomaly_map, -np.inf)
        max_idx = np.unravel_index(np.argmax(masked_anomaly), anomaly_map.shape)
        cx, cy = max_idx[1], max_idx[0]
        depth = anomaly_map[max_idx]

        # 🔸 畫不規則邊界
        cv2.drawContours(annotated, [cnt], -1, (255, 0, 0), 1)

        # 🔸 標註文字在最深點
        label1 = f"#{i+1}"
        label2 = f"a:{int(area)}"
        label3 = f"d:{depth:.2f}"

        line_height = 8 # 每行垂直間距（可微調）
        cv2.putText(annotated, label1, (cx, cy - line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(annotated, label2, (cx, cy - line_height),     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(annotated, label3, (cx, cy),                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.circle(annotated, (cx, cy), 1, (255, 0, 0), -1)  # 紅色小圓點

    return annotated

import pandas as pd
import os

def save_record_to_csv(record, csv_path="depreciation_records.csv"):
    df = pd.DataFrame([record])  # 單筆轉成 DataFrame
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
        
import re
def safe_load(path, map_location="cpu", weights_only=True, extra_globals=None):
    """
    自動安全載入 torch 模型檔案 (.pth)
    會根據錯誤訊息自動將缺少的 global 類別/函式加入安全清單
    
    Args:
        path (str): 模型檔案路徑
        map_location: 載入到哪個裝置 (預設 "cpu")
        weights_only (bool): 是否只載入權重 (推薦 True)
        extra_globals (list): 額外要允許的類別，例如 [MyModel]
    """
    if extra_globals:
        torch.serialization.add_safe_globals(extra_globals)

    while True:
        try:
            return torch.load(path, map_location=map_location, weights_only=weights_only)
        except Exception as e:
            msg = str(e)
            # 嘗試解析錯誤訊息中的類別名稱
            match = re.search(r"Unsupported global: GLOBAL (.+?) ", msg)
            if match:
                global_name = match.group(1)
                print(f"⚠️ 檔案需要允許：{global_name}")

                # 對於內建類型 (例如 builtins.set)，用 eval 取到對象
                try:
                    obj = eval(global_name.replace("builtins.", ""))
                    torch.serialization.add_safe_globals([obj])
                    print(f"✅ 已允許 {global_name}")
                except Exception as e2:
                    print(f"❌ 無法自動允許 {global_name}, 請手動加入: {e2}")
                    raise
            else:
                raise  # 不是 Unsupported global 就直接丟出

# ===== 主程式 =====
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1.一行載入完整模型（需訓練時整個 torch 模型保存）
    model = torch.load("fullmodel_wres50_bottle.pth", map_location=device, weights_only=False)
    model.to(device).eval()
    # feats, recons = model(img_tensor)

    # 2.讀取單張圖片
    img_bgr = cv2.imread("test_bottle.png")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # 3.推論
    with torch.no_grad():
        feats, recons = model(img_tensor)  # 假設你的模型 forward 回傳 (features, reconstructions)
        anomaly_map, _ = cal_anomaly_map([feats[-1]], [recons[-1]], img_tensor.shape[-1])
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        ano_map_norm = min_max_norm(anomaly_map) * 255
        ano_map_color = cvt2heatmap(ano_map_norm)

    # 4.疊加熱力圖
    overlay = show_cam_on_image(img_resized, ano_map_color)

    # 5.儲存
    cv2.imwrite("heatmap_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("✅ 單張影像缺陷熱力圖已完成 → heatmap_overlay.png")
    
    # 6. 缺陷區域分析/缺陷區域標註
    defects = extract_defect_regions(anomaly_map, threshold=0.8)
    for i, d in enumerate(defects):
        print(f"🔧 缺陷 {i+1}: 面積={d['area']:.1f}, 中心={d['center']}, 長寬={d['size']}, 深度={d['depth']:.3f}")
    annotated_img = extract_and_annotate_defects(img_resized, anomaly_map, threshold=0.8)
    cv2.imwrite("heatmap_annotated.png", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    print("📌 缺陷區域已標註 → heatmap_annotated.png")
    
    # 7.折舊分析
    record = generate_depreciation_record(defects)
    # ✅ 印出折舊分析結果
    print(f"\n📊 折舊分析報告（{record['timestamp']}）")
    print(f"等級：{record['grade']}")
    print(f"缺陷數量：{record['defect_count']}")
    print(f"總面積：{record['total_area']:.1f}")
    print(f"平均深度：{record['avg_depth']:.2f}")
    print(f"最大深度：{record['max_depth']:.2f}")
    print(f"折舊指數：{record['defect_index']:.2f}")
    
    #8.加入紀錄資料的CSV儲存
    save_record_to_csv(record)
    print("✅ 已儲存紀錄至 CSV")
    #9.訓練MLP模型
    # 可選：每新增 50 筆就 retrain
    if len(pd.read_csv("depreciation_records.csv")) % 1 == 0:
        train_mlp_from_csv()
    print("✅ 已重新訓練 MLP 模型")
    # ✅ 把 DepreciationMLP 加進 PyTorch 安全清單
    mlp_model = safe_load(
        "depreciation_mlp.pth",
        map_location=device,   # 或 device
        weights_only=True,
        extra_globals=[DepreciationMLP]  # 先加自己的模型 class
    )
    # torch.serialization.add_safe_globals([DepreciationMLP,set])
    # 載入 MLP 模型（需事先訓練好並儲存）
    # mlp_model = torch.load("depreciation_mlp.pth", map_location=device,weights_only=True)
    mlp_model.eval()

    # 折舊分析（使用 MLP）
    record = generate_depreciation_record(defects, mlp_model=mlp_model)
    # ✅ 印出完整折舊分析紀錄（含 MLP 等級）
print("\n📊 折舊分析紀錄（使用 MLP 模型）")
for key, value in record.items():
    if key != "defects":
        print(f"{key}: {value}")
    else:
        print(f"{key}:")
        for i, defect in enumerate(value):
            print(f"  🔧 缺陷 {i+1}: 面積={defect['area']:.1f}, 中心={defect['center']}, 長寬={defect['size']}, 深度={defect['depth']:.3f}")
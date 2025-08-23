import torch
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from depreciation_analysis import generate_depreciation_record

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

# ===== 主程式 =====
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ 一行載入完整模型（需訓練時整個 torch 模型保存）
    model = torch.load("fullmodel_wres50_bottle.pth", map_location=device, weights_only=False)
    model.to(device).eval()
    # feats, recons = model(img_tensor)

    # 2️⃣ 讀取單張圖片
    img_bgr = cv2.imread("test_bottle.png")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # 3️⃣ 推論
    with torch.no_grad():
        feats, recons = model(img_tensor)  # 假設你的模型 forward 回傳 (features, reconstructions)
        anomaly_map, _ = cal_anomaly_map([feats[-1]], [recons[-1]], img_tensor.shape[-1])
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        ano_map_norm = min_max_norm(anomaly_map) * 255
        ano_map_color = cvt2heatmap(ano_map_norm)

    # 4️⃣ 疊加熱力圖
    overlay = show_cam_on_image(img_resized, ano_map_color)

    # 5️⃣ 儲存
    cv2.imwrite("heatmap_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("✅ 單張影像缺陷熱力圖已完成 → heatmap_overlay.png")
    
    defects = extract_defect_regions(anomaly_map, threshold=0.8)
    for i, d in enumerate(defects):
        print(f"🔧 缺陷 {i+1}: 面積={d['area']:.1f}, 中心={d['center']}, 長寬={d['size']}, 深度={d['depth']:.3f}")
    annotated_img = extract_and_annotate_defects(img_resized, anomaly_map, threshold=0.8)
    cv2.imwrite("heatmap_annotated.png", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    print("📌 缺陷區域已標註 → heatmap_annotated.png")
    
    # 6️⃣ 折舊分析
    record = generate_depreciation_record(defects)
    # ✅ 印出折舊分析結果
    print(f"\n📊 折舊分析報告（{record['timestamp']}）")
    print(f"等級：{record['grade']}")
    print(f"缺陷數量：{record['defect_count']}")
    print(f"總面積：{record['total_area']:.1f}")
    print(f"平均深度：{record['avg_depth']:.2f}")
    print(f"最大深度：{record['max_depth']:.2f}")
    print(f"折舊指數：{record['defect_index']:.2f}")
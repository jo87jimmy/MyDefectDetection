import torch
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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

def generate_binary_mask(score_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    將 anomaly score map 轉成二值遮罩
    score_map: shape [1, H, W] 或 [H, W]
    threshold: 閾值，超過視為異常
    """
    if score_map.dim() == 3:
        score_map = score_map.squeeze(0)
    binary_mask = (score_map > threshold).float()
    return binary_mask  # shape: [H, W]

def analyze_defect_regions(binary_mask: torch.Tensor, image_shape=(256, 256), edge_margin=10):
    """
    深度分析二值遮罩中的缺陷區域，並自動分類缺陷類型
    回傳每個缺陷的面積、位置、形狀特徵、分類標籤與統計摘要
    """

    mask_np = binary_mask.cpu().numpy().astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defect_info = []
    total_area = 0
    max_area = 0
    min_area = float('inf')

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area == 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2) if h != 0 else 0
        perimeter = cv2.arcLength(cnt, True)
        compactness = round(4 * np.pi * area / (perimeter ** 2), 3) if perimeter != 0 else 0
        cx, cy = x + w // 2, y + h // 2
        near_edge = (
            x < edge_margin or y < edge_margin or
            x + w > image_shape[0] - edge_margin or
            y + h > image_shape[1] - edge_margin
        )

        defect_info.append({
            'area': area,
            'bbox': (x, y, w, h),
            'center': (cx, cy),
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'near_edge': near_edge
        })

        total_area += area
        max_area = max(max_area, area)
        min_area = min(min_area, area)

    summary = {
        'count': len(defect_info),
        'total_area': total_area,
        'max_area': max_area,
        'min_area': min_area if defect_info else 0,
        'avg_area': round(total_area / len(defect_info), 2) if defect_info else 0
    }

    # 加入分類標籤
    for defect in defect_info:
        defect['defect_type'] = classify_defect(defect, summary['avg_area'])

    return defect_info, summary
# 缺陷分類器邏輯整合
def classify_defect(defect, avg_area):
    """
    根據形狀與位置特徵分類缺陷
    """
    types = []

    if defect['compactness'] < 0.3 and defect['aspect_ratio'] > 2:
        types.append("細長型裂痕")

    if defect['near_edge']:
        types.append("邊緣缺陷")

    if defect['area'] > avg_area * 2:
        types.append("重大缺陷")

    if not types:
        types.append("一般缺陷")

    return types

# 7️⃣ 缺陷分類視覺化
def draw_defect_boxes(image: np.ndarray, defect_info: list, save_path: str = "defect_visualized.png"):
    """
    Draw colored bounding boxes and English labels based on defect type
    """
    color_map = {
        "重大缺陷": (0, 0, 255),         # Red
        "邊緣缺陷": (255, 0, 0),         # Blue
        "細長型裂痕": (0, 165, 255),     # Orange
        "一般缺陷": (255, 255, 255)      # White
    }

    type_translation = {
        "重大缺陷": "Major Defect",
        "邊緣缺陷": "Edge Defect",
        "細長型裂痕": "Elongated Crack",
        "一般缺陷": "Normal Defect"
    }

    img_vis = image.copy()

    for i, defect in enumerate(defect_info):
        x, y, w, h = defect['bbox']
        types = defect['defect_type']
        color = color_map.get(types[0], (128, 128, 128))  # Use first type for color

        # Translate types to English
        translated = [type_translation.get(t, t) for t in types]
        label = f"Defect {i+1}: " + "/".join(translated)

        # Draw box and label
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), color=color, thickness=2)
        cv2.putText(img_vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(save_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    print(f"✅ Defect classification visualization saved → {save_path}")


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
    
    
    
    
    
    # 5️⃣ 二值遮罩
    binary_mask = generate_binary_mask(torch.tensor(ano_map_norm / 255.0), threshold=0.5)
    mask_np = (binary_mask.numpy() * 255).astype(np.uint8)
    cv2.imwrite("binary_mask.png", mask_np)
    print("✅ 二值遮罩完成 → binary_mask.png")

    # 6️⃣ 缺陷分析進階
    defect_info, summary = analyze_defect_regions(binary_mask)

    print("✅ 缺陷區域分析完成：")
    print(f"  🔸 總缺陷數：{summary['count']}")
    print(f"  🔸 總面積：{summary['total_area']}")
    print(f"  🔸 最大缺陷面積：{summary['max_area']}")
    print(f"  🔸 最小缺陷面積：{summary['min_area']}")
    print(f"  🔸 平均缺陷面積：{summary['avg_area']}")
    print()

    for i, defect in enumerate(defect_info):
        print(f"🔹 缺陷 {i+1}")
        print(f"    ▸ 面積        : {defect['area']}")
        print(f"    ▸ BBox        : {defect['bbox']}")
        print(f"    ▸ 中心座標    : {defect['center']}")
        print(f"    ▸ 長寬比      : {defect['aspect_ratio']}")
        print(f"    ▸ 圓度        : {defect['compactness']}")
        print(f"    ▸ 靠邊緣      : {'是' if defect['near_edge'] else '否'}")
        print(f"    ▸ 缺陷類型    : {', '.join(defect['defect_type'])}")
        print()
    # 畫出缺陷分類框並儲存
    draw_defect_boxes(img_resized, defect_info, save_path="defect_visualized.png")

              
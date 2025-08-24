import torch  # 匯入 PyTorch，用於張量運算與模型構建
import cv2  # 匯入 OpenCV，用於影像處理
import numpy as np  # 匯入 NumPy，用於數值運算
from scipy.ndimage import gaussian_filter  # 匯入高斯濾波器，用於平滑 anomaly map
import torch.nn.functional as F  # 匯入 PyTorch 的函式庫，用於計算 cosine similarity 等
from train_depreciation_mlp import DepreciationMLP,train_mlp_from_csv,generate_depreciation_record  # 從自訂模組匯入 MLP 類別與訓練、分析函式

class FullModel(torch.nn.Module):
    """ 模組匯入與模型定義，包含 encoder、batch normalization、decoder """
    def __init__(self, encoder, bn, decoder):
        super().__init__()
        self.encoder = encoder  # 編碼器，用於提取特徵
        self.bn = bn  # 批次正規化層
        self.decoder = decoder  # 解碼器，用於重建影像

    def forward(self, x):
        feats = self.encoder(x)  # 提取特徵
        recons = self.decoder(self.bn(feats))  # 經過 BN 後重建影像
        return feats, recons  # 回傳特徵與重建結果

def min_max_norm(image):
    """將輸入的影像（通常是 anomaly map 或灰階圖）進行「最小最大值正規化」，
    讓所有像素值都落在 0 到 1 之間。讓後續的處理（例如二值化、熱力圖轉換）更穩定一致。
    image.min()：找出影像中最暗的像素值
    image.max()：找出影像中最亮的像素值
    (image - a_min) / (a_max - a_min + 1e-8)：將所有像素值依照最大最小範圍進行縮放，
    1e-8 是為了避免除以 0 的錯誤。"""
    a_min, a_max = image.min(), image.max()  # 取得影像中最小值與最大值
    return (image - a_min) / (a_max - a_min + 1e-8)  # 將影像進行 Min-Max 正規化，避免除以 0

def cvt2heatmap(gray):
    """將灰階影像（通常是 anomaly map 或正規化後的深度圖）轉換成彩色熱力圖，讓缺陷區域的視覺效果更明顯。
    np.uint8(gray)：將灰階影像轉成 8-bit 格式（0–255），以符合 OpenCV 的色彩映射需求
    cv2.COLORMAP_JET：使用 JET 色盤，常見於熱力圖，顏色從藍 → 綠 → 黃 → 紅，紅色代表高值（通常是缺陷最深處）
    這個步驟通常會接在 min_max_norm 之後，讓 anomaly map 的高值區域在熱力圖中呈現紅色、低值區域呈現藍色，方便辨識。 """
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)  # 將灰階影像轉換為彩色熱力圖（使用 JET 色盤）
    return heatmap  # 回傳彩色熱力圖

def show_cam_on_image(img, anomaly_map):
    """將anomaly map（通常是熱力圖）疊加在原始影像上，產生視覺化的疊圖效果，讓缺陷區域一目了然。
    np.float32(...) / 255：將影像與 anomaly map 都轉成 0~1 的浮點格式，方便進行加法疊加
    cam = cam / np.max(cam)：將疊加後的結果再正規化，確保整體亮度不超過 1
    np.uint8(255 * cam)：將結果轉回 0~255 的整數格式，方便用 OpenCV 儲存或顯示
    通常會在 cvt2heatmap 之後執行，讓 anomaly map 的彩色熱力圖能夠與原始 RGB 影像融合，產生清楚的視覺提示。 """
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255  # 將 anomaly map 疊加在原始影像上（都轉為 0~1 範圍）
    cam = cam / np.max(cam)  # 正規化疊加結果，避免超過 1
    return np.uint8(255 * cam)  # 將結果轉回 0~255 的 uint8 格式，方便儲存或顯示

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='a'):
    """     根據模型的特徵輸出與重建結果，計算 anomaly map（異常分佈圖），用來判斷影像中哪些區域可能存在缺陷。
        fs_list：原始特徵列表（通常是 encoder 的輸出）
        ft_list：重建特徵列表（通常是 decoder 的輸出）
        F.cosine_similarity(fs, ft)：計算每個位置的特徵相似度，值越小代表越異常
        1 - 相似度：將相似度轉為異常分數（高分代表異常）
        F.interpolate(...)：將每層的 anomaly map 放大到統一尺寸，方便融合與視覺化
        anomaly_map += a_map：將多層 anomaly map 疊加，形成最終的異常分佈圖
        通常會接在模型推論之後，並在 gaussian_filter 平滑後進行視覺化與缺陷分析。 """    
    anomaly_map = np.zeros([out_size, out_size])  # 初始化 anomaly map，大小為輸出尺寸
    a_map_list = []  # 儲存每層的 anomaly map（可選）
    for fs, ft in zip(fs_list, ft_list):  # 遍歷每一層的特徵與重建結果
        a_map = 1 - F.cosine_similarity(fs, ft)  # 計算 cosine 相似度，轉為 anomaly 分數（越不相似越異常）
        a_map = torch.unsqueeze(a_map, dim=1)  # 增加 channel 維度，符合 interpolate 的輸入格式
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)  # 將 anomaly map 放大到指定尺寸
        a_map = a_map[0, 0, :, :].cpu().numpy()  # 轉為 NumPy 格式並移除 batch/channel 維度
        a_map_list.append(a_map)  # 儲存每層的 anomaly map
        anomaly_map += a_map  # 累加 anomaly map（多層融合）
    return anomaly_map, a_map_list  # 回傳最終 anomaly map 與每層的 map 清單

def extract_defect_regions(anomaly_map, threshold=0.6):
    """從 anomaly map 中偵測並提取缺陷區域的核心邏輯
    會根據 anomaly map 的異常分佈，偵測出所有明顯的缺陷區域，並針對每個區域計算：
    面積（area）
    中心點（最深點）
    寬高（bounding box）
    深度（最大 anomaly 值）
    後續的折舊分析模組使用，來評估物件的健康狀態與維修建議。 """
    norm_map = min_max_norm(anomaly_map)  # 將 anomaly map 正規化到 0~1 範圍
    binary_mask = (norm_map > threshold).astype(np.uint8)  # 根據門檻值建立二值遮罩（1 表示異常）
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找出所有外部輪廓（缺陷區域）
    defects = []  # 儲存所有缺陷資訊
    for cnt in contours:  # 遍歷每個輪廓
        area = cv2.contourArea(cnt)  # 計算輪廓面積
        if area < 10:  # 忽略太小的區域（可能是雜訊）
            continue
        x, y, w, h = cv2.boundingRect(cnt)  # 取得缺陷區域的邊界框（左上角座標與寬高）
        # 建立遮罩並找出 anomaly map 中該區域最深的點（最大值）
        mask = np.zeros_like(anomaly_map, dtype=np.uint8)  # 建立與 anomaly map 同大小的遮罩
        cv2.drawContours(mask, [cnt], -1, 1, -1)  # 將輪廓填滿在遮罩上
        masked_anomaly = np.where(mask == 1, anomaly_map, -np.inf)  # 只保留遮罩內的 anomaly 值，其餘設為 -∞
        max_idx = np.unravel_index(np.argmax(masked_anomaly), anomaly_map.shape)  # 找出最大值的位置（最深點）
        cx, cy = max_idx[1], max_idx[0]  # OpenCV 座標順序為 (x, y)，所以要反轉
        depth = anomaly_map[max_idx]  # 取得最深點的 anomaly 值（代表缺陷深度）
        defects.append({  # 將缺陷資訊儲存為 dict
            "area": area,  # 缺陷面積
            "center": (cx, cy),  # 缺陷中心點座標（最深點）
            "size": (w, h),  # 缺陷的寬高
            "depth": depth  # 缺陷深度
        })
    return defects  # 回傳所有缺陷資訊列表

def extract_and_annotate_defects(img, anomaly_map, threshold=0.6):
    """
    用來將缺陷區域標註在原始影像上，並加上文字說明與視覺提示 
    根據 anomaly map 的異常分佈，將偵測到的缺陷區域：
    用藍色輪廓框出
    在最深點加上紅色文字標註（編號、面積、深度）
    畫出紅色小圓點作為視覺焦點
    適合用於報表、GUI、dashboard 顯示，以掌握缺陷位置與嚴重程度。
    """
    smoothed_map = cv2.GaussianBlur(anomaly_map, (5, 5), sigmaX=2)  # 對 anomaly map 套用高斯模糊，平滑邊緣
    norm_map = min_max_norm(smoothed_map)  # 將模糊後的 anomaly map 正規化到 0~1
    binary_mask = (norm_map > threshold).astype(np.uint8) * 255  # 根據門檻值建立二值遮罩（255 表示異常區域）
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 找出所有外部輪廓（缺陷區域）
    annotated = img.copy()  # 建立影像副本，用來繪製標註
    for i, cnt in enumerate(contours):  # 遍歷每個輪廓
        area = cv2.contourArea(cnt)  # 計算輪廓面積
        if area < 10:  # 忽略太小的區域（可能是雜訊）
            continue
        # 建立遮罩並找出 anomaly map 中該區域最深的點（最大值）
        mask = np.zeros_like(anomaly_map, dtype=np.uint8)  # 建立與 anomaly map 同大小的遮罩
        cv2.drawContours(mask, [cnt], -1, 1, -1)  # 將輪廓填滿在遮罩上
        masked_anomaly = np.where(mask == 1, anomaly_map, -np.inf)  # 只保留遮罩內的 anomaly 值，其餘設為 -∞
        max_idx = np.unravel_index(np.argmax(masked_anomaly), anomaly_map.shape)  # 找出最大值的位置（最深點）
        cx, cy = max_idx[1], max_idx[0]  # OpenCV 座標順序為 (x, y)，所以要反轉
        depth = anomaly_map[max_idx]  # 取得最深點的 anomaly 值（代表缺陷深度）
        # 🔸 畫出不規則邊界（藍色）
        cv2.drawContours(annotated, [cnt], -1, (255, 0, 0), 1)
        # 🔸 在最深點標註文字（紅色）
        label1 = f"#{i+1}"  # 缺陷編號
        label2 = f"a:{int(area)}"  # 面積
        label3 = f"d:{depth:.2f}"  # 深度
        line_height = 8  # 每行文字的垂直間距（可微調）
        cv2.putText(annotated, label1, (cx, cy - line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(annotated, label2, (cx, cy - line_height),     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(annotated, label3, (cx, cy),                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.circle(annotated, (cx, cy), 1, (255, 0, 0), -1)  # 在最深點畫紅色小圓點
    return annotated  # 回傳已標註的影像

import pandas as pd
import os

def save_record_to_csv(record, csv_path="depreciation_records.csv"):
    """
    負責將折舊分析結果儲存成 CSV 檔案
    將每次的折舊分析結果（record）儲存到指定的 CSV 檔案中，支援持續累積紀錄，方便：
    建立歷史資料庫
    進行趨勢分析
    觸發 MLP 模型訓練
    🔧 設計細節：
    使用 pandas.DataFrame 將 dict 包裝成表格格式
    若檔案已存在，則以 mode='a' 附加寫入，避免覆蓋舊資料
    header=False 表示不重複寫欄位名稱
    index=False 表示不寫入 DataFrame 的索引欄
    """
    df = pd.DataFrame([record])  # 將單筆紀錄轉成 DataFrame（以 list 包裝 dict）
    if os.path.exists(csv_path):  # 檢查 CSV 檔案是否已存在
        df.to_csv(csv_path, mode='a', header=False, index=False)  # 若存在則以附加模式寫入，不重複標題
    else:
        df.to_csv(csv_path, index=False)  # 若不存在則建立新檔並寫入，包含欄位標題

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
        torch.serialization.add_safe_globals(extra_globals)# 若有提供額外類別，先加入安全清單
    while True:
        try:
            return torch.load(path, map_location=map_location, weights_only=weights_only)# 嘗試載入模型
        except Exception as e:
            msg = str(e)# 取得錯誤訊息文字
            # 嘗試解析錯誤訊息中的類別名稱（通常是缺少的 global 類別）
            match = re.search(r"Unsupported global: GLOBAL (.+?) ", msg)
            if match:
                global_name = match.group(1)# 取得缺少的類別名稱
                print(f"⚠️ 檔案需要允許：{global_name}")
                # 對於內建類型 (例如 builtins.set)，用 eval 取到對象
                try:
                    obj = eval(global_name.replace("builtins.", ""))# 將文字轉為物件
                    torch.serialization.add_safe_globals([obj])# 加入安全清單
                    print(f"✅ 已允許 {global_name}")
                except Exception as e2:
                    print(f"❌ 無法自動允許 {global_name}, 請手動加入: {e2}")# 若無法自動處理則提示手動加入
                    raise # 丟出例外，終止載入流程
            else:
                raise  # 不是 Unsupported global 就直接丟出

# ===== 主程式 =====
if __name__ == "__main__":  # 判斷是否為主程式執行（避免被其他模組匯入時執行）
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 根據環境選擇 GPU 或 CPU 裝置
    # 1. 載入完整模型（需使用 torch.save(model) 儲存的模型）
    model = torch.load("fullmodel_wres50_bottle.pth", map_location=device, weights_only=False)
    model.to(device).eval()  # 移至指定裝置並設為推論模式
    # 2. 讀取單張測試影像
    img_bgr = cv2.imread("test_bottle.png")  # 以 BGR 格式讀入影像
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 轉換為 RGB 格式
    img_resized = cv2.resize(img_rgb, (256, 256))  # 調整尺寸為 256x256
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # 轉為 PyTorch 張量並正規化
    img_tensor = img_tensor.to(device)  # 移至指定裝置
    # 3. 模型推論：取得特徵與重建結果
    with torch.no_grad():  # 停用梯度計算（加速推論）
        feats, recons = model(img_tensor)  # 執行 forward，取得特徵與重建影像
        anomaly_map, _ = cal_anomaly_map([feats[-1]], [recons[-1]], img_tensor.shape[-1])  # 計算 anomaly map
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)  # 套用高斯模糊平滑結果
        ano_map_norm = min_max_norm(anomaly_map) * 255  # 正規化並轉為 0~255 範圍
        ano_map_color = cvt2heatmap(ano_map_norm)  # 轉為彩色熱力圖
    # 4. 疊加熱力圖至原始影像
    overlay = show_cam_on_image(img_resized, ano_map_color)
    # 5. 儲存疊圖結果
    cv2.imwrite("heatmap_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("✅ 單張影像缺陷熱力圖已完成 → heatmap_overlay.png")
    # 6. 缺陷區域分析與標註
    defects = extract_defect_regions(anomaly_map, threshold=0.8)  # 偵測缺陷區域
    for i, d in enumerate(defects):  # 印出每個缺陷的資訊
        print(f"🔧 缺陷 {i+1}: 面積={d['area']:.1f}, 中心={d['center']}, 長寬={d['size']}, 深度={d['depth']:.3f}")
    annotated_img = extract_and_annotate_defects(img_resized, anomaly_map, threshold=0.8)  # 標註缺陷區域
    cv2.imwrite("heatmap_annotated.png", cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    print("📌 缺陷區域已標註 → heatmap_annotated.png")
    # 7. 折舊分析（使用 rule-based 分級）
    record = generate_depreciation_record(defects)
    print(f"\n📊 折舊分析報告（{record['timestamp']}）")
    print(f"等級：{record['grade']}")
    print(f"缺陷數量：{record['defect_count']}")
    print(f"總面積：{record['total_area']:.1f}")
    print(f"平均深度：{record['avg_depth']:.2f}")
    print(f"最大深度：{record['max_depth']:.2f}")
    print(f"折舊指數：{record['defect_index']:.2f}")
    # 8. 儲存紀錄至 CSV 檔案
    save_record_to_csv(record)
    print("✅ 已儲存紀錄至 CSV")
    # 9. 訓練 MLP 模型（可依紀錄數量條件觸發）
    if len(pd.read_csv("depreciation_records.csv")) % 1 == 0:  # 每新增 1 筆就 retrain（可調整條件）
        train_mlp_from_csv()
    print("✅ 已重新訓練 MLP 模型")
    # 10. 載入 MLP 模型（使用 safe_load 確保安全）
    mlp_model = safe_load(
        "depreciation_mlp.pth",
        map_location=device,
        weights_only=True,
        extra_globals=[DepreciationMLP]  # 加入自訂類別至安全清單
    )
    mlp_model.eval()  # 設為推論模式
    # 11. 使用 MLP 模型進行折舊分析
    record = generate_depreciation_record(defects, mlp_model=mlp_model)
    print("\n📊 折舊分析紀錄（使用 MLP 模型）")
    for key, value in record.items():
        if key != "defects":
            print(f"{key}: {value}")  # 印出分析指標
        else:
            print(f"{key}:")  # 印出缺陷清單
            for i, defect in enumerate(value):
                print(f"  🔧 缺陷 {i+1}: 面積={defect['area']:.1f}, 中心={defect['center']}, 長寬={defect['size']}, 深度={defect['depth']:.3f}")
    print("✅ 已完成 MLP 折舊分析")
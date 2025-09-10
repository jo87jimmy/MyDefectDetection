import torch  # 匯入 PyTorch，用於張量運算與模型構建
import cv2  # 匯入 OpenCV，用於影像處理
import numpy as np  # 匯入 NumPy，用於數值運算
from scipy.ndimage import gaussian_filter  # 匯入高斯濾波器，用於平滑 anomaly map
import torch.nn.functional as F  # 匯入 PyTorch 的函式庫，用於計算 cosine similarity 等
from train_depreciation_mlp import save_record_to_csv, interpret_comparison_results, generate_enhanced_depreciation_record, generate_comparison_depreciation_record_with_excel, EnhancedDepreciationMLP, train_enhanced_mlp_from_csv  # 匯入改良版 MLP 類別與分析函式
import pickle
import os
import pandas as pd


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
    heatmap = cv2.applyColorMap(np.uint8(gray),
                                cv2.COLORMAP_JET)  # 將灰階影像轉換為彩色熱力圖（使用 JET 色盤）
    return heatmap  # 回傳彩色熱力圖


def show_cam_on_image(img, anomaly_map):
    """將anomaly map（通常是熱力圖）疊加在原始影像上，產生視覺化的疊圖效果，讓缺陷區域一目了然。
    np.float32(...) / 255：將影像與 anomaly map 都轉成 0~1 的浮點格式，方便進行加法疊加
    cam = cam / np.max(cam)：將疊加後的結果再正規化，確保整體亮度不超過 1
    np.uint8(255 * cam)：將結果轉回 0~255 的整數格式，方便用 OpenCV 儲存或顯示
    通常會在 cvt2heatmap 之後執行，讓 anomaly map 的彩色熱力圖能夠與原始 RGB 影像融合，產生清楚的視覺提示。 """
    cam = np.float32(anomaly_map) / 255 + np.float32(
        img) / 255  # 將 anomaly map 疊加在原始影像上（都轉為 0~1 範圍）
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
        a_map = 1 - F.cosine_similarity(
            fs, ft)  # 計算 cosine 相似度，轉為 anomaly 分數（越不相似越異常）
        a_map = torch.unsqueeze(a_map,
                                dim=1)  # 增加 channel 維度，符合 interpolate 的輸入格式
        a_map = F.interpolate(a_map,
                              size=out_size,
                              mode='bilinear',
                              align_corners=True)  # 將 anomaly map 放大到指定尺寸
        a_map = a_map[0,
                      0, :, :].cpu().numpy()  # 轉為 NumPy 格式並移除 batch/channel 維度
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
    輪廓（contour）
    後續的折舊分析模組使用，來評估物件的健康狀態與維修建議。 """
    norm_map = min_max_norm(anomaly_map)  # 將 anomaly map 正規化到 0~1 範圍
    binary_mask = (norm_map
                   > threshold).astype(np.uint8)  # 根據門檻值建立二值遮罩（1 表示異常）
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)  # 找出所有外部輪廓（缺陷區域）
    defects = []  # 儲存所有缺陷資訊
    for cnt in contours:  # 遍歷每個輪廓
        area = cv2.contourArea(cnt)  # 計算輪廓面積
        if area < 10:  # 忽略太小的區域（可能是雜訊）
            continue
        x, y, w, h = cv2.boundingRect(cnt)  # 取得缺陷區域的邊界框（左上角座標與寬高）
        # 建立遮罩並找出 anomaly map 中該區域最深的點（最大值）
        mask = np.zeros_like(anomaly_map,
                             dtype=np.uint8)  # 建立與 anomaly map 同大小的遮罩
        cv2.drawContours(mask, [cnt], -1, 1, -1)  # 將輪廓填滿在遮罩上
        masked_anomaly = np.where(mask == 1, anomaly_map,
                                  -np.inf)  # 只保留遮罩內的 anomaly 值，其餘設為 -∞
        max_idx = np.unravel_index(np.argmax(masked_anomaly),
                                   anomaly_map.shape)  # 找出最大值的位置（最深點）
        cx, cy = max_idx[1], max_idx[0]  # OpenCV 座標順序為 (x, y)，所以要反轉
        depth = anomaly_map[max_idx]  # 取得最深點的 anomaly 值（代表缺陷深度）
        defects.append({
            "area": float(area),  # 確保是 Python float
            "center": (int(cx), int(cy)),  # 確保是 Python int
            "size": (int(w), int(h)),  # 確保是 Python int
            "depth": float(depth),  # 確保是 Python float
            "contour": cnt  # 將輪廓一併回傳
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
    # 為了讓視覺化輪廓更平滑，對 anomaly map 進行高斯模糊。
    # 這一步驟不影響核心數據（如深度）的準確性。
    smoothed_map = cv2.GaussianBlur(anomaly_map, (5, 5), sigmaX=2)

    # 直接呼叫核心函式來提取所有缺陷的詳細資訊
    # 我們傳入平滑後的 map，以獲得與視覺上一致的輪廓
    defects = extract_defect_regions(smoothed_map, threshold)

    annotated = img.copy()  # 建立影像副本，用來繪製標註

    # 遍歷所有偵測到的缺陷，並在影像上進行標註
    for i, defect in enumerate(defects):
        # 從 defect 物件中直接取得所需資訊
        cnt = defect["contour"]
        area = defect["area"]
        cx, cy = defect["center"]

        # 為了數據的精準度，深度值應從「原始」的 anomaly_map 中獲取，
        # 因為 smoothed_map 的數值可能因模糊而略微降低。
        depth = anomaly_map[cy, cx]

        # 🔸 畫出不規則邊界（藍色）
        cv2.drawContours(annotated, [cnt], -1, (255, 0, 0), 1)

        # 🔸 在最深點標註文字（紅色）
        label1 = f"#{i+1}"  # 缺陷編號
        label2 = f"a:{int(area)}"  # 面積
        label3 = f"d:{depth:.2f}"  # 深度 (使用原始深度)
        line_height = 8  # 每行文字的垂直間距

        cv2.putText(annotated, label1, (cx, cy - line_height * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(annotated, label2, (cx, cy - line_height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.putText(annotated, label3, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 255), 1)

        # 🔸 在最深點畫紅色小圓點作為視覺焦點
        cv2.circle(annotated, (cx, cy), 1, (255, 0, 0), -1)

    return annotated  # 回傳已標註的影像


def clean_csv_file(csv_path):
    """清理損壞的 CSV 檔案，移除格式錯誤的行"""
    try:
        # 逐行讀取並驗證
        valid_lines = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return

        # 保留標題行
        header = lines[0].strip()
        valid_lines.append(header)
        expected_fields = len(header.split(','))

        # 檢查每一行
        for i, line in enumerate(lines[1:], 1):
            line = line.strip()
            if not line:
                continue

            # 簡單的欄位數量檢查
            fields = line.split(',')
            if len(fields) == expected_fields:
                valid_lines.append(line)
            else:
                print(
                    f"⚠️ 跳過第 {i+1} 行（欄位數不符）: {len(fields)} vs {expected_fields}"
                )

        # 重寫檔案
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_lines))

        print(f"✅ 已清理 CSV 檔案，保留 {len(valid_lines)-1} 筆有效記錄")

    except Exception as e:
        print(f"❌ 清理 CSV 檔案失敗: {e}")


import argparse
# ===== 主程式 =====
if __name__ == "__main__":  # 判斷是否為主程式執行（避免被其他模組匯入時執行）
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default='bottle', type=str)  # 訓練類別
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available(
    ) else "cpu"  # 根據環境選擇 GPU 或 CPU 裝置
    # 1. 載入完整模型（需使用 torch.save(model) 儲存的模型）
    model = torch.load("fullmodel_wres50_bottle.pth",
                       map_location=device,
                       weights_only=False)
    model.to(device).eval()  # 移至指定裝置並設為推論模式
    test_path = './mvtec/' + args.category + '/test'  # 測試資料路徑
    items = ['good', 'broken_large', 'broken_small', 'contamination']  # 測試資料標籤
    print(f"🔍 測試資料夾：{test_path}，共 {len(items)} 類別")

# 依類別逐張讀取影像並執行推論
for item in items:
    item_path = os.path.join(test_path, item)
    img_files = [
        f for f in os.listdir(item_path)
        if f.endswith('.png') or f.endswith('.jpg')
    ]

    print(f"\n📂 類別：{item}，共 {len(img_files)} 張影像")

    for img_name in img_files:
        img_path = os.path.join(item_path, img_name)
        print(f"\n🖼️ 處理影像：{img_path}")

        # 原有的影像預處理和模型推論保持不變
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        img_tensor = torch.from_numpy(img_resized).permute(
            2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        # 模型推論
        with torch.no_grad():
            feats, recons = model(img_tensor)
            anomaly_map, _ = cal_anomaly_map([feats[-1]], [recons[-1]],
                                             img_tensor.shape[-1])
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

        # 原有的視覺化輸出保持不變
        ano_map_norm = min_max_norm(anomaly_map) * 255
        ano_map_color = cvt2heatmap(ano_map_norm)
        overlay = show_cam_on_image(img_resized, ano_map_color)
        overlay_path = f"results/{item}_{img_name}_overlay.png"
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # 缺陷提取
        defects = extract_defect_regions(anomaly_map, threshold=0.8)
        for i, d in enumerate(defects):
            print(
                f"🔧 缺陷 {i+1}: 面積={d['area']:.1f}, 中心={d['center']}, 長寬={d['size']}, 深度={d['depth']:.3f}"
            )

        annotated_img = extract_and_annotate_defects(img_resized,
                                                     anomaly_map,
                                                     threshold=0.8)
        annotated_path = f"results/{item}_{img_name}_annotated.png"
        cv2.imwrite(annotated_path,
                    cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        print(f"📌 缺陷標註已儲存 → {annotated_path}")

        # === 改良版折舊分析開始 ===
        # 載入改良版組件
        enhanced_mlp_model = None
        scaler = None

        # 嘗試載入標準化器
        if os.path.exists("feature_scaler.pkl"):
            with open("feature_scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)

        # 嘗試載入改良版模型
        if os.path.exists("enhanced_depreciation_mlp.pth") and scaler:
            try:
                # 檢查 scaler 的特徵數量來決定模型輸入維度
                scaler_features = scaler.n_features_in_
                enhanced_mlp_model = EnhancedDepreciationMLP(
                    input_dim=scaler_features)
                enhanced_mlp_model.load_state_dict(
                    torch.load("enhanced_depreciation_mlp.pth",
                               weights_only=True))
                enhanced_mlp_model.eval()
                print(f"📂 已載入改良版 MLP 模型 (特徵數: {scaler_features})")
            except Exception as e:
                print(f"⚠️ 載入改良版模型失敗: {e}")
                enhanced_mlp_model = None

        # 使用改良版分析
        record = generate_enhanced_depreciation_record(defects,
                                                       enhanced_mlp_model,
                                                       scaler,
                                                       image_shape=(256, 256))
        # #對照組
        recordB = generate_comparison_depreciation_record_with_excel(
            defects, enhanced_mlp_model, scaler, image_shape=(256, 256))
        interpret_comparison_results(recordB)
        # save_record_to_csv(
        #     record, csv_path="generate_comparison_depreciation_record.csv")
        # === 改良版折舊分析結束 ===

        # 儲存記錄
        save_record_to_csv(record)

        # 改良版重訓練條件
        try:
            # 嘗試讀取 CSV，如果失敗則清理後重試
            df = pd.read_csv("depreciation_records.csv")
        except pd.errors.ParserError as e:
            print(f"⚠️ CSV 檔案格式錯誤: {e}")
            print("🔧 嘗試修復 CSV 檔案...")

            # 備份原檔案
            import shutil
            shutil.copy("depreciation_records.csv",
                        "depreciation_records_backup.csv")

            # 重新建立乾淨的 CSV
            clean_csv_file("depreciation_records.csv")

            # 重新讀取
            try:
                df = pd.read_csv("depreciation_records.csv")
                print("✅ CSV 檔案已修復")
            except Exception as e2:
                print(f"❌ 無法修復 CSV 檔案: {e2}")
                # 建立空的 DataFrame 繼續執行
                df = pd.DataFrame()
        # 在讀取 CSV 成功後加入重訓練邏輯
        if not df.empty:
            # 更智能的重訓練條件：至少50筆數據，每20筆重訓練一次
            if len(df) >= 50 and len(df) % 20 == 0:
                print("🔄 觸發改良版 MLP 重訓練...")
                try:
                    #todo
                    enhanced_mlp_model, scaler = train_enhanced_mlp_from_csv()
                    print("✅ 改良版 MLP 重訓練完成")
                except Exception as e:
                    print(f"⚠️ 重訓練失敗: {e}")

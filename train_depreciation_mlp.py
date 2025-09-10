import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json


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
    """"雖然規則式分類在初期很實用，但隨著數據累積，
    機器學習方法通常能提供更好的分類準確性和適應性。
    MyDefectDetection 系統的混合架構設計體現了這種漸進式改進的最佳實踐。"""

    if defect_index < 2000:
        return "A - normal"
    elif defect_index < 3000:
        return "B - Under_observation"
    else:
        return "C - Recommended_repair"


class EnhancedDepreciationMLP(nn.Module):
    """改良版折舊分析 MLP 模型
    - 更深的網路架構（3層隱藏層）
    - Dropout 防過擬合
    - BatchNorm 穩定訓練
    - 支援更多特徵輸入
    """

    def __init__(self,
                 input_dim=8,
                 hidden_dims=[64, 32, 16],
                 output_dim=3,
                 dropout_rate=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 建立多層隱藏層
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # 輸出層
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def compute_enhanced_depreciation_metrics(defects,
                                          image_shape=(256, 256),
                                          use_basic_features_only=True):
    """計算增強版折舊指標，可選擇只使用基礎特徵"""
    # 檢查是否有缺陷資料，若無則回傳零值指標
    if not defects:
        # 建立基礎指標的零值字典
        basic_metrics = {
            "defect_index": 0,
            "defect_count": 0,
            "avg_depth": 0,
            "max_depth": 0,
            "total_area": 0
        }
        # 根據參數決定是否只回傳基礎特徵
        if use_basic_features_only:
            return basic_metrics
        else:
            # 回傳包含額外特徵的完整零值字典
            return {
                **basic_metrics, "depth_std": 0,
                "area_ratio": 0,
                "defect_density": 0
            }

    # 基礎指標計算保持不變 - 計算核心折舊指標
    defect_index = sum([d['area'] * d['depth']
                        for d in defects])  # 計算折舊指數（面積×深度的總和）
    depths = [d['depth'] for d in defects]  # 提取所有缺陷的深度值
    areas = [d['area'] for d in defects]  # 提取所有缺陷的面積值

    avg_depth = np.mean(depths)  # 計算平均深度
    max_depth = np.max(depths)  # 找出最大深度
    total_area = sum(areas)  # 計算總面積

    # 組建基礎指標字典
    basic_metrics = {
        "defect_index": defect_index,  # 折舊指數
        "defect_count": len(defects),  # 缺陷數量
        "avg_depth": avg_depth,  # 平均深度
        "max_depth": max_depth,  # 最大深度
        "total_area": total_area  # 總面積
    }

    # 如果只需要基礎特徵，直接回傳
    if use_basic_features_only:
        return basic_metrics

    # 額外特徵 - 計算進階統計特徵
    depth_std = np.std(depths) if len(depths) > 1 else 0  # 計算深度標準差（需多於1個缺陷）
    total_image_area = image_shape[0] * image_shape[1]  # 計算影像總面積
    area_ratio = total_area / total_image_area  # 計算面積比例（缺陷面積/影像面積）
    defect_density = len(
        defects) / total_image_area * 10000  # 計算缺陷密度（每萬像素的缺陷數）

    # 回傳包含所有特徵的完整字典
    return {
        **basic_metrics,  # 展開基礎指標
        "depth_std": depth_std,  # 深度標準差
        "area_ratio": area_ratio,  # 面積比例
        "defect_density": defect_density  # 缺陷密度
    }


def train_enhanced_mlp_from_csv(csv_path="depreciation_records.csv",
                                output_path="enhanced_depreciation_mlp.pth",
                                scaler_path="feature_scaler.pkl",
                                epochs=100,
                                patience=10):
    """改良版 MLP 訓練函數
    - 數據標準化
    - 訓練/驗證分割
    - 早停機制
    - 學習率調度
    - 類別權重平衡
    """

    # 讀取數據 - 從 CSV 檔案載入折舊分析記錄
    df = pd.read_csv(csv_path)

    # 特徵選擇（支援更多特徵）- 先設定基礎特徵集
    feature_cols = ["defect_index", "avg_depth", "max_depth", "total_area"]
    # 檢查是否有額外特徵欄位，若有則擴展特徵集
    if "depth_std" in df.columns:
        feature_cols.extend(["depth_std", "area_ratio", "defect_density"])

    # 提取特徵矩陣和標籤向量
    X = df[feature_cols].values  # 特徵數據
    # 將等級字串映射為數值標籤 (A=0, B=1, C=2)
    y = df["grade"].map({
        "A - normal": 0,
        "B - Under_observation": 1,
        "C - Recommended_repair": 2
    }).values

    # 數據標準化 - 建立標準化器並對特徵進行標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # 計算均值和標準差並標準化

    # 儲存標準化器 - 保存以供後續推論使用
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # 訓練/驗證分割 - 分層抽樣確保各類別比例一致
    X_train, X_val, y_train, y_val = train_test_split(X_scaled,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=42,
                                                      stratify=y)

    # 轉換為張量 - 將 NumPy 陣列轉換為 PyTorch 張量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # 訓練特徵
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # 訓練標籤
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)  # 驗證特徵
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)  # 驗證標籤

    # 計算類別權重 - 處理類別不平衡問題
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # 建立模型 - 根據特徵數量設定輸入維度
    model = EnhancedDepreciationMLP(input_dim=len(feature_cols))

    # 載入已存在的模型（如果有）- 支援增量訓練
    if os.path.exists(output_path):
        print(f"📂 載入已存在模型 {output_path}")
        model.load_state_dict(torch.load(output_path, weights_only=True))

    # 訓練設定 - 配置損失函數、優化器和學習率調度器
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # 加權交叉熵損失
    optimizer = optim.AdamW(model.parameters(), lr=0.001,
                            weight_decay=0.01)  # AdamW 優化器
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=5,
                                  verbose=True)  # 學習率調度

    # 早停設定 - 防止過擬合
    best_val_loss = float('inf')  # 記錄最佳驗證損失
    patience_counter = 0  # 耐心計數器
    best_model_state = None  # 最佳模型狀態

    # 訓練迴圈 - 建立數據載入器並開始訓練
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor,
                                                   y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               shuffle=True)

    model.train()  # 設定為訓練模式
    for epoch in range(epochs):
        # 訓練階段 - 批次訓練
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()  # 清零梯度
            logits = model(batch_x)  # 前向傳播
            loss = criterion(logits, batch_y)  # 計算損失
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新參數
            train_loss += loss.item()  # 累積損失

        # 驗證階段 - 評估模型性能
        model.eval()  # 設定為評估模式
        with torch.no_grad():  # 停用梯度計算
            val_logits = model(X_val_tensor)  # 驗證集前向傳播
            val_loss = criterion(val_logits, y_val_tensor).item()  # 計算驗證損失
            val_acc = (torch.argmax(
                val_logits,
                dim=1) == y_val_tensor).float().mean().item()  # 計算準確率

        # 學習率調度 - 根據驗證損失調整學習率
        scheduler.step(val_loss)

        # 早停檢查 - 監控驗證損失改善情況
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # 更新最佳損失
            patience_counter = 0  # 重置耐心計數器
            best_model_state = model.state_dict().copy()  # 保存最佳模型狀態
        else:
            patience_counter += 1  # 增加耐心計數器

        # 每 10 個 epoch 輸出訓練進度
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # 早停條件檢查 - 若驗證損失長時間未改善則停止訓練
        if patience_counter >= patience:
            print(f"早停於 epoch {epoch}")
            break

        model.train()  # 回到訓練模式

    # 載入最佳模型 - 恢復驗證損失最低時的模型狀態
    if best_model_state:
        model.load_state_dict(best_model_state)

    # 儲存模型 - 保存訓練完成的模型權重
    torch.save(model.state_dict(), output_path)
    print(f"✅ 改良版模型已訓練並儲存至 {output_path}")

    # 回傳訓練完成的模型和標準化器
    return model, scaler


def classify_depreciation_enhanced_mlp(metrics, mlp_model, scaler):
    """使用改良版 MLP 模型進行分類，自動適應特徵數量"""
    # 根據 scaler 的特徵數量決定使用哪些特徵 - 檢查標準化器支援的特徵維度
    scaler_features = scaler.n_features_in_

    # 根據標準化器的特徵數量選擇對應的特徵集
    if scaler_features == 4:
        # 只使用基礎特徵 - 4個核心指標
        features = [
            metrics["defect_index"],
            metrics["avg_depth"],  # 折舊指數和平均深度
            metrics["max_depth"],
            metrics["total_area"]  # 最大深度和總面積
        ]
    else:
        # 使用所有特徵 - 包含基礎特徵和額外統計特徵
        features = [
            metrics["defect_index"],
            metrics["avg_depth"],  # 基礎特徵：折舊指數和平均深度
            metrics["max_depth"],
            metrics["total_area"],  # 基礎特徵：最大深度和總面積
            metrics.get("depth_std", 0),
            metrics.get("area_ratio", 0),  # 額外特徵：深度標準差和面積比例
            metrics.get("defect_density", 0)  # 額外特徵：缺陷密度
        ]

    # 其餘邏輯保持不變 - 特徵標準化和模型推論
    features_scaled = scaler.transform([features])  # 使用標準化器對特徵進行標準化
    input_tensor = torch.tensor(features_scaled,
                                dtype=torch.float32)  # 轉換為 PyTorch 張量

    # 停用梯度計算進行推論
    with torch.no_grad():
        """Logits 轉換為機率分布：使用 torch.softmax(logits, dim=1) 將模型的原始輸出 (logits) 轉換為機率分布，確保所有類別的機率總和為 1
           取得預測類別：使用 torch.argmax(probs, dim=1).item() 找出機率最高的類別索引
           提取信心度：confidence = probs[0, pred].item() 取得該預測類別的機率值作為信心度
           例如，如果機率分布為 [0.12, 0.82, 0.06]，預測類別為索引 1 (B級)，則信心度為 0.82"""
        logits = mlp_model(input_tensor)  # 模型前向傳播，取得原始輸出
        probs = torch.softmax(logits, dim=1)  # 將 logits 轉換為機率分布
        pred = torch.argmax(probs, dim=1).item()  # 取得預測類別索引
        confidence = probs[0, pred].item()  # 取得預測類別的信心度

        # 計算預測不確定性（基於熵）
        """不確定性基於資訊熵 (entropy) 來衡量模型預測的不確定程度：
            計算熵值：entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()
            使用資訊熵公式：H = -Σ(p_i * log(p_i))
            加入 1e-8 避免 log(0) 的數值錯誤
            正規化不確定性：uncertainty = entropy / np.log(3)
            除以最大可能熵值 (log(3)，因為有 3 個類別) 進行正規化
            結果範圍在 0-1 之間，1 表示最大不確定性 """
        entropy = -torch.sum(probs * torch.log(probs + 1e-8),
                             dim=1).item()  # 計算熵值
        uncertainty = entropy / np.log(3)  # 正規化不確定性（除以最大熵值）

    # 類別標籤對應
    labels = ["A - normal", "B - Under_observation", "C - Recommended_repair"]
    # 回傳預測等級、信心度和不確定性
    return labels[pred], confidence, uncertainty


def generate_enhanced_depreciation_record(defects,
                                          mlp_model=None,
                                          scaler=None,
                                          image_shape=(256, 256)):
    """生成增強版折舊分析記錄，自動適應特徵數量
    當有訓練好的 MLP 模型和標準化器時使用機器學習方法，否則回退到基於規則的分類。
    函數會根據 scaler.n_features_in_ 自動適應使用 4 個基礎特徵或 7 個完整特徵集，確保與不同版本的訓練模型相容。
    defect_index - 折舊指數（面積 × 深度的加總）
    avg_depth - 平均深度
    max_depth - 最大深度
    total_area - 所有缺陷的總面積
    額外:
    depth_std - 深度標準差（衡量深度變異性）
    area_ratio - 面積比例（缺陷總面積 / 影像總面積）
    defect_density - 缺陷密度（每萬像素的缺陷數量）
    """
    # 根據是否有 scaler 決定特徵類型 - 判斷是否使用基礎特徵或完整特徵集
    use_basic_only = scaler is None or scaler.n_features_in_ == 4
    # use_basic_only = False #要手動改成 False 才會用到完整特徵
    print("use_basic_only:", use_basic_only)
    # 計算增強版折舊指標，根據 use_basic_only 決定特徵數量
    metrics = compute_enhanced_depreciation_metrics(
        defects, image_shape, use_basic_features_only=use_basic_only)
    # 如果有 MLP 模型和標準化器，使用機器學習分類
    if mlp_model and scaler:
        # 使用增強版 MLP 模型進行分類，回傳等級、信心度和不確定性
        grade, confidence, uncertainty = classify_depreciation_enhanced_mlp(
            metrics, mlp_model, scaler)
    else:
        # 否則使用規則式分類方法
        from train_depreciation_mlp import classify_depreciation
        # 僅使用折舊指數進行簡單分級
        grade = classify_depreciation(metrics["defect_index"])
        # 規則式方法無法提供信心度和不確定性
        confidence = "N/A"
        uncertainty = "N/A"

    # 產生當前時間戳記，格式為 YYYY-MM-DD HH:MM
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    # 回傳完整的折舊分析記錄字典
    return {
        "timestamp": timestamp,  # 分析時間
        "grade": grade,  # 折舊等級 (A/B/C)
        "confidence": confidence,  # 模型信心度
        "uncertainty": uncertainty,  # 預測不確定性
        **metrics,  # 展開所有計算的指標
        "defects": defects  # 原始缺陷清單
    }


def generate_comparison_depreciation_record_with_excel(defects,
                                                       mlp_model=None,
                                                       scaler=None,
                                                       image_shape=(256, 256),
                                                       output_excel=True):
    """生成對照實驗的折舊分析記錄，同時測試 MLP 和規則式分類

    此函數會同時運行兩種分類方法並記錄結果差異：
    1. MLP 機器學習分類（如果模型可用）
    2. 規則式閾值分類

    返回包含兩種方法結果的完整比較記錄 生成對照實驗記錄並輸出 Excel 檔案
    """

    from train_depreciation_mlp import classify_depreciation
    from datetime import datetime
    import pandas as pd
    import os

    # 根據是否有 scaler 決定特徵類型
    use_basic_only = scaler is None or scaler.n_features_in_ == 4

    # 計算增強版折舊指標
    metrics = compute_enhanced_depreciation_metrics(
        defects, image_shape, use_basic_features_only=use_basic_only)

    # === 方法1：MLP 機器學習分類 ===
    mlp_result = {}
    if mlp_model and scaler:
        try:
            grade_mlp, confidence_mlp, uncertainty_mlp = classify_depreciation_enhanced_mlp(
                metrics, mlp_model, scaler)
            mlp_result = {
                "grade": grade_mlp,
                "confidence": confidence_mlp,
                "uncertainty": uncertainty_mlp,
                "method": "MLP",
                "available": True
            }
            print(
                f"📊 MLP 分析 - 等級: {grade_mlp}, 信心: {confidence_mlp:.3f}, 不確定性: {uncertainty_mlp:.3f}"
            )
        except Exception as e:
            mlp_result = {
                "grade": "ERROR",
                "confidence": "N/A",
                "uncertainty": "N/A",
                "method": "MLP",
                "available": False,
                "error": str(e)
            }
            print(f"⚠️ MLP 分析失敗: {e}")
    else:
        mlp_result = {
            "grade": "N/A",
            "confidence": "N/A",
            "uncertainty": "N/A",
            "method": "MLP",
            "available": False,
            "reason": "模型或標準化器不可用"
        }
        print("⚠️ MLP 模型不可用")

    # === 方法2：規則式閾值分類 ===
    try:
        grade_rule = classify_depreciation(metrics["defect_index"])
        rule_result = {
            "grade": grade_rule,
            "confidence": "N/A",
            "uncertainty": "N/A",
            "method": "Rule-based",
            "available": True,
            "threshold_used": "3876/5554"
        }
        print(f"📊 規則式分析 - 等級: {grade_rule}")
    except Exception as e:
        rule_result = {
            "grade": "ERROR",
            "confidence": "N/A",
            "uncertainty": "N/A",
            "method": "Rule-based",
            "available": False,
            "error": str(e)
        }
        print(f"⚠️ 規則式分析失敗: {e}")

    # === 結果比較分析 ===
    comparison = {
        "methods_agree":
        mlp_result["grade"] == rule_result["grade"]
        if mlp_result["available"] and rule_result["available"] else None,
        "grade_difference":
        None,
        "confidence_level":
        mlp_result.get("confidence", "N/A")
    }

    if mlp_result["available"] and rule_result["available"]:
        if mlp_result["grade"] != rule_result["grade"]:
            comparison[
                "grade_difference"] = f"{mlp_result['grade']} vs {rule_result['grade']}"
            print(
                f"⚠️ 分類結果不一致: MLP={mlp_result['grade']}, 規則式={rule_result['grade']}"
            )
        else:
            print(f"✅ 兩種方法結果一致: {mlp_result['grade']}")

    # 產生時間戳記
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 回傳完整的對照實驗記錄
    record = {
        "timestamp":
        timestamp,
        "experiment_type":
        "comparison",

        # 主要結果
        "grade":
        mlp_result["grade"]
        if mlp_result["available"] else rule_result["grade"],
        "confidence":
        mlp_result["confidence"],
        "uncertainty":
        mlp_result["uncertainty"],

        # MLP 方法結果
        "mlp_grade":
        mlp_result["grade"],
        "mlp_confidence":
        mlp_result["confidence"],
        "mlp_uncertainty":
        mlp_result["uncertainty"],
        "mlp_available":
        mlp_result["available"],

        # 規則式方法結果
        "rule_grade":
        rule_result["grade"],
        "rule_available":
        rule_result["available"],
        "rule_threshold":
        rule_result.get("threshold_used", "N/A"),

        # 比較分析
        "methods_agree":
        comparison["methods_agree"],
        "grade_difference":
        comparison["grade_difference"],

        # 原始指標和缺陷數據
        **metrics,
        "defects":
        defects
    }

    # 儲存到 CSV（使用現有函數）
    save_record_to_csv(record, "comparison_records.csv")

    # # 產出 Excel 檔案
    # if output_excel:
    #     save_comparison_to_excel(record, "comparison_analysis.xlsx")

    return record


def interpret_comparison_results(record):
    """解釋對照實驗結果"""

    print(f"📊 對照實驗結果解釋 ({record['timestamp']})")
    print("=" * 50)

    # 1. 基本資訊
    print(f"折舊指數: {record['defect_index']:.2f}")
    print(f"缺陷數量: {record['defect_count']}")
    print(f"總面積: {record['total_area']:.1f}")

    # 2. 方法比較
    mlp_grade = record['mlp_grade']
    rule_grade = record['rule_grade']
    methods_agree = record['methods_agree']

    if methods_agree:
        print(f"✅ 兩種方法結果一致: {mlp_grade}")
        print("   → 分類結果可信度高")
    elif methods_agree is False:
        print(f"⚠️ 分類結果不一致:")
        print(f"   MLP 預測: {mlp_grade}")
        print(f"   規則式: {rule_grade}")

        # 3. 信心度分析
        confidence = record['mlp_confidence']
        if isinstance(confidence, (int, float)) and confidence >= 0.7:
            print(f"   → MLP 高信心度 ({confidence:.3f})，建議採用 MLP 結果")
        else:
            print(f"   → MLP 信心度較低 ({confidence})，建議人工複核")

    # 4. 不確定性評估
    uncertainty = record['mlp_uncertainty']
    if isinstance(uncertainty, (int, float)):
        if uncertainty < 0.3:
            print(f"📈 低不確定性 ({uncertainty:.3f})：預測穩定")
        elif uncertainty > 0.7:
            print(f"📉 高不確定性 ({uncertainty:.3f})：建議謹慎處理")

    # 5. 建議行動
    print("\n💡 建議行動:")
    if methods_agree and isinstance(
            record['mlp_confidence'],
        (int, float)) and record['mlp_confidence'] >= 0.8:
        print("   → 可自動通過，無需人工介入")
    elif not methods_agree or (isinstance(record['mlp_confidence'],
                                          (int, float))
                               and record['mlp_confidence'] < 0.6):
        print("   → 建議人工複核確認")
    else:
        print("   → 可接受結果，但建議定期抽查")


def save_record_to_csv(record, csv_path="depreciation_records.csv"):
    """改良版 CSV 儲存函數，確保數據格式正確並處理 NumPy 類型"""
    record_copy = record.copy()

    if 'defects' in record_copy:
        # 轉換 NumPy 類型為 Python 原生類型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # 清理 defects 數據並轉為 JSON
        cleaned_defects = convert_numpy_types(record_copy['defects'])
        record_copy['defects'] = json.dumps(cleaned_defects)

    # 同時清理其他可能的 NumPy 類型
    for key, value in record_copy.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            record_copy[key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            record_copy[key] = float(value)

    df = pd.DataFrame([record_copy])

    try:
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)
        print(f"✅ 記錄已儲存至 {csv_path}")
    except Exception as e:
        print(f"⚠️ 儲存 CSV 失敗: {e}")


#邊界效應分析函數
def analyze_boundary_effects(record, boundary_tolerance=200):
    """分析接近閾值邊界的案例表現"""
    defect_index = record['defect_index']

    # 定義邊界區間
    ab_boundary = 3876
    bc_boundary = 5554

    boundary_analysis = {
        "is_boundary_case": False,
        "boundary_type": None,
        "distance_to_boundary": None,
        "risk_level": "normal"
    }

    # 檢查是否接近 A/B 邊界
    if abs(defect_index - ab_boundary) <= boundary_tolerance:
        boundary_analysis.update({
            "is_boundary_case":
            True,
            "boundary_type":
            "A/B",
            "distance_to_boundary":
            abs(defect_index - ab_boundary),
            "risk_level":
            "high" if abs(defect_index - ab_boundary) < 100 else "medium"
        })

    # 檢查是否接近 B/C 邊界
    elif abs(defect_index - bc_boundary) <= boundary_tolerance:
        boundary_analysis.update({
            "is_boundary_case":
            True,
            "boundary_type":
            "B/C",
            "distance_to_boundary":
            abs(defect_index - bc_boundary),
            "risk_level":
            "high" if abs(defect_index - bc_boundary) < 100 else "medium"
        })

    return boundary_analysis
